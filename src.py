
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from scipy import linalg as sla
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

@dataclass
class PlateInput:
    a: float; b: float; t: float; E: float; nu: float; unit: str
    edge_top: str; edge_bottom: str; edge_left: str; edge_right: str
    kr_top: float; kr_bottom: float; kr_left: float; kr_right: float
    J_top: float; J_bottom: float; J_left: float; J_right: float
    beta_x: float; eta_x: float; beta_y: float; eta_y: float
    stiffeners: pd.DataFrame
    s_xtl: float; s_xbl: float; s_xtr: float; s_xbr: float
    s_yut: float; s_yub: float; s_ypt: float; s_ypb: float
    c_t: float; c_b: float; tau_u: float
    imposed_x: bool; imposed_y: bool; imposed_tau: bool
    patch_with_flanges: bool; mesh_df: pd.DataFrame | None
    complexity: int; search_mode: str; plate_behaviour: bool

def _f(unit): return {'mm':1.0,'cm':10.0,'m':1000.0}[unit]
def _mm(v, unit): return float(v) * _f(unit)
def _mm2(v, unit): return float(v) * _f(unit)**2
def _mm4(v, unit): return float(v) * _f(unit)**4
def build_plate_input(**kwargs): return PlateInput(**kwargs)

def default_stiffeners_df():
    return pd.DataFrame([{
        'active': False, 'closed_section': False, 'orientation': 'longitudinale', 'type': 'trapezoid', 'location': 750.0,
        'A': 0.0, 'I': 0.0, 'J': 0.0, 'b1': 130.0, 'b2': 85.0, 'h': 130.0, 'tf': 9.0, 'tw': 9.0, 'ts': 9.0, 'd': 120.0,
        'Kr_local': 0.0,
    }])

def clean_stiffeners_df(df):
    if df is None or len(df)==0: return default_stiffeners_df().iloc[0:0].copy()
    out=df.copy(); cols=['active','closed_section','orientation','type','location','A','I','J','b1','b2','h','tf','tw','ts','d','Kr_local']
    for c in cols:
        if c not in out.columns: out[c]=0.0
    out['active']=out['active'].fillna(False).astype(bool)
    out['closed_section']=out['closed_section'].fillna(False).astype(bool)
    out['orientation']=out['orientation'].fillna('longitudinale'); out['type']=out['type'].fillna('general')
    for c in [c for c in cols if c not in ['active','closed_section','orientation','type']]: out[c]=pd.to_numeric(out[c], errors='coerce').fillna(0.0)
    return out

def _estimate_band_mm(r, t_mm):
    typ=r.get('type','general')
    if typ=='trapezoid': base=max(float(r.get('b1',0.0)), float(r.get('d',0.0)), 8.0*t_mm)
    elif typ in ('flat bar','sym flat bar'): base=max(float(r.get('h',0.0)), 6.0*t_mm)
    elif typ in ('T','angle'): base=max(float(r.get('b1',0.0)), float(r.get('h',0.0)), 6.0*t_mm)
    elif typ=='closed box': base=max(float(r.get('b1',0.0)), float(r.get('h',0.0)), 8.0*t_mm)
    else: base=max(float(r.get('d',0.0)), 6.0*t_mm)
    return max(base, 6.0*t_mm)

def _profile_membrane_factor(ptype):
    return {'general':1.00,'flat bar':1.05,'sym flat bar':1.10,'T':1.18,'angle':1.12,'trapezoid':1.22,'closed box':1.28}.get(ptype,1.00)

def _profile_bending_factor(ptype):
    return {'general':1.00,'flat bar':1.10,'sym flat bar':1.18,'T':1.28,'angle':1.18,'trapezoid':1.35,'closed box':1.45}.get(ptype,1.00)

def compute_stiffener_properties(df, t, unit, E=210000.0, nu=0.30):
    out=clean_stiffeners_df(df)
    if out.empty: return out
    tp=_mm(t,unit); Dp=E*tp**3/(12*(1-nu**2)); rows=[]
    for _,r in out.iterrows():
        typ=r['type']; closed=bool(r.get('closed_section',False)) or typ=='closed box'
        b1=_mm(r.get('b1',0.0),unit); b2=_mm(r.get('b2',0.0),unit); h=_mm(r.get('h',0.0),unit)
        tf=max(_mm(r.get('tf',0.0),unit),1e-9); tw=max(_mm(r.get('tw',0.0),unit),1e-9); ts=max(_mm(r.get('ts',0.0),unit),1e-9); d=_mm(r.get('d',0.0),unit)
        leff=15.0*tp
        if typ=='flat bar': A=h*tf; I=tf*h**3/12.0 + 2*leff*tp**3/12.0; J=h*tf**3/3.0
        elif typ=='sym flat bar': A=2*h*tf; I=2*(tf*h**3/12.0)+2*leff*tp**3/12.0; J=2*h*tf**3/3.0
        elif typ=='T': A=b1*tf+h*tw; I=b1*tf**3/12.0+tw*h**3/12.0+2*leff*tp**3/12.0; J=b1*tf**3/3.0+h*tw**3/3.0
        elif typ=='angle': A=b1*tf+h*tw-tw*tf; I=b1*tf**3/12.0+tw*h**3/12.0+2*leff*tp**3/12.0; J=b1*tf**3/3.0+h*tw**3/3.0
        elif typ=='closed box':
            width=max(b1,1e-9); A=2*ts*(width+h); I=2*(ts*h**3/12.0)+2*(width*ts*(h/2.0)**2)+2*(width*ts**3/12.0); per=2*(width+h); Acell=width*h; J=4*Acell**2*ts/max(per,1e-9)
        elif typ=='trapezoid':
            A=(b2+2*h)*ts; I=b2*ts**3/12.0+2*(ts*h**3/12.0)+2*leff*tp**3/12.0; per=max(2*h+b1+b2,1.0); Acell=max((b1+b2)*h/2.0,1.0)
            J=4*Acell**2*ts/per if closed else (2*h+b2)*ts**3/3.0
        else:
            A=_mm2(r.get('A',0.0),unit); I=_mm4(r.get('I',0.0),unit); J=_mm4(r.get('J',0.0),unit)
        gamma=E*I/max(1000.0*Dp,1e-9); theta=(E/(2*(1+nu)))*J/max(1000.0*Dp,1e-9); delta=A/max(tp*1000.0,1e-9)
        bw=_estimate_band_mm({'type':typ,'b1':b1,'d':d,'h':h}, tp)
        t_eq_mem = tp + _profile_membrane_factor(typ) * A / max(bw,1e-9)
        D_add = _profile_bending_factor(typ) * E * I / max(bw,1e-9)
        t_eq_bend = max((tp**3 + 12*(1-nu**2)*D_add/max(E,1e-9))**(1/3), tp)
        rr=dict(r)
        rr.update({'closed_section':closed,'A_eff':A,'I_eff':I,'J_eff':J,'delta':delta,'gamma':gamma,'theta':theta,'band_width':bw/_f(unit),'t_eq_mem':t_eq_mem/_f(unit),'t_eq_bend':t_eq_bend/_f(unit),'Kr_local':float(r.get('Kr_local',0.0)),'Jt_local':0.0})
        rows.append(rr)
    return pd.DataFrame(rows)

def orthotropy_from_smearing(df, a, b, t, unit):
    f=_f(unit); a_mm=a*f; b_mm=b*f; t_mm=t*f; active=df[df['active']]
    longi=active[active['orientation']=='longitudinale']; trans=active[active['orientation']=='trasversale']; bx=ex=by=ey=0.0
    if len(longi)>0:
        I=float(longi['I_eff'].mean()); A=float(longi['A_eff'].mean()); ds=b_mm/(len(longi)+1); bx=12*(1-0.3**2)*I/max(ds*t_mm**3,1e-9); ex=A/max(ds*t_mm,1e-9)
    if len(trans)>0:
        I=float(trans['I_eff'].mean()); A=float(trans['A_eff'].mean()); ds=a_mm/(len(trans)+1); by=12*(1-0.3**2)*I/max(ds*t_mm**3,1e-9); ey=A/max(ds*t_mm,1e-9)
    return bx,ex,by,ey

def make_stiffener_summary_df(df):
    if df is None or len(df)==0: return pd.DataFrame(columns=['active','orientation','type','location'])
    cols=[c for c in ['active','closed_section','orientation','type','location','band_width','t_eq_mem','t_eq_bend','gamma','theta'] if c in df.columns]
    return df[cols].copy()

def _mesh_fun(inp, col):
    if inp.mesh_df is None or col not in inp.mesh_df.columns: return lambda x,y: 0.0
    df=inp.mesh_df.copy(); xs=pd.to_numeric(df['x'],errors='coerce').fillna(0.0).to_numpy(); ys=pd.to_numeric(df['y'],errors='coerce').fillna(0.0).to_numpy(); vals=pd.to_numeric(df[col],errors='coerce').fillna(0.0).to_numpy(); a=_mm(inp.a,inp.unit); b=_mm(inp.b,inp.unit)
    if len(xs)==0 or len(ys)==0: return lambda x,y: 0.0
    if xs.max()<=1.0+1e-9: xs=xs*a
    if ys.max()<=1.0+1e-9: ys=ys*b
    def f(x,y):
        d=(xs-x)**2+(ys-y)**2
        return float(vals[int(np.argmin(d))])
    return f

def analytical_stress_functions(inp):
    a=_mm(inp.a,inp.unit); b=_mm(inp.b,inp.unit); sx_mesh=_mesh_fun(inp,'sigma_x'); sy_mesh=_mesh_fun(inp,'sigma_y'); tau_mesh=_mesh_fun(inp,'tau')
    def sx(x,y):
        xr=x/max(a,1e-9); yr=y/max(b,1e-9); top=inp.s_xtl*(1-xr)+inp.s_xtr*xr; bot=inp.s_xbl*(1-xr)+inp.s_xbr*xr
        return top*(1-yr)+bot*yr+sx_mesh(x,y)
    def patch(x,c,s):
        if c<=0 or s==0: return 0.0
        x0=a/2-c/2; x1=a/2+c/2
        return float(s) if x0<=x<=x1 else 0.0
    def sy(x,y):
        yr=y/max(b,1e-9); uni=inp.s_yut*(1-yr)+inp.s_yub*yr
        loc=patch(x,_mm(inp.c_t,inp.unit),inp.s_ypt)*(1-yr)+patch(x,_mm(inp.c_b,inp.unit),inp.s_ypb)*yr
        return uni+loc+sy_mesh(x,y)
    def tau(x,y):
        base=inp.tau_u+tau_mesh(x,y)
        if inp.s_ypt==0 and inp.s_ypb==0: return base
        ctm=_mm(inp.c_t,inp.unit); cbm=_mm(inp.c_b,inp.unit)
        if inp.patch_with_flanges:
            tau_t=inp.s_ypt*ctm/max(2*b,1e-9); tau_b=inp.s_ypb*cbm/max(2*b,1e-9)
        else:
            par=4*(y/b)*(1-y/b); tau_t=inp.s_ypt*ctm/max(2*b,1e-9)*par; tau_b=inp.s_ypb*cbm/max(2*b,1e-9)*par
        return base+tau_t+tau_b
    return sx,sy,tau

def _terms(inp): return (10,10) if inp.complexity==1 else (20,20) if inp.complexity==2 else (30,30)
def _edge_penalty(kind,kr): return 1e8 if kind=='Fisso' else max(float(kr),0.0) if kind=='Elastico' else 0.0
def _basis(m,n): return [(i,j) for i in range(1,m+1) for j in range(1,n+1)]
def _shape_x(m,n,x,y,a,b): return (m*np.pi/a)*np.cos(m*np.pi*x/a)*np.sin(n*np.pi*y/b)
def _shape_y(m,n,x,y,a,b): return (n*np.pi/b)*np.sin(m*np.pi*x/a)*np.cos(n*np.pi*y/b)
def _shape_xx(m,n,x,y,a,b): return -((m*np.pi/a)**2)*np.sin(m*np.pi*x/a)*np.sin(n*np.pi*y/b)
def _shape_yy(m,n,x,y,a,b): return -((n*np.pi/b)**2)*np.sin(m*np.pi*x/a)*np.sin(n*np.pi*y/b)
def _shape_xy(m,n,x,y,a,b): return (m*np.pi/a)*(n*np.pi/b)*np.cos(m*np.pi*x/a)*np.cos(n*np.pi*y/b)
def _shape(m,n,x,y,a,b): return np.sin(m*np.pi*x/a)*np.sin(n*np.pi*y/b)

def _integrate_plate(inp, fun):
    a=_mm(inp.a,inp.unit); b=_mm(inp.b,inp.unit); nx=35 if inp.complexity==1 else 45 if inp.complexity==2 else 55; ny=25 if inp.complexity==1 else 35 if inp.complexity==2 else 45
    xs=np.linspace(0,a,nx); ys=np.linspace(0,b,ny); X,Y=np.meshgrid(xs,ys,indexing='xy'); V=fun(X,Y)
    return np.trapezoid(np.trapezoid(V,xs,axis=1),ys,axis=0)

def solve_buckling_problem(inp):
    a=_mm(inp.a,inp.unit); b=_mm(inp.b,inp.unit); t=_mm(inp.t,inp.unit); D=inp.E*t**3/(12*(1-inp.nu**2)); Dx=D*(1+inp.beta_x); Dy=D*(1+inp.beta_y)
    sx,sy,tau=analytical_stress_functions(inp); fact_x=fact_y=fact_t=1.0; basis=_basis(*_terms(inp)); N=len(basis); R0=np.zeros((N,N)); RG=np.zeros((N,N))
    kp={'top':_edge_penalty(inp.edge_top,inp.kr_top),'bottom':_edge_penalty(inp.edge_bottom,inp.kr_bottom),'left':_edge_penalty(inp.edge_left,inp.kr_left),'right':_edge_penalty(inp.edge_right,inp.kr_right)}
    for i,(mi,ni) in enumerate(basis):
        for j,(mj,nj) in enumerate(basis):
            def f0(X,Y):
                wxx_i=_shape_xx(mi,ni,X,Y,a,b); wyy_i=_shape_yy(mi,ni,X,Y,a,b); wxy_i=_shape_xy(mi,ni,X,Y,a,b); wxx_j=_shape_xx(mj,nj,X,Y,a,b); wyy_j=_shape_yy(mj,nj,X,Y,a,b); wxy_j=_shape_xy(mj,nj,X,Y,a,b)
                return Dx*wxx_i*wxx_j + Dy*wyy_i*wyy_j + 2*D*inp.nu*wxx_i*wyy_j + 2*D*(1-inp.nu)*wxy_i*wxy_j
            def fg(X,Y):
                wx_i=_shape_x(mi,ni,X,Y,a,b); wy_i=_shape_y(mi,ni,X,Y,a,b); wx_j=_shape_x(mj,nj,X,Y,a,b); wy_j=_shape_y(mj,nj,X,Y,a,b)
                return t*(fact_x*np.vectorize(sx)(X,Y)*wx_i*wx_j + fact_y*np.vectorize(sy)(X,Y)*wy_i*wy_j + 2*fact_t*np.vectorize(tau)(X,Y)*wx_i*wy_j)
            R0[i,j]=_integrate_plate(inp,f0); RG[i,j]=_integrate_plate(inp,fg)
            xs=np.linspace(0,a,100); ys=np.linspace(0,b,100)
            if kp['top']>0: R0[i,j]+=np.trapezoid(kp['top']*_shape_y(mi,ni,xs,0.0,a,b)*_shape_y(mj,nj,xs,0.0,a,b),xs)
            if kp['bottom']>0: R0[i,j]+=np.trapezoid(kp['bottom']*_shape_y(mi,ni,xs,b,a,b)*_shape_y(mj,nj,xs,b,a,b),xs)
            if kp['left']>0: R0[i,j]+=np.trapezoid(kp['left']*_shape_x(mi,ni,0.0,ys,a,b)*_shape_x(mj,nj,0.0,ys,a,b),ys)
            if kp['right']>0: R0[i,j]+=np.trapezoid(kp['right']*_shape_x(mi,ni,a,ys,a,b)*_shape_x(mj,nj,a,ys,a,b),ys)
    R0=0.5*(R0+R0.T); RG=0.5*(RG+RG.T); R0=R0+np.eye(N)*max(np.linalg.norm(R0,ord='fro')*1e-12,1e-9)
    eigvals,eigvecs=(sla.eig(R0,RG+np.eye(N)*1e-12) if SCIPY_OK else np.linalg.eig(np.linalg.pinv(RG+np.eye(N)*1e-12)@R0))
    eigvals=np.real(eigvals); eigvecs=np.real(eigvecs); mask=np.isfinite(eigvals)&(eigvals>1e-8); pos=eigvals[mask]; vec=eigvecs[:,mask]; order=np.argsort(pos); pos=pos[order]; vec=vec[:,order] if vec.size else vec
    nm=1 if inp.search_mode=='1° modo' else 20 if inp.search_mode=='primi 20' else len(pos); pos=pos[:nm]
    eig_list=[]
    for k in range(min(nm, vec.shape[1] if vec.ndim==2 else 0)):
        v=vec[:,k]; vmax=np.max(np.abs(v)) if np.max(np.abs(v))>0 else 1.0; eig_list.append(v/vmax)
    phi=float(pos[0]) if len(pos) else float('nan'); sxr=max(abs(inp.s_xtl),abs(inp.s_xbl),abs(inp.s_xtr),abs(inp.s_xbr),1e-9); syr=max(abs(inp.s_yut),abs(inp.s_yub),abs(inp.s_ypt),abs(inp.s_ypb),1e-9); taur=max(abs(inp.tau_u),1e-9)
    return {'phi_cr':phi,'phi_positive':pos,'eigenvectors':eig_list,'basis_modes':basis,'a_mm':a,'b_mm':b,'sigma_x_cr':phi*sxr,'sigma_y_cr':phi*syr,'tau_cr':phi*taur,'modes_df':pd.DataFrame({'Modo':np.arange(1,len(pos)+1),'phi':pos}), 'calc_log':pd.DataFrame([('Dimensione matrici',f'{N} x {N}'),('phi_cr',phi)], columns=['Parametro','Valore'])}

def _build_aligned_axes(inp, fem_nx, fem_ny):
    a_mm=_mm(inp.a,inp.unit); b_mm=_mm(inp.b,inp.unit); xs=list(np.linspace(0.0,a_mm,fem_nx+1)); ys=list(np.linspace(0.0,b_mm,fem_ny+1))
    if inp.stiffeners is not None and len(inp.stiffeners)>0:
        active=inp.stiffeners[inp.stiffeners['active']]
        for _,st in active.iterrows():
            loc=_mm(st['location'],inp.unit); bw=_mm(st.get('band_width',max(6.0*inp.t,1.0)),inp.unit); c0=max(0.0,loc-bw/2.0); c1=min(b_mm if st['orientation']=='longitudinale' else a_mm, loc+bw/2.0)
            if st['orientation']=='longitudinale': ys.extend([c0,c1])
            else: xs.extend([c0,c1])
    xs=np.array(sorted(set(np.round(xs,9)))); ys=np.array(sorted(set(np.round(ys,9)))); return xs,ys

def _check_closed_stiffener_connectivity(inp, mesh, tol=1e-8):
    if inp.stiffeners is None or len(inp.stiffeners)==0: return []
    pts=mesh.p.T; a_mm=_mm(inp.a,inp.unit); b_mm=_mm(inp.b,inp.unit); checks=[]; active=inp.stiffeners[inp.stiffeners['active']]
    for idx,st in active.iterrows():
        if not bool(st.get('closed_section',False)) and st.get('type') != 'closed box':
            continue
        loc=_mm(st['location'],inp.unit); bw=_mm(st.get('band_width',max(6.0*inp.t,1.0)),inp.unit); c0=max(0.0,loc-bw/2.0); c1=min(b_mm if st['orientation']=='longitudinale' else a_mm, loc+bw/2.0)
        if st['orientation']=='longitudinale':
            n0=int(np.sum(np.isclose(pts[:,1],c0,atol=tol))); n1=int(np.sum(np.isclose(pts[:,1],c1,atol=tol))); ok=(n0>1) and (n1>1)
        else:
            n0=int(np.sum(np.isclose(pts[:,0],c0,atol=tol))); n1=int(np.sum(np.isclose(pts[:,0],c1,atol=tol))); ok=(n0>1) and (n1>1)
        checks.append({'idx': int(idx), 'orientation': st['orientation'], 'border_1': c0, 'border_2': c1, 'nodes_border_1': n0, 'nodes_border_2': n1, 'ok': ok})
    return checks

def _build_stiffener_field(inp, xc_mm, yc_mm):
    tp=_mm(inp.t,inp.unit); Dref=inp.E*tp**3/(12*(1-inp.nu**2)); Dcoef=np.full_like(xc_mm,Dref,dtype=float); memcoef=np.full_like(xc_mm,tp,dtype=float); tags=np.zeros_like(xc_mm,dtype=int); tag=1
    if inp.stiffeners is None or len(inp.stiffeners)==0: return Dcoef, memcoef, tags
    active=inp.stiffeners[inp.stiffeners['active']]
    for _,st in active.iterrows():
        loc=_mm(st['location'],inp.unit); bw=_mm(st.get('band_width',max(6.0*inp.t,1.0)),inp.unit); t_mem=_mm(st.get('t_eq_mem',inp.t),inp.unit); t_bend=_mm(st.get('t_eq_bend',inp.t),inp.unit)
        if st['orientation']=='longitudinale': mask=np.abs(yc_mm-loc)<=bw/2.0
        else: mask=np.abs(xc_mm-loc)<=bw/2.0
        Dcoef[mask]=inp.E*t_bend**3/(12*(1-inp.nu**2)); memcoef[mask]=t_mem; tags[mask]=tag; tag+=1
    return Dcoef, memcoef, tags

def solve_buckling_problem_fem(inp, fem_nx=40, fem_ny=20, n_modes=6):
    try:
        from skfem import MeshTri, Basis, ElementTriP2, BilinearForm, asm, condense
        from skfem.helpers import dot, grad
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import eye as speye
    except Exception as e:
        return {'ok': False, 'message': f'scikit-fem o dipendenze non disponibili: {e}'}

    a_mm = _mm(inp.a, inp.unit)
    b_mm = _mm(inp.b, inp.unit)
    sx_fun, sy_fun, tau_fun = analytical_stress_functions(inp)

    # mesh conforme ai bordi delle bande irrigidenti
    xs, ys = _build_aligned_axes(inp, fem_nx, fem_ny)
    mesh = MeshTri.init_tensor(xs, ys)
    basis = Basis(mesh, ElementTriP2())

    D_field, mem_field = _build_stiffener_property_functions(inp)
    connectivity_checks = _check_closed_stiffener_connectivity(inp, mesh)

    @BilinearForm
    def k_form(u, v, w):
        xq = w.x[0]
        yq = w.x[1]
        Dq = D_field(xq, yq)
        mq = mem_field(xq, yq)
        return (Dq / np.maximum(mq, 1e-9)) * dot(grad(u), grad(v))

    @BilinearForm
    def kg_form(u, v, w):
        xq = w.x[0]
        yq = w.x[1]
        sxq = sx_fun(xq, yq)
        syq = sy_fun(xq, yq)
        tauq = tau_fun(xq, yq)
        gu = grad(u)
        gv = grad(v)
        return sxq * gu[0] * gv[0] + syq * gu[1] * gv[1] + tauq * (gu[0] * gv[1] + gu[1] * gv[0])

    K = asm(k_form, basis)
    KG = asm(kg_form, basis)

    x0 = basis.dofs.get_facet_dofs(mesh.facets_satisfying(lambda x: np.isclose(x[0], 0.0))).all()
    x1 = basis.dofs.get_facet_dofs(mesh.facets_satisfying(lambda x: np.isclose(x[0], a_mm))).all()
    y0 = basis.dofs.get_facet_dofs(mesh.facets_satisfying(lambda x: np.isclose(x[1], 0.0))).all()
    y1 = basis.dofs.get_facet_dofs(mesh.facets_satisfying(lambda x: np.isclose(x[1], b_mm))).all()
    D = np.unique(np.concatenate([x0, x1, y0, y1]))

    Kc, _ = condense(K, D=D)
    KGc, _ = condense(KG, D=D)
    Kc = Kc + 1e-9 * speye(Kc.shape[0])

    try:
        ksolve = max(2, min(n_modes, max(KGc.shape[0] - 2, 2)))
        vals, vecs = eigsh(KGc, k=ksolve, M=Kc, sigma=0.0, which='LM')
        finite = np.isfinite(vals) & (np.abs(vals) > 1e-12)
        vals = vals[finite]
        vecs = vecs[:, finite]
        lam = 1.0 / vals
        pos = np.sort(lam[np.isfinite(lam) & (lam > 1e-8)])
    except Exception as e:
        return {'ok': False, 'message': f'Errore nel solve FEM: {e}'}

    eig_list = []
    for k in range(min(len(pos), vecs.shape[1])):
        v = vecs[:, k]
        vmax = np.max(np.abs(v)) if np.max(np.abs(v)) > 0 else 1.0
        eig_list.append(v / vmax)

    log_rows = [
        ('Backend FEM', 'scikit-fem'),
        ('Mesh conforme a bande', 'Sì'),
        ('Mesh nx base', fem_nx),
        ('Mesh ny base', fem_ny),
        ('N ascisse effettive', len(xs)),
        ('N ordinate effettive', len(ys)),
        ('Elementi triangolari', int(mesh.t.shape[1])),
        ('DOF liberi', int(Kc.shape[0])),
        ('λcr', float(pos[0]) if len(pos) else np.nan),
    ]
    for chk in connectivity_checks:
        log_rows.append((
            f"Conn. stiffener chiuso #{chk['idx']}",
            f"ok={chk['ok']} | nodi bordo1={chk['nodes_border_1']} | nodi bordo2={chk['nodes_border_2']}"
        ))

    return {
        'ok': True,
        'lambda_cr': float(pos[0]) if len(pos) else np.nan,
        'phi_cr': float(pos[0]) if len(pos) else np.nan,
        'eigenvalues': pos,
        'eigs_df': pd.DataFrame({'Modo': np.arange(1, len(pos) + 1), 'lambda': pos}),
        'eigenvectors': eig_list,
        'basis_modes': [(i + 1, 1) for i in range(len(eig_list))],
        'a_mm': a_mm,
        'b_mm': b_mm,
        'ndof': int(Kc.shape[0]),
        'calc_log': pd.DataFrame(log_rows, columns=['Parametro', 'Valore']),
        'connectivity_checks': connectivity_checks,
    }

def _mode_surface(inp,result,mode_index,nx=70,ny=45):
    a=result['a_mm']; b=result['b_mm']; Xv=np.linspace(0,a,nx); Yv=np.linspace(0,b,ny); X,Y=np.meshgrid(Xv,Yv,indexing='xy'); Z=np.zeros_like(X)
    if 'phi_positive' in result:
        vec=result['eigenvectors'][mode_index]
        for c,(m,n) in zip(vec,result['basis_modes']): Z += c*_shape(m,n,X,Y,a,b)
    else:
        Z=np.sin((mode_index+1)*np.pi*X/max(a,1e-9))*np.sin(np.pi*Y/max(b,1e-9))
    return X,Y,Z

def make_mode_figure(inp,result,mode_index=0):
    X,Y,Z=_mode_surface(inp,result,mode_index); fig=go.Figure(data=[go.Surface(x=X/_f(inp.unit), y=Y/_f(inp.unit), z=Z, colorscale='Turbo')]); fig.update_layout(template='plotly_white',height=420,title=f'Modo di buckling {mode_index+1}',scene=dict(xaxis_title=f'x [{inp.unit}]', yaxis_title=f'y [{inp.unit}]', zaxis_title='w norm.')); return fig

def make_aij_table(result, mode_index=0):
    if 'phi_positive' not in result or len(result.get('eigenvectors',[]))==0: return pd.DataFrame(columns=['m','n','a_mn'])
    vec=result['eigenvectors'][mode_index]; return pd.DataFrame([{'m':m,'n':n,'a_mn':aij} for aij,(m,n) in zip(vec,result['basis_modes'])])

def summary_results_df(result, title='Solver'):
    rows=[('Backend',title)]
    if 'sigma_x_cr' in result: rows += [('φcr',result['phi_cr']),('σx,cr [MPa]',result['sigma_x_cr']),('σy,cr [MPa]',result['sigma_y_cr']),('τcr [MPa]',result['tau_cr'])]
    else: rows += [('λcr FEM',result.get('lambda_cr',np.nan)),('N modi',len(result.get('eigenvalues',[]))),('DOF',result.get('ndof',np.nan))]
    return pd.DataFrame(rows, columns=['Parametro','Valore'])

def summary_model_df(inp):
    return pd.DataFrame([('a',inp.a),('b',inp.b),('t',inp.t),('E [MPa]',inp.E),('ν',inp.nu)], columns=['Parametro','Valore'])

def make_compare_df(sem,fem):
    rows=[]; rows.append({'Parametro':'Autovalore critico','Semianalitico': sem.get('phi_cr') if sem is not None else np.nan,'FEM scikit-fem': fem.get('lambda_cr') if (fem is not None and fem.get('ok',False)) else np.nan}); rows.append({'Parametro':'Numero modi','Semianalitico': len(sem.get('phi_positive',[])) if sem is not None else np.nan,'FEM scikit-fem': len(fem.get('eigenvalues',[])) if (fem is not None and fem.get('ok',False)) else np.nan});
    if sem is not None and fem is not None and fem.get('ok',False):
        s=float(sem.get('phi_cr',np.nan)); f=float(fem.get('lambda_cr',np.nan)); rows.append({'Parametro':'Rapporto FEM / semianalitico','Semianalitico':1.0,'FEM scikit-fem': f/s if np.isfinite(s) and abs(s)>1e-12 else np.nan})
    return pd.DataFrame(rows)

def export_case_json(inp):
    d=asdict(inp); d['stiffeners']=inp.stiffeners.to_dict(orient='records') if isinstance(inp.stiffeners,pd.DataFrame) else []; d['mesh_df']=inp.mesh_df.to_dict(orient='records') if isinstance(inp.mesh_df,pd.DataFrame) else None; return json.dumps(d, ensure_ascii=False, indent=2).encode('utf-8')

def make_stress_surface(inp, stress_fun, name):
    a=_mm(inp.a,inp.unit); b=_mm(inp.b,inp.unit); Xv=np.linspace(0,a,50); Yv=np.linspace(0,b,30); X,Y=np.meshgrid(Xv,Yv,indexing='xy'); Z=np.vectorize(stress_fun)(X,Y); fig=go.Figure(data=[go.Surface(x=X/_f(inp.unit), y=Y/_f(inp.unit), z=Z, colorscale='RdBu_r')]); fig.update_layout(template='plotly_white', height=360, title=f'{name}(x,y)', scene=dict(xaxis_title=f'x [{inp.unit}]', yaxis_title=f'y [{inp.unit}]', zaxis_title=f'{name} [MPa]')); return fig

def _edge_style(kind):
    if kind=='Fisso': return '#dc2626',6,'solid'
    if kind=='Elastico': return '#f59e0b',5,'dash'
    return '#16a34a',4,'solid'

def _draw_support_symbols(fig,a,b,kind,side):
    color,_,_= _edge_style(kind); n=12
    if side=='top': xs=np.linspace(0,a,n); ys=np.full(n,b); symbol='triangle-down' if kind=='Semplice/hinged' else ('square' if kind=='Fisso' else 'diamond')
    elif side=='bottom': xs=np.linspace(0,a,n); ys=np.zeros(n); symbol='triangle-up' if kind=='Semplice/hinged' else ('square' if kind=='Fisso' else 'diamond')
    elif side=='left': xs=np.zeros(n); ys=np.linspace(0,b,n); symbol='triangle-right' if kind=='Semplice/hinged' else ('square' if kind=='Fisso' else 'diamond')
    else: xs=np.full(n,a); ys=np.linspace(0,b,n); symbol='triangle-left' if kind=='Semplice/hinged' else ('square' if kind=='Fisso' else 'diamond')
    fig.add_trace(go.Scatter(x=xs,y=ys,mode='markers',marker=dict(color=color,size=7,symbol=symbol),showlegend=False))

def make_plate_preview_figure(inp):
    a=inp.a; b=inp.b; fig=go.Figure(); fig.add_trace(go.Scatter(x=[0,a,a,0,0],y=[0,0,b,b,0],mode='lines',fill='toself',fillcolor='rgba(59,130,246,0.08)',line=dict(color='rgba(30,64,175,0.5)',width=1),name='Piastra'))
    for (xs,ys,kind,side,name) in [([0,a],[b,b],inp.edge_top,'top','Bordo alto'),([0,a],[0,0],inp.edge_bottom,'bottom','Bordo basso'),([0,0],[0,b],inp.edge_left,'left','Bordo sinistro'),([a,a],[0,b],inp.edge_right,'right','Bordo destro')]:
        color,width,dash=_edge_style(kind); fig.add_trace(go.Scatter(x=xs,y=ys,mode='lines',line=dict(color=color,width=width,dash=dash),name=f'{name}: {kind}')); _draw_support_symbols(fig,a,b,kind,side)
    if inp.stiffeners is not None and len(inp.stiffeners)>0:
        active=inp.stiffeners[inp.stiffeners['active']]
        for _,st in active.iterrows():
            loc=float(st['location']); bw=float(st.get('band_width', max(6.0*inp.t,1.0))); fc='rgba(37,99,235,0.18)' if not bool(st.get('closed_section',False)) else 'rgba(16,185,129,0.18)'; lc='rgba(37,99,235,0.35)' if not bool(st.get('closed_section',False)) else 'rgba(16,185,129,0.45)'
            if st['orientation']=='longitudinale':
                y0=max(0.0,loc-bw/2.0); y1=min(b,loc+bw/2.0); fig.add_trace(go.Scatter(x=[0,a,a,0,0],y=[y0,y0,y1,y1,y0],mode='lines',fill='toself',fillcolor=fc,line=dict(color=lc,width=1),showlegend=False)); fig.add_trace(go.Scatter(x=[0,a],y=[loc,loc],mode='lines',line=dict(color='#2563eb' if not bool(st.get('closed_section',False)) else '#10b981',width=3),name=f"{'Chiuso' if bool(st.get('closed_section',False)) else 'Aperto'} @ {loc:g}"))
            else:
                x0=max(0.0,loc-bw/2.0); x1=min(a,loc+bw/2.0); fig.add_trace(go.Scatter(x=[x0,x1,x1,x0,x0],y=[0,0,b,b,0],mode='lines',fill='toself',fillcolor=fc,line=dict(color=lc,width=1),showlegend=False)); fig.add_trace(go.Scatter(x=[loc,loc],y=[0,b],mode='lines',line=dict(color='#7c3aed' if not bool(st.get('closed_section',False)) else '#10b981',width=3),name=f"{'Chiusa' if bool(st.get('closed_section',False)) else 'Aperta'} @ {loc:g}"))
    fig.update_layout(template='plotly_white',height=520,title='Anteprima geometrica con simboli di vincolo e bande irrigidenti',xaxis_title=f'a [{inp.unit}]',yaxis_title=f'b [{inp.unit}]',legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0),margin=dict(l=20,r=20,t=70,b=20)); fig.update_yaxes(scaleanchor='x',scaleratio=1); return fig

def make_fem_model_figure(inp, fem_nx=40, fem_ny=20):
    xs, ys = _build_aligned_axes(inp, fem_nx, fem_ny)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    pts = np.column_stack([X.ravel(), Y.ravel()])
    tris=[]; nx=len(xs)-1; ny=len(ys)-1
    def nid(i,j): return j*(nx+1)+i
    for j in range(ny):
        for i in range(nx):
            n1=nid(i,j); n2=nid(i+1,j); n3=nid(i,j+1); n4=nid(i+1,j+1); tris.append([n1,n2,n4]); tris.append([n1,n4,n3])
    tris=np.array(tris, dtype=int)
    centers=pts[tris].mean(axis=1); _,_,tags=_build_stiffener_field(inp, centers[:,0], centers[:,1]); fig=go.Figure()
    for tri,tag in zip(tris,tags):
        x=pts[tri,0]/_f(inp.unit); y=pts[tri,1]/_f(inp.unit); fc='rgba(37,99,235,0.18)' if tag>0 else 'rgba(255,255,255,0)'; lc='rgba(37,99,235,0.40)' if tag>0 else 'rgba(100,116,139,0.28)'; fig.add_trace(go.Scatter(x=[x[0],x[1],x[2],x[0]],y=[y[0],y[1],y[2],y[0]],mode='lines',fill='toself',fillcolor=fc,line=dict(color=lc,width=1),showlegend=False,hoverinfo='skip'))
    for (xs2,ys2,kind,side) in [([0,inp.a],[inp.b,inp.b],inp.edge_top,'top'),([0,inp.a],[0,0],inp.edge_bottom,'bottom'),([0,0],[0,inp.b],inp.edge_left,'left'),([inp.a,inp.a],[0,inp.b],inp.edge_right,'right')]:
        color,width,dash=_edge_style(kind); fig.add_trace(go.Scatter(x=xs2,y=ys2,mode='lines',line=dict(color=color,width=width,dash=dash),showlegend=False)); _draw_support_symbols(fig,inp.a,inp.b,kind,side)
    fig.update_layout(template='plotly_white',height=520,title='Modello FEM discretizzato (mesh conforme + sottodomini plate irrigidenti)',xaxis_title=f'a [{inp.unit}]',yaxis_title=f'b [{inp.unit}]',margin=dict(l=20,r=20,t=70,b=20)); fig.update_yaxes(scaleanchor='x',scaleratio=1); return fig

def _build_stiffener_property_functions(inp):
    """
    Restituisce funzioni continue/piecewise sui punti (x, y) in mm:
    - D_field(x, y): rigidezza flessionale equivalente locale
    - mem_field(x, y): spessore membranale equivalente locale
    """
    tp = _mm(inp.t, inp.unit)
    Dref = inp.E * tp**3 / (12 * (1 - inp.nu**2))

    active = inp.stiffeners[inp.stiffeners['active']] if inp.stiffeners is not None and len(inp.stiffeners) > 0 else pd.DataFrame()

    def D_field(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        out = np.full_like(x, Dref, dtype=float)
        if len(active) == 0:
            return out
        for _, st in active.iterrows():
            loc = _mm(st['location'], inp.unit)
            bw = _mm(st.get('band_width', max(6.0 * inp.t, 1.0)), inp.unit)
            t_bend = _mm(st.get('t_eq_bend', inp.t), inp.unit)
            Dloc = inp.E * t_bend**3 / (12 * (1 - inp.nu**2))
            if st['orientation'] == 'longitudinale':
                mask = np.abs(y - loc) <= bw / 2.0
            else:
                mask = np.abs(x - loc) <= bw / 2.0
            out = np.where(mask, Dloc, out)
        return out

    def mem_field(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        out = np.full_like(x, tp, dtype=float)
        if len(active) == 0:
            return out
        for _, st in active.iterrows():
            loc = _mm(st['location'], inp.unit)
            bw = _mm(st.get('band_width', max(6.0 * inp.t, 1.0)), inp.unit)
            t_mem = _mm(st.get('t_eq_mem', inp.t), inp.unit)
            if st['orientation'] == 'longitudinale':
                mask = np.abs(y - loc) <= bw / 2.0
            else:
                mask = np.abs(x - loc) <= bw / 2.0
            out = np.where(mask, t_mem, out)
        return out

    return D_field, mem_field
