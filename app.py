
# -*- coding: utf-8 -*-
import json
import pandas as pd
import streamlit as st
from src import (
    default_stiffeners_df, clean_stiffeners_df, compute_stiffener_properties,
    orthotropy_from_smearing, build_plate_input, solve_buckling_problem,
    solve_buckling_problem_fem, analytical_stress_functions, make_stress_surface,
    make_mode_figure, make_aij_table, summary_results_df, summary_model_df,
    export_case_json, make_plate_preview_figure, make_stiffener_summary_df,
    make_fem_model_figure, make_compare_df,
)

st.set_page_config(page_title='EBPlateLite', layout='wide')
st.title('EBPlateLite v1.6 – Instabilità elastica di piastre in acciaio')
st.caption('Pacchetto completo attualmente implementato: backend semianalitico tipo EBPlate + backend FEM equivalente con scikit-fem, preview prima del run, confronto metodi, irrigidimenti aperti/chiusi e controllo di connettività per sezioni chiuse.')

if 'sem_res' not in st.session_state:
    st.session_state.sem_res = None
if 'fem_res' not in st.session_state:
    st.session_state.fem_res = None

with st.sidebar:
    st.header('Caso')
    unit = st.selectbox('Unità', ['mm','cm','m'], index=0)
    up = st.file_uploader('Importa caso JSON', type=['json'])
    imported = {}
    if up is not None:
        try:
            imported = json.load(up)
            st.success('Caso importato')
        except Exception:
            st.error('JSON non valido')

def g(k, d):
    return imported.get(k, d)

st.subheader('1) Parametri piastra')
c1, c2 = st.columns(2)
with c1:
    a = st.number_input('Larghezza a', min_value=1.0, value=float(g('a', 3000.0)), step=100.0)
    b = st.number_input('Altezza b', min_value=1.0, value=float(g('b', 1500.0)), step=100.0)
    t = st.number_input('Spessore t', min_value=0.1, value=float(g('t', 10.0)), step=0.5)
with c2:
    E = st.number_input('E [MPa]', min_value=1000.0, value=float(g('E', 210000.0)), step=1000.0)
    nu = st.number_input('ν', min_value=0.0, max_value=0.49, value=float(g('nu', 0.30)), step=0.01)

edge_types = ['Semplice/hinged','Fisso','Elastico']
st.markdown('**Vincoli rotazionali ai bordi** (w = 0 su tutti i lati)')
e1, e2 = st.columns(2)
with e1:
    edge_top = st.selectbox('Bordo alto', edge_types, index=edge_types.index(g('edge_top', 'Semplice/hinged')))
    edge_bottom = st.selectbox('Bordo basso', edge_types, index=edge_types.index(g('edge_bottom', 'Semplice/hinged')))
with e2:
    edge_left = st.selectbox('Bordo sinistro', edge_types, index=edge_types.index(g('edge_left', 'Semplice/hinged')))
    edge_right = st.selectbox('Bordo destro', edge_types, index=edge_types.index(g('edge_right', 'Semplice/hinged')))

k1, k2, k3, k4 = st.columns(4)
with k1:
    kr_top = st.number_input('Kr alto', value=float(g('kr_top', 0.0)))
with k2:
    kr_bottom = st.number_input('Kr basso', value=float(g('kr_bottom', 0.0)))
with k3:
    kr_left = st.number_input('Kr sinistro', value=float(g('kr_left', 0.0)))
with k4:
    kr_right = st.number_input('Kr destro', value=float(g('kr_right', 0.0)))

st.subheader('2) Ortotropia e irrigidimenti')
o1, o2 = st.columns(2)
with o1:
    ortho = st.checkbox('Piastra ortotropa', value=bool(g('ortho', False)))
    smear = st.checkbox('Ortotropia da smearing', value=bool(g('smear', False)))
with o2:
    beta_x = st.number_input('βx', value=float(g('beta_x', 0.0)), step=0.05)
    eta_x = st.number_input('ηx', value=float(g('eta_x', 0.0)), step=0.05)
    beta_y = st.number_input('βy', value=float(g('beta_y', 0.0)), step=0.05)
    eta_y = st.number_input('ηy', value=float(g('eta_y', 0.0)), step=0.05)

st.markdown('**Irrigidimenti** — supporto a sezioni aperte e chiuse (`closed_section`). Nel backend FEM attuale gli irrigidimenti sono modellati come sottodomini plate equivalenti con spessore membranale/flessionale dedicato; per le sezioni chiuse viene controllata la connettività nodale sui due bordi della banda.')
stiff_df = pd.DataFrame(g('stiffeners', default_stiffeners_df().to_dict(orient='records')))
stiff_df = st.data_editor(
    stiff_df, num_rows='dynamic', use_container_width=True,
    column_config={
        'active': st.column_config.CheckboxColumn('Attivo'),
        'closed_section': st.column_config.CheckboxColumn('Sezione chiusa'),
        'orientation': st.column_config.SelectboxColumn('Orientamento', options=['longitudinale','trasversale']),
        'type': st.column_config.SelectboxColumn('Tipo', options=['general','flat bar','sym flat bar','T','angle','trapezoid','closed box'])
    }, key='stiffeners')
stiff_df = clean_stiffeners_df(stiff_df)
props_df = compute_stiffener_properties(stiff_df, t=t, unit=unit, E=E, nu=nu)
if smear and not props_df.empty:
    beta_x, eta_x, beta_y, eta_y = orthotropy_from_smearing(props_df, a=a, b=b, t=t, unit=unit)
    st.info(f'Smearing -> βx={beta_x:.4f}, ηx={eta_x:.4f}, βy={beta_y:.4f}, ηy={eta_y:.4f}')
if not props_df.empty:
    cols = ['active','closed_section','orientation','type','location','band_width','t_eq_mem','t_eq_bend','gamma','theta','J_eff']
    st.dataframe(props_df[cols], use_container_width=True, height=280)

st.subheader('3) Tensioni nel piano')
s1, s2, s3 = st.columns(3)
with s1:
    s_xtl = st.number_input('σx tl [MPa]', value=float(g('s_xtl', 100.0)))
    s_xbl = st.number_input('σx bl [MPa]', value=float(g('s_xbl', 100.0)))
    s_xtr = st.number_input('σx tr [MPa]', value=float(g('s_xtr', 100.0)))
    s_xbr = st.number_input('σx br [MPa]', value=float(g('s_xbr', 100.0)))
with s2:
    s_yut = st.number_input('σy alto [MPa]', value=float(g('s_yut', 0.0)))
    s_yub = st.number_input('σy basso [MPa]', value=float(g('s_yub', 0.0)))
    s_ypt = st.number_input('σy patch alto [MPa]', value=float(g('s_ypt', 0.0)))
    s_ypb = st.number_input('σy patch basso [MPa]', value=float(g('s_ypb', 0.0)))
    c_t = st.number_input('c_t', min_value=0.0, value=float(g('c_t', 0.0)))
    c_b = st.number_input('c_b', min_value=0.0, value=float(g('c_b', 0.0)))
    patch_with_flanges = st.checkbox('Plate with flanges', value=bool(g('patch_with_flanges', False)))
with s3:
    tau_u = st.number_input('τ uniforme [MPa]', value=float(g('tau_u', 0.0)))
    mesh_up = st.file_uploader('CSV stress meshed', type=['csv'])
    mesh_df = None
    if mesh_up is not None:
        try:
            mesh_df = pd.read_csv(mesh_up)
            st.success('CSV importato')
        except Exception:
            st.error('CSV non leggibile')

st.subheader('4) Opzioni di calcolo')
p1, p2, p3 = st.columns(3)
with p1:
    complexity = st.selectbox('Complessità', [1,2,3], index=int(g('complexity', 1)) - 1)
with p2:
    search_mode = st.selectbox('Modi da ricercare', ['1° modo','primi 20','tutti'], index=['1° modo','primi 20','tutti'].index(g('search_mode', '1° modo')))
with p3:
    plate_behaviour = st.checkbox('Plate behaviour', value=bool(g('plate_behaviour', False)))

f1, f2, f3 = st.columns(3)
with f1:
    fem_nx = st.slider('Divisioni FEM x', 8, 120, int(g('fem_nx', 40)))
with f2:
    fem_ny = st.slider('Divisioni FEM y', 4, 80, int(g('fem_ny', 20)))
with f3:
    fem_modes = st.slider('N modi FEM', 1, 12, int(g('fem_modes', 6)))

inp = build_plate_input(
    a=a,b=b,t=t,E=E,nu=nu,unit=unit,
    edge_top=edge_top,edge_bottom=edge_bottom,edge_left=edge_left,edge_right=edge_right,
    kr_top=kr_top,kr_bottom=kr_bottom,kr_left=kr_left,kr_right=kr_right,
    J_top=0.0,J_bottom=0.0,J_left=0.0,J_right=0.0,
    beta_x=beta_x if ortho else 0.0, eta_x=eta_x if ortho else 0.0, beta_y=beta_y if ortho else 0.0, eta_y=eta_y if ortho else 0.0,
    stiffeners=props_df, s_xtl=s_xtl,s_xbl=s_xbl,s_xtr=s_xtr,s_xbr=s_xbr,
    s_yut=s_yut,s_yub=s_yub,s_ypt=s_ypt,s_ypb=s_ypb,c_t=c_t,c_b=c_b,
    tau_u=tau_u, imposed_x=False, imposed_y=False, imposed_tau=False,
    patch_with_flanges=patch_with_flanges, mesh_df=mesh_df,
    complexity=complexity, search_mode=search_mode, plate_behaviour=plate_behaviour)

st.subheader('Anteprima pre-analisi')
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(make_plate_preview_figure(inp), use_container_width=True)
with g2:
    st.plotly_chart(make_fem_model_figure(inp, fem_nx=fem_nx, fem_ny=fem_ny), use_container_width=True)
    st.dataframe(make_stiffener_summary_df(inp.stiffeners), use_container_width=True, height=220)

b1, b2 = st.columns(2)
if b1.button('Calcola instabilità elastica (solver semianalitico)', use_container_width=True):
    st.session_state.sem_res = solve_buckling_problem(inp)
if b2.button('Calcola instabilità elastica con FEM (scikit-fem)', type='primary', use_container_width=True):
    st.session_state.fem_res = solve_buckling_problem_fem(inp, fem_nx=fem_nx, fem_ny=fem_ny, n_modes=fem_modes)

T1, T2, T3 = st.tabs(['Risultati semianalitici','Risultati FEM','Confronto metodi'])
with T1:
    r = st.session_state.sem_res
    if r is None:
        st.info('Esegui il solver semianalitico.')
    else:
        m1,m2,m3,m4 = st.columns(4)
        m1.metric('φcr', f"{r['phi_cr']:.4f}")
        m2.metric('σx,cr', f"{r['sigma_x_cr']:.2f}")
        m3.metric('σy,cr', f"{r['sigma_y_cr']:.2f}")
        m4.metric('τcr', f"{r['tau_cr']:.2f}")
        st.dataframe(summary_results_df(r,'Semianalitico'), use_container_width=True)
        st.dataframe(r['modes_df'], use_container_width=True, height=220)
        if len(r['phi_positive']) > 0:
            st.plotly_chart(make_mode_figure(inp, r, 0), use_container_width=True)
        st.dataframe(r['calc_log'], use_container_width=True, height=220)
with T2:
    r = st.session_state.fem_res
    if r is None:
        st.info('Esegui il backend FEM.')
    elif not r.get('ok', False):
        st.error(r.get('message', 'Errore backend FEM'))
    else:
        m1,m2,m3 = st.columns(3)
        m1.metric('λcr FEM', f"{r['lambda_cr']:.4f}")
        m2.metric('N modi', str(len(r['eigenvalues'])))
        m3.metric('DOF', str(r['ndof']))
        st.dataframe(summary_results_df(r,'FEM scikit-fem'), use_container_width=True)
        st.dataframe(r['eigs_df'], use_container_width=True, height=220)
        if len(r['eigenvalues']) > 0:
            st.plotly_chart(make_mode_figure(inp, r, 0), use_container_width=True)
        st.dataframe(r['calc_log'], use_container_width=True, height=220)
        if 'connectivity_checks' in r and len(r['connectivity_checks']) > 0:
            st.markdown('**Controlli di connettività irrigidimenti chiusi**')
            st.dataframe(pd.DataFrame(r['connectivity_checks']), use_container_width=True)
with T3:
    st.dataframe(make_compare_df(st.session_state.sem_res, st.session_state.fem_res), use_container_width=True)
    if st.session_state.sem_res is None or st.session_state.fem_res is None:
        st.info('Esegui entrambi i metodi per vedere il confronto completo.')

st.download_button('Scarica caso JSON', export_case_json(inp), 'ebplatelite_case.json', 'application/json')
