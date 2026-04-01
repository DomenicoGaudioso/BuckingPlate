
# -*- coding: utf-8 -*-
"""
EBPlateLite - app.py consolidato
================================

Versione UI consolidata, semplice e tecnica, con:
- preview sempre visibile prima del run;
- backend semianalitico tipo EBPlate prioritario;
- backend FEM equivalente mantenuto come confronto secondario;
- pannello manual check EC3-like con riepilogo, dettagli e log.
"""

import json
import pandas as pd
import streamlit as st

from src import (
    default_stiffeners_df,
    clean_stiffeners_df,
    compute_stiffener_properties,
    orthotropy_from_smearing,
    build_plate_input,
    solve_buckling_problem,
    solve_buckling_problem_fem,
    make_mode_figure,
    summary_results_df,
    summary_model_df,
    export_case_json,
    make_plate_preview_figure,
    make_stiffener_summary_df,
    make_fem_model_figure,
    make_compare_df,
    compute_ec3_manual_checks,
    estimate_psi_x_from_inputs,
    estimate_psi_y_from_inputs,
)

st.set_page_config(page_title='EBPlateLite', layout='wide')
st.title('EBPlateLite v2.1 - EBPlate + EC3 manual check consolidato')
st.caption('Priorita corrente: backend EBPlate trasparente e verificabile, confronto manuale EC3-like, FEM equivalente secondario.')

if 'sem_res' not in st.session_state:
    st.session_state.sem_res = None
if 'fem_res' not in st.session_state:
    st.session_state.fem_res = None

with st.sidebar:
    st.header('Caso')
    unit = st.selectbox('Unita', ['mm', 'cm', 'm'], index=0)
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
    nu = st.number_input('nu', min_value=0.0, max_value=0.49, value=float(g('nu', 0.30)), step=0.01)
    fy = st.number_input('fy [MPa]', min_value=100.0, value=float(g('fy', 355.0)), step=5.0)
    gamma_M1 = st.number_input('gamma_M1', min_value=0.5, value=float(g('gamma_M1', 1.0)), step=0.05)

edge_types = ['Semplice/hinged', 'Fisso', 'Elastico']
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
    beta_x = st.number_input('beta_x', value=float(g('beta_x', 0.0)), step=0.05)
    eta_x = st.number_input('eta_x', value=float(g('eta_x', 0.0)), step=0.05)
    beta_y = st.number_input('beta_y', value=float(g('beta_y', 0.0)), step=0.05)
    eta_y = st.number_input('eta_y', value=float(g('eta_y', 0.0)), step=0.05)

stiff_df = pd.DataFrame(g('stiffeners', default_stiffeners_df().to_dict(orient='records')))
stiff_df = st.data_editor(
    stiff_df,
    num_rows='dynamic',
    use_container_width=True,
    column_config={
        'active': st.column_config.CheckboxColumn('Attivo'),
        'closed_section': st.column_config.CheckboxColumn('Sezione chiusa'),
        'orientation': st.column_config.SelectboxColumn('Orientamento', options=['longitudinale', 'trasversale']),
        'type': st.column_config.SelectboxColumn('Tipo', options=['general', 'flat bar', 'sym flat bar', 'T', 'angle', 'trapezoid', 'closed box']),
    },
    key='stiffeners',
)
stiff_df = clean_stiffeners_df(stiff_df)
props_df = compute_stiffener_properties(stiff_df, t=t, unit=unit, E=E, nu=nu)
if smear and not props_df.empty:
    beta_x, eta_x, beta_y, eta_y = orthotropy_from_smearing(props_df, a=a, b=b, t=t, unit=unit)
    st.info(f'Smearing -> beta_x={beta_x:.4f}, eta_x={eta_x:.4f}, beta_y={beta_y:.4f}, eta_y={eta_y:.4f}')
if not props_df.empty:
    cols = [c for c in ['active', 'closed_section', 'orientation', 'type', 'location', 'band_width', 't_eq_mem', 't_eq_bend', 'gamma', 'theta', 'J_eff'] if c in props_df.columns]
    st.dataframe(props_df[cols], use_container_width=True, height=260)

st.subheader('3) Tensioni nel piano')
s1, s2, s3 = st.columns(3)
with s1:
    s_xtl = st.number_input('sigma_x tl [MPa]', value=float(g('s_xtl', 100.0)))
    s_xbl = st.number_input('sigma_x bl [MPa]', value=float(g('s_xbl', 100.0)))
    s_xtr = st.number_input('sigma_x tr [MPa]', value=float(g('s_xtr', 100.0)))
    s_xbr = st.number_input('sigma_x br [MPa]', value=float(g('s_xbr', 100.0)))
with s2:
    s_yut = st.number_input('sigma_y alto [MPa]', value=float(g('s_yut', 0.0)))
    s_yub = st.number_input('sigma_y basso [MPa]', value=float(g('s_yub', 0.0)))
    s_ypt = st.number_input('sigma_y patch alto [MPa]', value=float(g('s_ypt', 0.0)))
    s_ypb = st.number_input('sigma_y patch basso [MPa]', value=float(g('s_ypb', 0.0)))
    c_t = st.number_input('c_t', min_value=0.0, value=float(g('c_t', 0.0)))
    c_b = st.number_input('c_b', min_value=0.0, value=float(g('c_b', 0.0)))
    patch_with_flanges = st.checkbox('Plate with flanges', value=bool(g('patch_with_flanges', False)))
with s3:
    tau_u = st.number_input('tau uniforme [MPa]', value=float(g('tau_u', 0.0)))
    mesh_up = st.file_uploader('CSV stress meshed', type=['csv'])
    mesh_df = None
    if mesh_up is not None:
        try:
            mesh_df = pd.read_csv(mesh_up)
            st.success('CSV importato')
        except Exception:
            st.error('CSV non leggibile')

st.subheader('4) Parametri manual check EC3')
psi_x_auto = estimate_psi_x_from_inputs(s_xtl, s_xbl, s_xtr, s_xbr)
psi_y_auto = estimate_psi_y_from_inputs(s_yut, s_yub)
m1, m2, m3, m4 = st.columns(4)
with m1:
    panel_type_x = st.selectbox('Tipo pannello x', ['internal', 'external'], index=['internal', 'external'].index(g('panel_type_x', 'internal')))
with m2:
    panel_type_y = st.selectbox('Tipo pannello y', ['internal', 'external'], index=['internal', 'external'].index(g('panel_type_y', 'internal')))
with m3:
    psi_x = st.number_input('psi_x manuale', min_value=-3.0, max_value=1.0, value=float(g('psi_x', psi_x_auto)), step=0.05)
with m4:
    psi_y = st.number_input('psi_y manuale', min_value=-3.0, max_value=1.0, value=float(g('psi_y', psi_y_auto)), step=0.05)
st.caption(f'psi_x auto stimato = {psi_x_auto:.3f} | psi_y auto stimato = {psi_y_auto:.3f}')

st.subheader('5) Opzioni di calcolo')
p1, p2, p3 = st.columns(3)
with p1:
    complexity = st.selectbox('Complessita', [1, 2, 3], index=int(g('complexity', 1)) - 1)
with p2:
    search_mode = st.selectbox('Modi da ricercare', ['1° modo', 'primi 20', 'tutti'], index=['1° modo', 'primi 20', 'tutti'].index(g('search_mode', '1° modo')))
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
    a=a, b=b, t=t, E=E, nu=nu, unit=unit,
    edge_top=edge_top, edge_bottom=edge_bottom, edge_left=edge_left, edge_right=edge_right,
    kr_top=kr_top, kr_bottom=kr_bottom, kr_left=kr_left, kr_right=kr_right,
    J_top=0.0, J_bottom=0.0, J_left=0.0, J_right=0.0,
    beta_x=beta_x if ortho else 0.0,
    eta_x=eta_x if ortho else 0.0,
    beta_y=beta_y if ortho else 0.0,
    eta_y=eta_y if ortho else 0.0,
    stiffeners=props_df,
    s_xtl=s_xtl, s_xbl=s_xbl, s_xtr=s_xtr, s_xbr=s_xbr,
    s_yut=s_yut, s_yub=s_yub, s_ypt=s_ypt, s_ypb=s_ypb,
    c_t=c_t, c_b=c_b, tau_u=tau_u,
    imposed_x=False, imposed_y=False, imposed_tau=False,
    patch_with_flanges=patch_with_flanges,
    mesh_df=mesh_df,
    complexity=complexity,
    search_mode=search_mode,
    plate_behaviour=plate_behaviour,
    fy=fy,
    gamma_M1=gamma_M1,
    panel_type_x=panel_type_x,
    panel_type_y=panel_type_y,
    psi_x=psi_x,
    psi_y=psi_y,
)

manual_res = compute_ec3_manual_checks(inp, st.session_state.sem_res, st.session_state.fem_res)

st.subheader('Anteprima pre-analisi')
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(make_plate_preview_figure(inp), use_container_width=True)
with g2:
    st.plotly_chart(make_fem_model_figure(inp, fem_nx=fem_nx, fem_ny=fem_ny), use_container_width=True)
    st.dataframe(make_stiffener_summary_df(inp.stiffeners), use_container_width=True, height=220)

b1, b2 = st.columns(2)
if b1.button('Calcola instabilita elastica (solver semianalitico)', use_container_width=True):
    st.session_state.sem_res = solve_buckling_problem(inp)
    manual_res = compute_ec3_manual_checks(inp, st.session_state.sem_res, st.session_state.fem_res)
if b2.button('Calcola instabilita elastica con FEM (OpenSeesPy)', type='primary', use_container_width=True):
    st.session_state.fem_res = solve_buckling_problem_fem(inp, fem_nx=fem_nx, fem_ny=fem_ny, n_modes=fem_modes)
    manual_res = compute_ec3_manual_checks(inp, st.session_state.sem_res, st.session_state.fem_res)

T1, T2, T3, T4, T5 = st.tabs(['Risultati EBPlate', 'Risultati FEM', 'Verifiche manuali EC3', 'Confronto', 'Log tecnico'])

with T1:
    r = st.session_state.sem_res
    if r is None:
        st.info('Esegui il solver semianalitico.')
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('phi_cr', f"{r['phi_cr']:.4f}")
        c2.metric('sigma_x,cr', f"{r['sigma_x_cr']:.2f}")
        c3.metric('sigma_y,cr', f"{r['sigma_y_cr']:.2f}")
        c4.metric('tau_cr', f"{r['tau_cr']:.2f}")
        st.dataframe(summary_results_df(r, 'Semianalitico tipo EBPlate'), use_container_width=True)
        st.dataframe(r['modes_df'], use_container_width=True, height=220)
        if len(r.get('phi_positive', [])) > 0:
            st.plotly_chart(make_mode_figure(inp, r, 0), use_container_width=True)
        st.dataframe(r['calc_log'], use_container_width=True, height=260)

with T2:
    r = st.session_state.fem_res
    if r is None:
        st.info('Esegui il backend FEM equivalente.')
    elif not r.get('ok', False):
        st.error(r.get('message', 'Errore backend FEM'))
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric('lambda_cr FEM', f"{r['lambda_cr']:.4f}")
        c2.metric('N modi', str(len(r['eigenvalues'])))
        c3.metric('DOF', str(r['ndof']))
        if r.get('sanity_warning'):
            st.warning(r['sanity_warning'])
        st.dataframe(summary_results_df(r, 'FEM OpenSeesPy (ShellMITC4)'), use_container_width=True)
        st.dataframe(r['eigs_df'], use_container_width=True, height=220)
        if len(r.get('eigenvalues', [])) > 0:
            st.plotly_chart(make_mode_figure(inp, r, 0), use_container_width=True)
        st.dataframe(r['calc_log'], use_container_width=True, height=260)
        if 'connectivity_checks' in r and len(r['connectivity_checks']) > 0:
            st.markdown('**Controlli di connettivita irrigidimenti chiusi**')
            st.dataframe(pd.DataFrame(r['connectivity_checks']), use_container_width=True)

with T3:
    st.dataframe(
        manual_res.get('summary_df', pd.DataFrame()),
        use_container_width=True,
        height=360
    )

    st.markdown('**Tabella k_sigma**')
    st.dataframe(
        manual_res.get('ksigma_table_df', pd.DataFrame()),
        use_container_width=True,
        height=180
    )

    st.markdown('**Tabella rho**')
    st.dataframe(
        manual_res.get('rho_table_df', pd.DataFrame()),
        use_container_width=True,
        height=140
    )

    st.markdown('**Tabella larghezze efficaci**')
    st.dataframe(
        manual_res.get('beff_table_df', pd.DataFrame()),
        use_container_width=True,
        height=180
    )

    st.markdown('**Dettaglio subpannelli**')
    st.dataframe(
        manual_res.get('details_df', pd.DataFrame()),
        use_container_width=True,
        height=320
    )

    st.dataframe(
        manual_res.get('calc_log', pd.DataFrame()),
        use_container_width=True,
        height=220
    )

    st.warning(
        'Le formule EC3 implementate sono quelle leggibili nel workflow documentale allegato. '
        'I casi normativi completi external/psi generico richiedono ancora il testo completo '
        'EN 1993-1-5 o commentary ECCS.'
    )
    
with T4:
    st.dataframe(make_compare_df(st.session_state.sem_res, st.session_state.fem_res, manual_res), use_container_width=True)
    if not manual_res['compare_df'].empty:
        st.markdown('**Confronto EBPlate ↔ EC3 manuale**')
        st.dataframe(manual_res['compare_df'], use_container_width=True)

with T5:
    log_blocks = []
    log_blocks.append(pd.DataFrame([
        ('Preview pre-run', 'Sempre attiva: anteprima geometrica + preview modello equivalente/FEM'),
        ('Priorita backend', 'EBPlate prioritario; FEM equivalente secondario'),
        ('Import/export JSON', 'Mantenuto'),
        ('Manual check EC3', 'Attivo con riepilogo, dettagli e log'),
    ], columns=['Voce', 'Stato']))
    if st.session_state.sem_res is not None:
        sem_log = st.session_state.sem_res['calc_log'].copy(); sem_log.columns = ['Voce', 'Stato']; log_blocks.append(sem_log)
    if st.session_state.fem_res is not None and st.session_state.fem_res.get('ok', False):
        fem_log = st.session_state.fem_res['calc_log'].copy(); fem_log.columns = ['Voce', 'Stato']; log_blocks.append(fem_log)
    man_log = manual_res['calc_log'].copy(); man_log.columns = ['Voce', 'Stato']; log_blocks.append(man_log)
    full_log = pd.concat(log_blocks, ignore_index=True)
    st.dataframe(full_log, use_container_width=True, height=520)

st.download_button('Scarica caso JSON', export_case_json(inp), 'ebplatelite_case.json', 'application/json')
