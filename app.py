# -*- coding: utf-8 -*-
import json
import pandas as pd
import streamlit as st

from src import (
    PlateInput,
    default_stiffeners_df,
    clean_stiffeners_df,
    compute_stiffener_properties,
    orthotropy_from_smearing,
    build_plate_input,
    solve_buckling_problem,
    analytical_stress_functions,
    make_stress_surface,
    make_mode_figure,
    make_aij_table,
    summary_results_df,
    summary_model_df,
    export_case_json,
    make_plate_preview_figure,
    make_stiffener_summary_df,
)

st.set_page_config(page_title="EBPlateLite", layout="wide")
st.title("EBPlateLite v1.1.2 – Instabilità elastica di piastre in acciaio")
st.caption(
    "App Streamlit self-contained ispirata al flusso EBPlate per piastre rettangolari "
    "con carichi nel piano, irrigidimenti, ortotropia e anteprima geometrica con simboli di vincolo."
)

# =========================
# Sidebar / import caso
# =========================
with st.sidebar:
    st.header("Caso")
    unit = st.selectbox("Unità di lunghezza", ["mm", "cm", "m"], index=0)

    up = st.file_uploader("Importa caso JSON", type=["json"])
    imported = {}
    if up is not None:
        try:
            imported = json.load(up)
            st.success("Caso importato")
        except Exception:
            st.error("JSON non valido")

def g(key, default):
    return imported.get(key, default)

# =========================
# 1) Parametri della piastra
# =========================
st.subheader("1) Parametri della piastra")
c1, c2 = st.columns(2)

with c1:
    a = st.number_input("Larghezza a", min_value=1.0, value=float(g("a", 3000.0)), step=100.0)
    b = st.number_input("Altezza b", min_value=1.0, value=float(g("b", 1500.0)), step=100.0)
    t = st.number_input("Spessore t", min_value=0.1, value=float(g("t", 10.0)), step=0.5)

with c2:
    E = st.number_input("Modulo di Young E [MPa]", min_value=1000.0, value=float(g("E", 210000.0)), step=1000.0)
    nu = st.number_input("Poisson ν", min_value=0.0, max_value=0.49, value=float(g("nu", 0.30)), step=0.01)

st.markdown("**Vincoli rotazionali ai bordi** (si assume sempre **w = 0** su tutti i lati).")

edge_types = ["Semplice/hinged", "Fisso", "Elastico"]
ce1, ce2 = st.columns(2)

with ce1:
    edge_top = st.selectbox("Bordo alto", edge_types, index=edge_types.index(g("edge_top", "Semplice/hinged")))
    edge_bottom = st.selectbox("Bordo basso", edge_types, index=edge_types.index(g("edge_bottom", "Semplice/hinged")))

with ce2:
    edge_left = st.selectbox("Bordo sinistro", edge_types, index=edge_types.index(g("edge_left", "Semplice/hinged")))
    edge_right = st.selectbox("Bordo destro", edge_types, index=edge_types.index(g("edge_right", "Semplice/hinged")))

ck1, ck2, ck3, ck4 = st.columns(4)
with ck1:
    kr_top = st.number_input("Kr alto", value=float(g("kr_top", 0.0)))
    J_top = st.number_input("J alto", value=float(g("J_top", 0.0)))
with ck2:
    kr_bottom = st.number_input("Kr basso", value=float(g("kr_bottom", 0.0)))
    J_bottom = st.number_input("J basso", value=float(g("J_bottom", 0.0)))
with ck3:
    kr_left = st.number_input("Kr sinistro", value=float(g("kr_left", 0.0)))
    J_left = st.number_input("J sinistro", value=float(g("J_left", 0.0)))
with ck4:
    kr_right = st.number_input("Kr destro", value=float(g("kr_right", 0.0)))
    J_right = st.number_input("J destro", value=float(g("J_right", 0.0)))

# =========================
# 2) Ortotropia e irrigidimenti
# =========================
st.subheader("2) Ortotropia e irrigidimenti")
co1, co2 = st.columns(2)

with co1:
    ortho = st.checkbox("Piastra ortotropa", value=bool(g("ortho", False)))
    smear = st.checkbox(
        "Ortotropia da smearing di irrigidimenti uguali e regolari",
        value=bool(g("smear", False)),
    )

with co2:
    beta_x = st.number_input("βx", value=float(g("beta_x", 0.0)), step=0.05)
    eta_x = st.number_input("ηx", value=float(g("eta_x", 0.0)), step=0.05)
    beta_y = st.number_input("βy", value=float(g("beta_y", 0.0)), step=0.05)
    eta_y = st.number_input("ηy", value=float(g("eta_y", 0.0)), step=0.05)

stiff_df = pd.DataFrame(g("stiffeners", default_stiffeners_df().to_dict(orient="records")))
stiff_df = st.data_editor(
    stiff_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "active": st.column_config.CheckboxColumn("Attivo"),
        "orientation": st.column_config.SelectboxColumn(
            "Orientamento", options=["longitudinale", "trasversale"]
        ),
        "type": st.column_config.SelectboxColumn(
            "Tipo", options=["general", "flat bar", "sym flat bar", "T", "angle", "trapezoid"]
        ),
    },
    key="stiffeners",
)

stiff_df = clean_stiffeners_df(stiff_df)
props_df = compute_stiffener_properties(stiff_df, t=t, unit=unit)

if smear and not props_df.empty:
    beta_x, eta_x, beta_y, eta_y = orthotropy_from_smearing(props_df, a=a, b=b, t=t, unit=unit)
    st.info(f"Smearing -> βx={beta_x:.4f}, ηx={eta_x:.4f}, βy={beta_y:.4f}, ηy={eta_y:.4f}")

if not props_df.empty:
    st.dataframe(
        props_df[
            [
                "active",
                "orientation",
                "type",
                "location",
                "A_eff",
                "I_eff",
                "J_eff",
                "delta",
                "gamma",
                "theta",
                "Kr_local",
                "Jt_local",
            ]
        ],
        use_container_width=True,
        height=260,
    )

# =========================
# 3) Tensioni nel piano
# =========================
st.subheader("3) Tensioni nel piano")
st.markdown("Le tensioni possono essere **analitiche**, **meshed (CSV)** oppure la **somma** delle due.")

cs1, cs2, cs3 = st.columns(3)

with cs1:
    st.markdown("**σx ai 4 angoli**")
    s_xtl = st.number_input("σx tl [MPa]", value=float(g("s_xtl", 100.0)))
    s_xbl = st.number_input("σx bl [MPa]", value=float(g("s_xbl", 100.0)))
    s_xtr = st.number_input("σx tr [MPa]", value=float(g("s_xtr", 100.0)))
    s_xbr = st.number_input("σx br [MPa]", value=float(g("s_xbr", 100.0)))
    imposed_x = st.checkbox("σx imposta", value=bool(g("imposed_x", False)))

with cs2:
    st.markdown("**σy + patch loading**")
    s_yut = st.number_input("σy uniforme alto [MPa]", value=float(g("s_yut", 0.0)))
    s_yub = st.number_input("σy uniforme basso [MPa]", value=float(g("s_yub", 0.0)))
    s_ypt = st.number_input("σy patch alto [MPa]", value=float(g("s_ypt", 0.0)))
    s_ypb = st.number_input("σy patch basso [MPa]", value=float(g("s_ypb", 0.0)))
    c_t = st.number_input("c_t", min_value=0.0, value=float(g("c_t", 0.0)))
    c_b = st.number_input("c_b", min_value=0.0, value=float(g("c_b", 0.0)))
    patch_with_flanges = st.checkbox(
        "Plate with flanges per patch loading",
        value=bool(g("patch_with_flanges", False)),
    )
    imposed_y = st.checkbox("σy imposta", value=bool(g("imposed_y", False)))

with cs3:
    st.markdown("**τ uniforme**")
    tau_u = st.number_input("τ uniforme [MPa]", value=float(g("tau_u", 0.0)))
    imposed_tau = st.checkbox("τ imposta", value=bool(g("imposed_tau", False)))

    st.markdown("**CSV stress meshed**")
    st.caption("Colonne richieste: x, y, sigma_x, sigma_y, tau. x e y possono essere 0..1 oppure coordinate assolute.")
    mesh_up = st.file_uploader("Importa CSV stress meshed", type=["csv"])
    mesh_df = None
    if mesh_up is not None:
        try:
            mesh_df = pd.read_csv(mesh_up)
            st.success("CSV importato")
        except Exception:
            st.error("CSV non leggibile")

# =========================
# 4) Opzioni di calcolo
# =========================
st.subheader("4) Opzioni di calcolo")
cp1, cp2, cp3 = st.columns(3)

with cp1:
    complexity = st.selectbox("Complessità", [1, 2, 3], index=int(g("complexity", 1)) - 1)
with cp2:
    search_mode = st.selectbox(
        "Modi da ricercare",
        ["1° modo", "primi 20", "tutti"],
        index=["1° modo", "primi 20", "tutti"].index(g("search_mode", "1° modo")),
    )
with cp3:
    plate_behaviour = st.checkbox(
        "Plate behaviour σcr,p (Annex C semplificato)",
        value=bool(g("plate_behaviour", False)),
    )

# =========================
# Costruzione input modello
# =========================
inp = build_plate_input(
    a=a,
    b=b,
    t=t,
    E=E,
    nu=nu,
    unit=unit,
    edge_top=edge_top,
    edge_bottom=edge_bottom,
    edge_left=edge_left,
    edge_right=edge_right,
    kr_top=kr_top,
    kr_bottom=kr_bottom,
    kr_left=kr_left,
    kr_right=kr_right,
    J_top=J_top,
    J_bottom=J_bottom,
    J_left=J_left,
    J_right=J_right,
    beta_x=beta_x if ortho else 0.0,
    eta_x=eta_x if ortho else 0.0,
    beta_y=beta_y if ortho else 0.0,
    eta_y=eta_y if ortho else 0.0,
    stiffeners=props_df,
    s_xtl=s_xtl,
    s_xbl=s_xbl,
    s_xtr=s_xtr,
    s_xbr=s_xbr,
    s_yut=s_yut,
    s_yub=s_yub,
    s_ypt=s_ypt,
    s_ypb=s_ypb,
    c_t=c_t,
    c_b=c_b,
    tau_u=tau_u,
    imposed_x=imposed_x,
    imposed_y=imposed_y,
    imposed_tau=imposed_tau,
    patch_with_flanges=patch_with_flanges,
    mesh_df=mesh_df,
    complexity=complexity,
    search_mode=search_mode,
    plate_behaviour=plate_behaviour,
)

# =========================
# Preview pre-analisi
# =========================
st.subheader("Anteprima pre-analisi: geometria, vincoli e irrigidimenti")
pg1, pg2 = st.columns([1.25, 0.75])

with pg1:
    st.plotly_chart(make_plate_preview_figure(inp), use_container_width=True)

with pg2:
    st.dataframe(make_stiffener_summary_df(inp.stiffeners), use_container_width=True, height=360)
    st.caption(
        "Legenda vincoli: verde = semplice/hinged, rosso = fisso, arancione = elastico. "
        "Gli irrigidimenti longitudinali sono tracciati in blu; quelli trasversali in viola."
    )

# =========================
# Calcolo
# =========================
if st.button("Calcola instabilità elastica della piastra", type="primary", use_container_width=True):
    result = solve_buckling_problem(inp)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("φcr", f"{result['phi_cr']:.4f}")
    k2.metric("σx,cr [MPa]", f"{result['sigma_x_cr']:.2f}")
    k3.metric("σy,cr [MPa]", f"{result['sigma_y_cr']:.2f}")
    k4.metric("τcr [MPa]", f"{result['tau_cr']:.2f}")

    T1, T2, T3, T4, T5, T6 = st.tabs(
        ["Sintesi", "Geometria", "Check stresses", "Calcolo", "Post processing", "Export"]
    )

    with T1:
        lc, rc = st.columns([1, 1])
        with lc:
            st.dataframe(summary_results_df(result), use_container_width=True, height=300)
            st.dataframe(summary_model_df(inp), use_container_width=True, height=420)
        with rc:
            st.dataframe(result["modes_df"], use_container_width=True, height=250)
            st.plotly_chart(make_mode_figure(inp, result, 0), use_container_width=True)

    with T2:
        st.plotly_chart(make_plate_preview_figure(inp), use_container_width=True)
        st.dataframe(make_stiffener_summary_df(inp.stiffeners), use_container_width=True, height=320)

    with T3:
        sx_fun, sy_fun, tau_fun = analytical_stress_functions(inp)
        st.plotly_chart(make_stress_surface(inp, sx_fun, "sigma_x"), use_container_width=True)
        st.plotly_chart(make_stress_surface(inp, sy_fun, "sigma_y"), use_container_width=True)
        st.plotly_chart(make_stress_surface(inp, tau_fun, "tau"), use_container_width=True)

    with T4:
        st.markdown("### Log di calcolo")
        st.dataframe(result["calc_log"], use_container_width=True, height=420)

        st.markdown("### Coefficienti Aij del 1° modo")
        st.dataframe(make_aij_table(result, mode_index=0), use_container_width=True, height=300)

    with T5:
        if len(result["phi_positive"]) > 0:
            mode_idx = st.selectbox(
                "Modo",
                list(range(len(result["phi_positive"]))),
                format_func=lambda i: f"Modo {i+1} – φ={result['phi_positive'][i]:.4f}",
            )
            st.plotly_chart(make_mode_figure(inp, result, mode_idx), use_container_width=True)
            st.dataframe(make_aij_table(result, mode_index=mode_idx), use_container_width=True, height=280)
        else:
            st.warning("Nessun modo positivo disponibile.")

    with T6:
        st.download_button(
            "Scarica caso JSON",
            export_case_json(inp),
            "ebplatelite_case.json",
            "application/json",
        )
        st.download_button(
            "Scarica modi CSV",
            result["modes_df"].to_csv(index=False).encode("utf-8"),
            "ebplatelite_modes.csv",
            "text/csv",
        )
        st.download_button(
            "Scarica Aij CSV",
            make_aij_table(result, 0).to_csv(index=False).encode("utf-8"),
            "ebplatelite_aij.csv",
            "text/csv",
        )
else:
    st.info(
        "Ora puoi **vedere la geometria, i simboli dei vincoli e gli irrigidimenti prima del calcolo**. "
        "Quando sei soddisfatto della configurazione, premi **Calcola instabilità elastica della piastra**."
    )