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


# =========================================================
# Data model
# =========================================================
@dataclass
class PlateInput:
    a: float
    b: float
    t: float
    E: float
    nu: float
    unit: str

    edge_top: str
    edge_bottom: str
    edge_left: str
    edge_right: str

    kr_top: float
    kr_bottom: float
    kr_left: float
    kr_right: float

    J_top: float
    J_bottom: float
    J_left: float
    J_right: float

    beta_x: float
    eta_x: float
    beta_y: float
    eta_y: float

    stiffeners: pd.DataFrame

    s_xtl: float
    s_xbl: float
    s_xtr: float
    s_xbr: float

    s_yut: float
    s_yub: float
    s_ypt: float
    s_ypb: float

    c_t: float
    c_b: float

    tau_u: float

    imposed_x: bool
    imposed_y: bool
    imposed_tau: bool

    patch_with_flanges: bool
    mesh_df: pd.DataFrame | None

    complexity: int
    search_mode: str
    plate_behaviour: bool


# =========================================================
# Utilities
# =========================================================
def _f(unit: str) -> float:
    return {"mm": 1.0, "cm": 10.0, "m": 1000.0}[unit]


def _mm(v: float, unit: str) -> float:
    return float(v) * _f(unit)


def _mm2(v: float, unit: str) -> float:
    return float(v) * _f(unit) ** 2


def _mm4(v: float, unit: str) -> float:
    return float(v) * _f(unit) ** 4


def build_plate_input(**kwargs) -> PlateInput:
    return PlateInput(**kwargs)


# =========================================================
# Stiffeners
# =========================================================
def default_stiffeners_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "active": False,
                "orientation": "longitudinale",
                "type": "trapezoid",
                "location": 750.0,
                "A": 0.0,
                "I": 0.0,
                "J": 0.0,
                "b1": 130.0,
                "b2": 85.0,
                "h": 130.0,
                "tf": 9.0,
                "tw": 9.0,
                "ts": 9.0,
                "d": 120.0,
                "Kr_local": 0.0,
            }
        ]
    )


def clean_stiffeners_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return default_stiffeners_df().iloc[0:0].copy()

    out = df.copy()
    cols = [
        "active",
        "orientation",
        "type",
        "location",
        "A",
        "I",
        "J",
        "b1",
        "b2",
        "h",
        "tf",
        "tw",
        "ts",
        "d",
        "Kr_local",
    ]

    for c in cols:
        if c not in out.columns:
            out[c] = 0.0

    out["active"] = out["active"].fillna(False).astype(bool)
    out["orientation"] = out["orientation"].fillna("longitudinale")
    out["type"] = out["type"].fillna("general")

    for c in [c for c in cols if c not in ["active", "orientation", "type"]]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out


def compute_stiffener_properties(df: pd.DataFrame, t: float, unit: str) -> pd.DataFrame:
    out = clean_stiffeners_df(df)
    if out.empty:
        return out

    tp = _mm(t, unit)
    rows = []

    for _, r in out.iterrows():
        typ = r["type"]
        b1 = _mm(r.get("b1", 0.0), unit)
        b2 = _mm(r.get("b2", 0.0), unit)
        h = _mm(r.get("h", 0.0), unit)
        tf = max(_mm(r.get("tf", 0.0), unit), 1e-9)
        tw = max(_mm(r.get("tw", 0.0), unit), 1e-9)
        ts = max(_mm(r.get("ts", 0.0), unit), 1e-9)
        d = _mm(r.get("d", 0.0), unit)

        # effective width k*t each side (consistent with manual concept)
        k = 15.0
        leff = k * tp

        if typ == "flat bar":
            A = h * tf
            I = tf * h**3 / 12.0 + 2 * leff * tp**3 / 12.0
            J = h * tf**3 / 3.0
            Kr_local = 0.0
            Jt_local = 0.0

        elif typ == "sym flat bar":
            A = 2 * h * tf
            I = 2 * (tf * h**3 / 12.0) + 2 * leff * tp**3 / 12.0
            J = 2 * h * tf**3 / 3.0
            Kr_local = 0.0
            Jt_local = 0.0

        elif typ == "T":
            A = b1 * tf + h * tw
            I = b1 * tf**3 / 12.0 + tw * h**3 / 12.0 + 2 * leff * tp**3 / 12.0
            J = b1 * tf**3 / 3.0 + h * tw**3 / 3.0
            Kr_local = 0.0
            Jt_local = 0.0

        elif typ == "angle":
            A = b1 * tf + h * tw - tw * tf
            I = b1 * tf**3 / 12.0 + tw * h**3 / 12.0 + 2 * leff * tp**3 / 12.0
            J = b1 * tf**3 / 3.0 + h * tw**3 / 3.0
            Kr_local = 0.0
            Jt_local = 0.0

        elif typ == "trapezoid":
            # simplified closed section handling
            A = (b2 + 2 * h) * ts
            I = b2 * ts**3 / 12.0 + 2 * (ts * h**3 / 12.0) + 2 * leff * tp**3 / 12.0
            per = max(2 * h + b1 + b2, 1.0)
            Acell = max((b1 + b2) * h / 2.0, 1.0)
            J = 4 * Acell**2 * ts / per

            # local wall stiffness inspired by EBPlate trapezoidal treatment
            Kr_local = 8 * 210000.0 * ts**3 / max(h, 1e-9)
            delta1 = 0.00125 * h / ts - 0.05
            delta1 = min(max(delta1, 0.0), 0.7)
            Jt_local = (1 - delta1) * h * ts**3 / 9.0

        else:  # general
            A = _mm2(r.get("A", 0.0), unit)
            I = _mm4(r.get("I", 0.0), unit)
            J = _mm4(r.get("J", 0.0), unit)
            Kr_local = float(r.get("Kr_local", 0.0))
            Jt_local = _mm4(r.get("J", 0.0), unit) if d > 30 else 0.0

        D = 210000.0 * tp**3 / (12 * (1 - 0.3**2))

        # relative parameters in spirit of EBPlate
        delta = A / max(tp * 1000.0, 1e-9)
        gamma = 210000.0 * I / max(1000.0 * D, 1e-9)
        theta = 81000.0 * J / max(1000.0 * D, 1e-9)

        r2 = dict(r)
        r2.update(
            {
                "A_eff": A,
                "I_eff": I,
                "J_eff": J,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "Kr_local": Kr_local,
                "Jt_local": Jt_local,
            }
        )
        rows.append(r2)

    return pd.DataFrame(rows)


def orthotropy_from_smearing(df: pd.DataFrame, a: float, b: float, t: float, unit: str):
    f = _f(unit)
    a_mm = a * f
    b_mm = b * f
    t_mm = t * f

    active = df[df["active"]]
    longi = active[active["orientation"] == "longitudinale"]
    trans = active[active["orientation"] == "trasversale"]

    bx = ex = by = ey = 0.0

    if len(longi) > 0:
        I = float(longi["I_eff"].mean())
        A = float(longi["A_eff"].mean())
        ds = b_mm / (len(longi) + 1)
        bx = 12 * (1 - 0.3**2) * I / max(ds * t_mm**3, 1e-9)
        ex = A / max(ds * t_mm, 1e-9)

    if len(trans) > 0:
        I = float(trans["I_eff"].mean())
        A = float(trans["A_eff"].mean())
        ds = a_mm / (len(trans) + 1)
        by = 12 * (1 - 0.3**2) * I / max(ds * t_mm**3, 1e-9)
        ey = A / max(ds * t_mm, 1e-9)

    return bx, ex, by, ey


def make_stiffener_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["active", "orientation", "type", "location"])

    cols = [c for c in ["active", "orientation", "type", "location", "delta", "gamma", "theta"] if c in df.columns]
    return df[cols].copy()


# =========================================================
# Stress definitions
# =========================================================
def _mesh_fun(inp: PlateInput, col: str):
    if inp.mesh_df is None or col not in inp.mesh_df.columns:
        return lambda x, y: 0.0

    df = inp.mesh_df.copy()
    xs = pd.to_numeric(df["x"], errors="coerce").fillna(0.0).to_numpy()
    ys = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).to_numpy()
    vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()

    a = _mm(inp.a, inp.unit)
    b = _mm(inp.b, inp.unit)

    if xs.max() <= 1.0 + 1e-9:
        xs = xs * a
    if ys.max() <= 1.0 + 1e-9:
        ys = ys * b

    def fun(x, y):
        d = (xs - x) ** 2 + (ys - y) ** 2
        return float(vals[int(np.argmin(d))])

    return fun


def analytical_stress_functions(inp: PlateInput):
    a = _mm(inp.a, inp.unit)
    b = _mm(inp.b, inp.unit)

    sx_mesh = _mesh_fun(inp, "sigma_x")
    sy_mesh = _mesh_fun(inp, "sigma_y")
    tau_mesh = _mesh_fun(inp, "tau")

    def sx(x, y):
        xr = x / max(a, 1e-9)
        yr = y / max(b, 1e-9)
        top = inp.s_xtl * (1 - xr) + inp.s_xtr * xr
        bot = inp.s_xbl * (1 - xr) + inp.s_xbr * xr
        return top * (1 - yr) + bot * yr + sx_mesh(x, y)

    def patch(x, c, s):
        if c <= 0 or s == 0:
            return 0.0
        x0 = a / 2.0 - c / 2.0
        x1 = a / 2.0 + c / 2.0
        return float(s) if (x0 <= x <= x1) else 0.0

    def sy(x, y):
        yr = y / max(b, 1e-9)
        uni = inp.s_yut * (1 - yr) + inp.s_yub * yr
        loc = patch(x, _mm(inp.c_t, inp.unit), inp.s_ypt) * (1 - yr) + patch(
            x, _mm(inp.c_b, inp.unit), inp.s_ypb
        ) * yr
        return uni + loc + sy_mesh(x, y)

    def tau(x, y):
        base = inp.tau_u + tau_mesh(x, y)
        if inp.s_ypt == 0 and inp.s_ypb == 0:
            return base

        ctm = _mm(inp.c_t, inp.unit)
        cbm = _mm(inp.c_b, inp.unit)

        if inp.patch_with_flanges:
            tau_t = inp.s_ypt * ctm / max(2 * b, 1e-9)
            tau_b = inp.s_ypb * cbm / max(2 * b, 1e-9)
        else:
            par = 4 * (y / b) * (1 - y / b)
            tau_t = inp.s_ypt * ctm / max(2 * b, 1e-9) * par
            tau_b = inp.s_ypb * cbm / max(2 * b, 1e-9) * par

        return base + tau_t + tau_b

    return sx, sy, tau


# =========================================================
# Fourier basis + eigenproblem
# =========================================================
def _terms(inp: PlateInput):
    return (10, 10) if inp.complexity == 1 else (20, 20) if inp.complexity == 2 else (30, 30)


def _edge_penalty(kind: str, kr: float):
    if kind == "Fisso":
        return 1e8
    if kind == "Elastico":
        return max(float(kr), 0.0)
    return 0.0


def _basis(m, n):
    return [(i, j) for i in range(1, m + 1) for j in range(1, n + 1)]


def _shape_x(m, n, x, y, a, b):
    return (m * np.pi / a) * np.cos(m * np.pi * x / a) * np.sin(n * np.pi * y / b)


def _shape_y(m, n, x, y, a, b):
    return (n * np.pi / b) * np.sin(m * np.pi * x / a) * np.cos(n * np.pi * y / b)


def _shape_xx(m, n, x, y, a, b):
    return -((m * np.pi / a) ** 2) * np.sin(m * np.pi * x / a) * np.sin(n * np.pi * y / b)


def _shape_yy(m, n, x, y, a, b):
    return -((n * np.pi / b) ** 2) * np.sin(m * np.pi * x / a) * np.sin(n * np.pi * y / b)


def _shape_xy(m, n, x, y, a, b):
    return (m * np.pi / a) * (n * np.pi / b) * np.cos(m * np.pi * x / a) * np.cos(n * np.pi * y / b)


def _shape(m, n, x, y, a, b):
    return np.sin(m * np.pi * x / a) * np.sin(n * np.pi * y / b)


def _integrate_plate(inp: PlateInput, fun):
    a = _mm(inp.a, inp.unit)
    b = _mm(inp.b, inp.unit)

    nx = 35 if inp.complexity == 1 else 45 if inp.complexity == 2 else 55
    ny = 25 if inp.complexity == 1 else 35 if inp.complexity == 2 else 45

    xs = np.linspace(0, a, nx)
    ys = np.linspace(0, b, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    V = fun(X, Y)
    return np.trapz(np.trapz(V, xs, axis=1), ys, axis=0)


def solve_buckling_problem(inp: PlateInput):
    a = _mm(inp.a, inp.unit)
    b = _mm(inp.b, inp.unit)
    t = _mm(inp.t, inp.unit)

    D = inp.E * t**3 / (12 * (1 - inp.nu**2))
    Dx = D * (1 + inp.beta_x)
    Dy = D * (1 + inp.beta_y)

    fact_x = 0.0 if inp.imposed_x else 1.0
    fact_y = 0.0 if inp.imposed_y else 1.0
    fact_t = 0.0 if inp.imposed_tau else 1.0

    sx, sy, tau = analytical_stress_functions(inp)

    # Simplified Annex C plate behaviour
    stiff_df = inp.stiffeners.copy() if inp.stiffeners is not None else pd.DataFrame()
    sigma_top_mid = max(inp.s_xtl, inp.s_xtr)
    sigma_bot_mid = max(inp.s_xbl, inp.s_xbr)

    if inp.plate_behaviour and not stiff_df.empty:
        fact_x = 0.0
        comp = stiff_df["orientation"].eq("longitudinale") & stiff_df["active"]
        for idx in stiff_df[comp].index:
            loc = _mm(stiff_df.loc[idx, "location"], inp.unit)
            s_loc = sigma_top_mid * (1 - loc / max(b, 1e-9)) + sigma_bot_mid * (loc / max(b, 1e-9))
            if s_loc > 0:
                stiff_df.loc[idx, "A_eff"] = float(stiff_df.loc[idx, "A_eff"]) * 2.2

    mx, ny = _terms(inp)
    basis = _basis(mx, ny)
    N = len(basis)

    R0 = np.zeros((N, N), float)
    RG = np.zeros((N, N), float)

    kp = {
        "top": _edge_penalty(inp.edge_top, inp.kr_top),
        "bottom": _edge_penalty(inp.edge_bottom, inp.kr_bottom),
        "left": _edge_penalty(inp.edge_left, inp.kr_left),
        "right": _edge_penalty(inp.edge_right, inp.kr_right),
    }

    for i, (mi, ni) in enumerate(basis):
        for j, (mj, nj) in enumerate(basis):

            def f0(X, Y):
                wxx_i = _shape_xx(mi, ni, X, Y, a, b)
                wyy_i = _shape_yy(mi, ni, X, Y, a, b)
                wxy_i = _shape_xy(mi, ni, X, Y, a, b)

                wxx_j = _shape_xx(mj, nj, X, Y, a, b)
                wyy_j = _shape_yy(mj, nj, X, Y, a, b)
                wxy_j = _shape_xy(mj, nj, X, Y, a, b)

                return (
                    Dx * wxx_i * wxx_j
                    + Dy * wyy_i * wyy_j
                    + 2 * D * inp.nu * wxx_i * wyy_j
                    + 2 * D * (1 - inp.nu) * wxy_i * wxy_j
                )

            def fg(X, Y):
                wx_i = _shape_x(mi, ni, X, Y, a, b)
                wy_i = _shape_y(mi, ni, X, Y, a, b)
                wx_j = _shape_x(mj, nj, X, Y, a, b)
                wy_j = _shape_y(mj, nj, X, Y, a, b)

                return t * (
                    fact_x * np.vectorize(sx)(X, Y) * wx_i * wx_j
                    + fact_y * np.vectorize(sy)(X, Y) * wy_i * wy_j
                    + 2 * fact_t * np.vectorize(tau)(X, Y) * wx_i * wy_j
                )

            R0[i, j] = _integrate_plate(inp, f0)
            RG[i, j] = _integrate_plate(inp, fg)

            xs = np.linspace(0, a, 100)
            ys = np.linspace(0, b, 100)

            if kp["top"] > 0:
                R0[i, j] += np.trapz(
                    kp["top"] * _shape_y(mi, ni, xs, 0.0, a, b) * _shape_y(mj, nj, xs, 0.0, a, b),
                    xs,
                )
            if kp["bottom"] > 0:
                R0[i, j] += np.trapz(
                    kp["bottom"] * _shape_y(mi, ni, xs, b, a, b) * _shape_y(mj, nj, xs, b, a, b),
                    xs,
                )
            if kp["left"] > 0:
                R0[i, j] += np.trapz(
                    kp["left"] * _shape_x(mi, ni, 0.0, ys, a, b) * _shape_x(mj, nj, 0.0, ys, a, b),
                    ys,
                )
            if kp["right"] > 0:
                R0[i, j] += np.trapz(
                    kp["right"] * _shape_x(mi, ni, a, ys, a, b) * _shape_x(mj, nj, a, ys, a, b),
                    ys,
                )

    if stiff_df is not None and len(stiff_df) > 0:
        active = stiff_df[stiff_df["active"]]

        for _, st in active.iterrows():
            ori = st["orientation"]
            loc = _mm(st["location"], inp.unit)
            A = float(st["A_eff"])
            I = float(st["I_eff"])
            J = float(st["J_eff"])
            Kr_loc = float(st.get("Kr_local", 0.0))
            Jt_loc = float(st.get("Jt_local", 0.0))

            for i, (mi, ni) in enumerate(basis):
                for j, (mj, nj) in enumerate(basis):
                    if ori == "longitudinale":
                        xs = np.linspace(0, a, 140)

                        R0[i, j] += np.trapz(
                            inp.E
                            * I
                            * _shape_xx(mi, ni, xs, loc, a, b)
                            * _shape_xx(mj, nj, xs, loc, a, b)
                            + 81000.0
                            * J
                            * _shape_xy(mi, ni, xs, loc, a, b)
                            * _shape_xy(mj, nj, xs, loc, a, b),
                            xs,
                        )

                        if Kr_loc > 0:
                            R0[i, j] += np.trapz(
                                Kr_loc
                                * _shape_y(mi, ni, xs, loc, a, b)
                                * _shape_y(mj, nj, xs, loc, a, b),
                                xs,
                            )

                        if Jt_loc > 0:
                            R0[i, j] += np.trapz(
                                81000.0
                                * Jt_loc
                                * _shape_xy(mi, ni, xs, loc, a, b)
                                * _shape_xy(mj, nj, xs, loc, a, b),
                                xs,
                            )

                        sig = np.array([sx(xv, loc) for xv in xs])
                        RG[i, j] += np.trapz(
                            A
                            * sig
                            * _shape_x(mi, ni, xs, loc, a, b)
                            * _shape_x(mj, nj, xs, loc, a, b),
                            xs,
                        )

                    else:  # trasversale
                        ys = np.linspace(0, b, 140)

                        R0[i, j] += np.trapz(
                            inp.E
                            * I
                            * _shape_yy(mi, ni, loc, ys, a, b)
                            * _shape_yy(mj, nj, loc, ys, a, b)
                            + 81000.0
                            * J
                            * _shape_xy(mi, ni, loc, ys, a, b)
                            * _shape_xy(mj, nj, loc, ys, a, b),
                            ys,
                        )

                        sig = np.array([sy(loc, yv) for yv in ys])
                        RG[i, j] += np.trapz(
                            A
                            * sig
                            * _shape_y(mi, ni, loc, ys, a, b)
                            * _shape_y(mj, nj, loc, ys, a, b),
                            ys,
                        )

    R0 = 0.5 * (R0 + R0.T)
    RG = 0.5 * (RG + RG.T)
    R0 = R0 + np.eye(N) * max(np.linalg.norm(R0, ord="fro") * 1e-12, 1e-9)

    if SCIPY_OK:
        eigvals, eigvecs = sla.eig(R0, RG + np.eye(N) * 1e-12)
    else:
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(RG + np.eye(N) * 1e-12) @ R0)

    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    mask = np.isfinite(eigvals) & (eigvals > 1e-8)
    pos_vals = eigvals[mask]
    pos_vecs = eigvecs[:, mask]

    order = np.argsort(pos_vals)
    pos_vals = pos_vals[order]
    pos_vecs = pos_vecs[:, order] if pos_vecs.size else pos_vecs

    nm = 1 if inp.search_mode == "1° modo" else 20 if inp.search_mode == "primi 20" else len(pos_vals)
    phi_positive = pos_vals[:nm]

    eig_list = []
    for k in range(min(nm, pos_vecs.shape[1] if pos_vecs.ndim == 2 else 0)):
        v = pos_vecs[:, k]
        mxv = np.max(np.abs(v)) if np.max(np.abs(v)) > 0 else 1.0
        eig_list.append(v / mxv)

    phi_cr = float(phi_positive[0]) if len(phi_positive) else float("nan")

    sigma_x_ref = max(abs(inp.s_xtl), abs(inp.s_xbl), abs(inp.s_xtr), abs(inp.s_xbr), 1e-9)
    sigma_y_ref = max(abs(inp.s_yut), abs(inp.s_yub), abs(inp.s_ypt), abs(inp.s_ypb), 1e-9)
    tau_ref = max(abs(inp.tau_u), 1e-9)

    sigma_x_cr = sigma_x_ref if inp.imposed_x else phi_cr * sigma_x_ref
    sigma_y_cr = sigma_y_ref if inp.imposed_y else phi_cr * sigma_y_ref
    tau_cr = tau_ref if inp.imposed_tau else phi_cr * tau_ref

    modes_df = pd.DataFrame({"Modo": np.arange(1, len(phi_positive) + 1), "phi": phi_positive})

    calc_log = pd.DataFrame(
        [
            ("Dimensione matrici", f"{N} x {N}"),
            ("m_max", mx),
            ("n_max", ny),
            ("D [N mm]", D),
            ("Dx [N mm]", Dx),
            ("Dy [N mm]", Dy),
            (
                "N stiffeners attivi",
                int(len(stiff_df[stiff_df["active"]])) if stiff_df is not None and len(stiff_df) else 0,
            ),
            ("Plate behaviour", inp.plate_behaviour),
            ("Autovalori positivi trovati", int(len(phi_positive))),
            ("phi_cr", phi_cr),
        ],
        columns=["Parametro", "Valore"],
    )

    return {
        "phi_cr": phi_cr,
        "phi_positive": phi_positive,
        "eigenvectors": eig_list,
        "basis_modes": basis,
        "a_mm": a,
        "b_mm": b,
        "sigma_x_cr": sigma_x_cr,
        "sigma_y_cr": sigma_y_cr,
        "tau_cr": tau_cr,
        "modes_df": modes_df,
        "calc_log": calc_log,
    }


# =========================================================
# Results / figures
# =========================================================
def _mode_surface(inp: PlateInput, result: dict, mode_index: int, nx=60, ny=40):
    a = result["a_mm"]
    b = result["b_mm"]

    Xv = np.linspace(0, a, nx)
    Yv = np.linspace(0, b, ny)
    X, Y = np.meshgrid(Xv, Yv, indexing="xy")
    Z = np.zeros_like(X)

    vec = result["eigenvectors"][mode_index]
    for c, (m, n) in zip(vec, result["basis_modes"]):
        Z += c * _shape(m, n, X, Y, a, b)

    return X, Y, Z


def make_mode_figure(inp: PlateInput, result: dict, mode_index: int = 0):
    X, Y, Z = _mode_surface(inp, result, mode_index)
    fig = go.Figure(
        data=[go.Surface(x=X / _f(inp.unit), y=Y / _f(inp.unit), z=Z, colorscale="Turbo")]
    )
    fig.update_layout(
        template="plotly_white",
        height=430,
        title=f"Modo di buckling {mode_index + 1}",
        scene=dict(
            xaxis_title=f"x [{inp.unit}]",
            yaxis_title=f"y [{inp.unit}]",
            zaxis_title="w norm.",
        ),
    )
    return fig


def make_aij_table(result: dict, mode_index: int = 0):
    if len(result["eigenvectors"]) == 0:
        return pd.DataFrame(columns=["m", "n", "a_mn"])

    vec = result["eigenvectors"][mode_index]
    return pd.DataFrame(
        [{"m": m, "n": n, "a_mn": aij} for aij, (m, n) in zip(vec, result["basis_modes"])]
    )


def summary_results_df(result: dict):
    return pd.DataFrame(
        [
            ("φcr", result["phi_cr"]),
            ("σx,cr [MPa]", result["sigma_x_cr"]),
            ("σy,cr [MPa]", result["sigma_y_cr"]),
            ("τcr [MPa]", result["tau_cr"]),
        ],
        columns=["Parametro", "Valore"],
    )


def summary_model_df(inp: PlateInput):
    return pd.DataFrame(
        [
            ("a", inp.a),
            ("b", inp.b),
            ("t", inp.t),
            ("E [MPa]", inp.E),
            ("ν", inp.nu),
            ("Bordo alto", inp.edge_top),
            ("Bordo basso", inp.edge_bottom),
            ("Bordo sinistro", inp.edge_left),
            ("Bordo destro", inp.edge_right),
            ("βx", inp.beta_x),
            ("ηx", inp.eta_x),
            ("βy", inp.beta_y),
            ("ηy", inp.eta_y),
            ("σx tl", inp.s_xtl),
            ("σx bl", inp.s_xbl),
            ("σx tr", inp.s_xtr),
            ("σx br", inp.s_xbr),
            ("σy ut", inp.s_yut),
            ("σy ub", inp.s_yub),
            ("σy pt", inp.s_ypt),
            ("σy pb", inp.s_ypb),
            ("τu", inp.tau_u),
            ("Complessità", inp.complexity),
            ("Search mode", inp.search_mode),
        ],
        columns=["Parametro", "Valore"],
    )


def export_case_json(inp: PlateInput):
    d = asdict(inp)
    d["stiffeners"] = inp.stiffeners.to_dict(orient="records") if isinstance(inp.stiffeners, pd.DataFrame) else []
    d["mesh_df"] = inp.mesh_df.to_dict(orient="records") if isinstance(inp.mesh_df, pd.DataFrame) else None
    return json.dumps(d, ensure_ascii=False, indent=2).encode("utf-8")


def make_stress_surface(inp: PlateInput, stress_fun, name: str):
    a = _mm(inp.a, inp.unit)
    b = _mm(inp.b, inp.unit)

    Xv = np.linspace(0, a, 50)
    Yv = np.linspace(0, b, 30)
    X, Y = np.meshgrid(Xv, Yv, indexing="xy")
    Z = np.vectorize(stress_fun)(X, Y)

    fig = go.Figure(
        data=[go.Surface(x=X / _f(inp.unit), y=Y / _f(inp.unit), z=Z, colorscale="RdBu_r")]
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        title=f"{name}(x,y)",
        scene=dict(
            xaxis_title=f"x [{inp.unit}]",
            yaxis_title=f"y [{inp.unit}]",
            zaxis_title=f"{name} [MPa]",
        ),
    )
    return fig


# =========================================================
# Geometry preview with symbols by support
# =========================================================
def _edge_style(kind: str):
    # Plotly-valid dash values only
    if kind == "Fisso":
        return "#dc2626", 6, "solid"
    if kind == "Elastico":
        return "#f59e0b", 5, "dash"
    return "#16a34a", 4, "solid"


def make_plate_preview_figure(inp: PlateInput):
    a = inp.a
    b = inp.b

    fig = go.Figure()

    # plate area
    fig.add_trace(
        go.Scatter(
            x=[0, a, a, 0, 0],
            y=[0, 0, b, b, 0],
            mode="lines",
            fill="toself",
            fillcolor="rgba(59,130,246,0.08)",
            line=dict(color="rgba(30,64,175,0.5)", width=1),
            name="Piastra",
        )
    )

    # edges
    edge_data = [
        ([0, a], [b, b], inp.edge_top, "Bordo alto"),
        ([0, a], [0, 0], inp.edge_bottom, "Bordo basso"),
        ([0, 0], [0, b], inp.edge_left, "Bordo sinistro"),
        ([a, a], [0, b], inp.edge_right, "Bordo destro"),
    ]

    for xs, ys, kind, name in edge_data:
        color, width, dash = _edge_style(kind)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                name=f"{name}: {kind}",
            )
        )

    # symbolic markers based on support type
    nmark = 14

    # top and bottom
    for kind, y in [(inp.edge_top, b), (inp.edge_bottom, 0)]:
        color, _, _ = _edge_style(kind)
        xs = np.linspace(0, a, nmark)

        if kind == "Semplice/hinged":
            symbol = "triangle-down" if y == b else "triangle-up"
            size = 8
        elif kind == "Fisso":
            symbol = "square"
            size = 7
        else:  # elastico
            symbol = "diamond"
            size = 7

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=np.full(nmark, y),
                mode="markers",
                marker=dict(color=color, size=size, symbol=symbol),
                showlegend=False,
            )
        )

    # left and right
    for kind, x in [(inp.edge_left, 0), (inp.edge_right, a)]:
        color, _, _ = _edge_style(kind)
        ys = np.linspace(0, b, nmark)

        if kind == "Semplice/hinged":
            symbol = "triangle-right" if x == 0 else "triangle-left"
            size = 8
        elif kind == "Fisso":
            symbol = "square"
            size = 7
        else:
            symbol = "diamond"
            size = 7

        fig.add_trace(
            go.Scatter(
                x=np.full(nmark, x),
                y=ys,
                mode="markers",
                marker=dict(color=color, size=size, symbol=symbol),
                showlegend=False,
            )
        )

    # stiffeners
    if inp.stiffeners is not None and len(inp.stiffeners) > 0:
        active = inp.stiffeners[inp.stiffeners["active"]]

        for _, st in active.iterrows():
            loc = float(st["location"])

            if st["orientation"] == "longitudinale":
                fig.add_trace(
                    go.Scatter(
                        x=[0, a],
                        y=[loc, loc],
                        mode="lines",
                        line=dict(color="#2563eb", width=3),
                        name=f"Longitudinale @ {loc:g}",
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[loc, loc],
                        y=[0, b],
                        mode="lines",
                        line=dict(color="#7c3aed", width=3),
                        name=f"Trasversale @ {loc:g}",
                    )
                )

    # annotations
    fig.add_annotation(x=a / 2, y=b + 0.05 * max(b, 1), text="alto", showarrow=False, font=dict(size=12, color="#374151"))
    fig.add_annotation(x=a / 2, y=-0.06 * max(b, 1), text="basso", showarrow=False, font=dict(size=12, color="#374151"))
    fig.add_annotation(x=-0.06 * max(a, 1), y=b / 2, text="sinistro", showarrow=False, textangle=-90, font=dict(size=12, color="#374151"))
    fig.add_annotation(x=a + 0.06 * max(a, 1), y=b / 2, text="destro", showarrow=False, textangle=90, font=dict(size=12, color="#374151"))

    fig.update_layout(
        template="plotly_white",
        height=560,
        title="Anteprima geometrica della piastra con vincoli e irrigidimenti",
        xaxis_title=f"a [{inp.unit}]",
        yaxis_title=f"b [{inp.unit}]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig