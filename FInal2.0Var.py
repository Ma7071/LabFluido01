#!/usr/bin/env python3
import os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline, make_interp_spline
from scipy import integrate

# === 0) Constants ===
FREESTREAM_PRESSURE = 190    # Pa
FREESTREAM_VELOCITY = 20.0   # m/s
AIR_DENSITY        = 1.174   # kg/m³
CHORD              = 0.1     # m

# === 1) Original Results directories ===
RESULTS      = "Results"
DIRS = {
    "airfoil":    os.path.join(RESULTS, "airfoil_plots"),
    "interpol":   os.path.join(RESULTS, "interpolated_plots"),
    "cp":         os.path.join(RESULTS, "cp_plots"),
    "cl_alpha":   os.path.join(RESULTS, "cl_alpha_plots"),
    "cd_alpha":   os.path.join(RESULTS, "cd_alpha_plots"),
    "polar":      os.path.join(RESULTS, "polar_plots"),
    "xfoil":      os.path.join(RESULTS, "xfoil_plots"),
    "compare":    os.path.join(RESULTS, "comparison_plots"),
    "data":       os.path.join(RESULTS, "data")
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# === A) Variability (max/min) directories ===
DIR_VAR       = os.path.join(RESULTS, "variability")
DIR_VAR_MAX   = os.path.join(DIR_VAR, "max")
DIR_VAR_MIN   = os.path.join(DIR_VAR, "min")
for base in (DIR_VAR_MAX, DIR_VAR_MIN):
    for sub in DIRS:
        os.makedirs(os.path.join(base, os.path.basename(DIRS[sub])), exist_ok=True)

# === 2) Airfoil geometry & surface normal calc ===
def naca_23012(x):
    t, m, p = 0.12, 0.02, 0.3
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 +
              0.2843*x**3 - 0.1015*x**4)
    yc = np.where(x<p,
                  m*(x/p**2)*(2*p - x),
                  m*((1-x)/(1-p)**2)*(1 + x - 2*p))
    dyc = np.where(x<p,
                   (2*m/p**2)*(p - x),
                   (2*m/(1-p)**2)*(p - x))
    th = np.arctan(dyc)
    return yc + yt*np.cos(th), yc - yt*np.cos(th), dyc

def compute_normals(sensors, x_profile, dy_up, dy_lo):
    normals = {}
    for num, xs, zs in sensors:
        slope = np.interp(xs, x_profile, dy_up if zs>=0 else dy_lo)
        v = np.array([-slope, 1.0]); v /= np.linalg.norm(v)
        normals[num] = np.degrees(np.arctan2(v[1], v[0]))
    return normals

# === 3) Load pressure data (avg, max, min) ===
def load_pressure_data_extremes(data_dir="CLDATA", pattern="*_gradi.txt"):
    avg_p, max_p, min_p = {}, {}, {}
    for fname in sorted(glob.glob(os.path.join(data_dir, pattern))):
        try:
            α = float(os.path.basename(fname).split("_")[0])
            arr = np.loadtxt(fname)[:, :16]   # time × sensors
            avg_p[α] = arr.mean(axis=0)
            max_p[α] = arr.max(axis=0)
            min_p[α] = arr.min(axis=0)
        except Exception:
            continue
    return avg_p, max_p, min_p

# === 4) Interpolator & Cl from pressure array ===
def make_interpolators(sensors, avg_p):
    half = len(sensors)//2
    x_up, p_up = sensors[:half,1], avg_p[:half]
    x_lo, p_lo = sensors[half:,1], avg_p[half:]
    qinf = 0.5*AIR_DENSITY*FREESTREAM_VELOCITY**2
    cp_up, cp_lo = p_up/qinf, p_lo/qinf
    cp_te = 0.5*(cp_up[-1] + cp_lo[-1])

    xu, cu = np.append(x_up,1.0), np.append(cp_up,cp_te)
    xl, cll= np.append(x_lo,1.0), np.append(cp_lo,cp_te)

    cs_up = CubicSpline(xu, cu, extrapolate=True)
    cs_lo = CubicSpline(xl, cll, extrapolate=True)

    xi = np.linspace(0,1,1000)
    cl = integrate.simpson(cs_lo(xi) - cs_up(xi), xi)
    return {"upper": cs_up, "lower": cs_lo}, cl

# === 5) Wake‐derived drag ===
def compute_cd_from_wake(data_dir="CDdata"):
    alphas, CDs = [], []
    for fn in os.listdir(data_dir):
        if not fn.endswith('.txt'): continue
        α = float(fn[:-4])
        data = np.loadtxt(os.path.join(data_dir,fn),
                          delimiter='\t', skiprows=1)
        z_mm, _, qloc = data[:,0], data[:,1], data[:,2]
        z_m = np.unique(z_mm)/1000.0
        avg_q = np.array([qloc[z_mm==z].mean() for z in np.unique(z_mm)])
        qinf = qloc.max()
        Δ = np.sqrt(avg_q/qinf) - (avg_q/qinf)
        Dp = 2.0 * np.trapezoid(Δ, z_m)
        alphas.append(α); CDs.append(Dp/CHORD)
    alphas = np.array(alphas); CDs = np.array(CDs)
    idx = np.argsort(alphas)
    coeffs = np.polyfit(alphas[idx], CDs[idx], 2)
    return alphas[idx], CDs[idx], coeffs

def save_drag_data(alphas, CDs, coeffs, outdir):
    # CSV
    np.savetxt(os.path.join(outdir, "drag_polar.csv"),
               np.column_stack((alphas, alphas**2, CDs)),
               delimiter=",",
               header="alpha_deg,alpha2,CD",
               comments="")
    # coeffs
    with open(os.path.join(outdir, "drag_polar_coeffs.txt"), 'w') as f:
        f.write(f"Cd(α)={coeffs[0]:.6e}·α²+{coeffs[1]:.6e}·α+{coeffs[2]:.6e}\n")

# === 6) XFOIL I/O & plots ===
def read_xfoil(fname="rawdata_03.txt"):
    a, cl, cd = [], [], []
    for L in open(fname):
        ls = L.strip().lower()
        if not ls or ls.startswith('alpha'): continue
        α,c1,c2 = map(float, L.split()[:3])
        a.append(α); cl.append(c1); cd.append(c2)
    return np.array(a), np.array(cl), np.array(cd)

# === 7) Plot helpers (param: base_outdir) ===
def plot_airfoil(xp, yu, yl, sensors, normals, avg_p, α, base_out):
    plt.figure(figsize=(8, 6))

    # Plot airfoil shape
    plt.plot(xp, yu, 'b-', label='Upper Surface')
    plt.plot(xp, yl, 'r-', label='Lower Surface')

    # Plot pressure normals using annotate for proper arrowheads
    for i, (num, xs, zs) in enumerate(sensors):
        # Determine which surface we're on and compute slope
        surface = yu if zs >= 0 else yl
        slope = np.interp(xs, xp, np.gradient(surface, xp))

        # Compute normal vector and normalize it
        v = np.array([-slope, 1.0])
        v /= np.linalg.norm(v)

        # Scale arrow length based on relative pressure
        scale = 0.05 * (avg_p[i] / avg_p.mean())
        dx, dy = v[0] * scale, v[1] * scale

        # Draw arrow with head exactly at tip using annotate
        plt.annotate(
            "",
            xy=(xs + dx, zs + dy),     # arrow tip
            xytext=(xs, zs),           # arrow base
            arrowprops=dict(
                arrowstyle="->",
                color="g",
                lw=1.0,
                shrinkA=0,
                shrinkB=0
            )
        )

    # Labels and formatting
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel("x/c")
    plt.ylabel("z/c")
    plt.title(f"Airfoil + Normals @ {α:.1f}°")
    plt.legend()
    out_path = os.path.join(base_out, f"airfoil_{α:.1f}deg.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_cp(interps, α, base_out):
    x = np.linspace(0,1,1000)
    plt.figure(figsize=(6,4))
    plt.plot(x, interps["upper"](x), 'b-', label='Cp up')
    plt.plot(x, interps["lower"](x), 'r-', label='Cp lo')
    plt.gca().invert_yaxis()
    plt.legend(); plt.xlabel("x/c"); plt.ylabel("Cp")
    plt.title(f"Cp @ {α:.1f}°")
    plt.savefig(os.path.join(base_out, f"cp_{α:.1f}deg.png"), dpi=300)
    plt.close()

def plot_selected_cps(all_interps, angles, base_out):
    sel = [α for α in angles if α in all_interps]
    x = np.linspace(0,1,1000)
    plt.figure(figsize=(8,5))
    for α in sel:
        cs = all_interps[α]
        plt.plot(x, cs["upper"](x), '--', label=f'Up {α}°')
        plt.plot(x, cs["lower"](x), '-',  label=f'Lo {α}°')
    plt.gca().invert_yaxis(); plt.legend(ncol=2, fontsize=8)
    plt.xlabel("x/c"); plt.ylabel("Cp"); plt.title("Cp Comparison")
    plt.grid(True)
    plt.savefig(os.path.join(base_out, "selected_angles.png"), dpi=300)
    plt.close()

def plot_cl_alpha(alphas, cls, outpath, title="Cl vs α"):
    cs = CubicSpline(alphas, cls)
    af = np.linspace(alphas.min(), alphas.max(), 500)
    plt.figure(figsize=(6,4))
    plt.plot(alphas, cls, 'o', af, cs(af), '-')
    plt.xlabel("α (°)"); plt.ylabel("Cl"); plt.title(title)
    plt.grid(True)
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_cd_alpha(alphas, cds, outpath, title="Cd vs α"):
    # Ensure alphas are sorted (required for interpolation)
    sorted_indices = np.argsort(alphas)
    alphas_sorted = alphas[sorted_indices]
    cds_sorted = cds[sorted_indices]

    # Create quadratic spline (k=2)
    spline = make_interp_spline(alphas_sorted, cds_sorted, k=2)
    alpha_smooth = np.linspace(alphas_sorted[0], alphas_sorted[-1], 300)
    cd_smooth = spline(alpha_smooth)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(alpha_smooth, cd_smooth, color='darkorange', label="Quadratic Fit")
    plt.plot(alphas, cds, 'o', color='darkcyan', label="Data Points")
    plt.xlabel("α (°)")
    plt.ylabel("Cd")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_matched_polar_and_fits(αs, cls, αw, CDw, base_polar, base_interpol):
    common = np.intersect1d(αs, αw)
    if len(common)<1:
        print("No common α for polar."); return

    # matched
    Cl_c = np.array([cls[np.where(αs==a)[0][0]] for a in common])
    Cd_c = np.array([CDw[np.where(αw==a)[0][0]] for a in common])
    plt.figure(figsize=(6,5))
    plt.plot(Cd_c, Cl_c, 'o-', label='Matched Polar')
    plt.xlabel("Cd"); plt.ylabel("Cl"); plt.title("Matched Polar Curve")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(base_polar, "cl_vs_cd_matched.png"), dpi=300)
    plt.close()

    # super‐interpolation
    αc_arr, Cl_arr = np.unique(sorted(zip(αs, cls)), axis=0).T
    αd_arr, Cd_arr = np.unique(sorted(zip(αw, CDw)), axis=0).T
    α_min, α_max = max(αc_arr.min(), αd_arr.min()), min(αc_arr.max(), αd_arr.max())
    α_fine = np.linspace(α_min, α_max, 1000)
    spline_Cl = UnivariateSpline(αc_arr, Cl_arr, s=1e-8)
    spline_Cd = UnivariateSpline(αd_arr, Cd_arr, s=1e-8)
    Cl_fine, Cd_fine = spline_Cl(α_fine), spline_Cd(α_fine)

    plt.figure(figsize=(6,5))
    plt.plot(Cd_fine, Cl_fine, '--', label='Interpolated Polar')
    plt.xlabel("Cd"); plt.ylabel("Cl"); plt.title("Interpolated Polar Curve")
    plt.grid(True)
    plt.savefig(os.path.join(base_interpol, "interpolated_polar.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6,5))
    plt.plot(α_fine, Cd_fine, '--', label='Interpolated CD')
    plt.xlabel("α"); plt.ylabel("Cd"); plt.title("Interpolated CD Curve")
    plt.grid(True)
    plt.savefig(os.path.join(base_interpol, "interpolated_CD.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6,5))
    plt.plot(α_fine, Cl_fine, '--', label='Interpolated CL')
    plt.xlabel("α"); plt.ylabel("Cl"); plt.title("Interpolated CL Curve")
    plt.grid(True)
    plt.savefig(os.path.join(base_interpol, "interpolated_CL.png"), dpi=300)
    plt.close()

    # quadratic fit
    common_pts = sorted(set(αc_arr).intersection(αd_arr))
    Cl_c2 = np.array([dict(zip(αs,cls))[a] for a in common_pts])
    Cd_c2 = np.array([dict(zip(αw,CDw))[a] for a in common_pts])
    a,b,c = np.polyfit(Cl_c2, Cd_c2, 2)
    Cl_fit = np.linspace(Cl_c2.min(), Cl_c2.max(), 500)
    Cd_fit = a*Cl_fit**2 + b*Cl_fit + c

    plt.figure(figsize=(6,5))
    plt.plot(Cd_c2, Cl_c2, 'ro', label='Data')
    plt.plot(Cd_fit, Cl_fit, 'b--',
             label=f'Cd={a:.2e}Cl²+{b:.2e}Cl+{c:.2e}')
    plt.xlabel("Cd"); plt.ylabel("Cl"); plt.title("Quadratic Fit of CD vs CL")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(base_polar, "cl_vs_cd_quad_fit.png"), dpi=300)
    plt.close()

def plot_xfoil(αx, clx, cdx, base_out):
    plt.figure(figsize=(6,4))
    plt.plot(αx, clx, 'o-')
    plt.xlabel("α"); plt.ylabel("Cl"); plt.title("XFOIL Cl vs α")
    plt.grid(True); plt.savefig(os.path.join(base_out, "xfoil_cl_alpha.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(αx, cdx, 'o-')
    plt.xlabel("α"); plt.ylabel("Cd"); plt.title("XFOIL Cd vs α")
    plt.grid(True); plt.savefig(os.path.join(base_out, "xfoil_cd_alpha.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(cdx, clx, 'o-')
    plt.xlabel("Cd"); plt.ylabel("Cl"); plt.title("XFOIL Polar")
    plt.grid(True); plt.savefig(os.path.join(base_out, "xfoil_polar.png"), dpi=300)
    plt.close()

def compare_exp_xfoil(αe, cle, cde, αx, clx, cdx, base_out, qc_exp):
    """
    αe,cle,cde : experimental aoa, Cl, Cd
    αx,clx,cdx : XFOIL    aoa, Cl, Cd
    qc_exp      : tuple (a,b,c) of experimental Cd vs α quadratic fit
    """
    # — 1) Cl vs α (linear, 0 ≤ α ≤ 8) —
    mask_e = (αe >= 0) & (αe <= 8)
    mask_x = (αx >= 0) & (αx <= 8)
    αe_cl, cle_cl = αe[mask_e], cle[mask_e]
    αx_cl, clx_cl = αx[mask_x], clx[mask_x]

    # Calculate linear fit coefficients for Cl vs α (0≤α≤8)
    m1_e, q1_e = np.polyfit(αe_cl, cle_cl, 1)
    m1_x, q1_x = np.polyfit(αx_cl, clx_cl, 1)

    # Save lift curve data with explanations
    # Determine the correct data directory based on base_out (comparison directory)
    if base_out.startswith(DIR_VAR_MAX) or base_out.startswith(DIR_VAR_MIN):
        # We're in variability mode
        base_dir = os.path.dirname(base_out)  # Go up one level from compare dir
        data_dir = os.path.join(base_dir, os.path.basename(DIRS["data"]))
    else:
        # We're in normal mode
        data_dir = DIRS["data"]

    with open(os.path.join(data_dir, "lift_curve_coefficients.txt"), 'w') as f:
        f.write("# Lift Curve Linear Coefficients (Cl = m·α + q) for 0° ≤ α ≤ 8°\n")
        f.write("# These coefficients represent the linear portion of the lift curve\n")
        f.write("# where the relationship between lift coefficient (Cl) and angle of attack (α) is linear.\n\n")
        f.write("# Experimental Data:\n")
        f.write(f"m_exp = {m1_e:.6f}  # Lift curve slope (per degree)\n")
        f.write(f"q_exp = {q1_e:.6f}  # Lift coefficient at zero angle of attack\n\n")
        f.write("# XFOIL Data:\n")
        f.write(f"m_xfoil = {m1_x:.6f}  # Lift curve slope (per degree)\n")
        f.write(f"q_xfoil = {q1_x:.6f}  # Lift coefficient at zero angle of attack\n")

    # — 2) Cd vs α (quadratic formulas only) —
    a_cd_e, b_cd_e, c_cd_e = qc_exp
    # compute XFOIL quadratic for Cd vs α:
    a_cd_x, b_cd_x, c_cd_x = np.polyfit(αx, cdx, 2)

    # — 3) Polar: Cl vs Cd (quadratic formulas only) —
    # Check if we're using actual data points or values from the quadratic fit
    if len(αe) == len(cde):  # Actual data points
        a_p_e, b_p_e, c_p_e = np.polyfit(cle, cde, 2)
    else:  # Using the quadratic fit values
        # Calculate coefficients for Cd as a function of Cl using the existing α-based fit
        a_p_e, b_p_e, c_p_e = None, None, None  # These won't be used

    a_p_x, b_p_x, c_p_x = np.polyfit(clx, cdx, 2)

    # — Plot —
    fig, axs = plt.subplots(1,3, figsize=(18,5))

    # 1) Cl vs α
    axs[0].plot(αe, cle, 'o-', label='Exp')
    axs[0].plot(αx, clx, 's--', label='XFOIL')
    axs[0].set_title("Cl vs α (formula for 0≤α≤8)")
    axs[0].set_xlabel("α (°)"); axs[0].set_ylabel("Cl")
    axs[0].grid(True)
    txt_e = f"Exp: Cl = {m1_e:.5f}·α + {q1_e:.5f}"
    txt_x = f"XF:  Cl = {m1_x:.5f}·α + {q1_x:.5f}"
    axs[0].text(0.05, 0.95, txt_e, transform=axs[0].transAxes,
                va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    axs[0].text(0.05, 0.85, txt_x, transform=axs[0].transAxes,
                va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    axs[0].legend()

    # 2) Cd vs α
    axs[1].plot(αe, cde, 'o-', label='Exp')
    axs[1].plot(αx, cdx, 's--', label='XFOIL')
    axs[1].set_title("Cd vs α (quadratic fit)")
    axs[1].set_xlabel("α (°)"); axs[1].set_ylabel("Cd")
    axs[1].grid(True)
    eq_e = f"Exp: Cd = {a_cd_e:.2e}·α² + {b_cd_e:.2e}·α + {c_cd_e:.2e}"
    eq_x = f"XF:  Cd = {a_cd_x:.2e}·α² + {b_cd_x:.2e}·α + {c_cd_x:.2e}"
    axs[1].text(0.15, 0.95, eq_e, transform=axs[1].transAxes,
                va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    axs[1].text(0.15, 0.85, eq_x, transform=axs[1].transAxes,
                va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    axs[1].legend()

    # PANEL 3: Cd vs Cl (Cd = f(Cl)), display formula only
    axs[2].plot(cde, cle, 'o-', label='Exp')
    axs[2].plot(cdx, clx, 's--', label='XFOIL')
    axs[2].set_title("Polar: Cd vs Cl (quadratic fit shown only as formula)")
    axs[2].set_xlabel("Cd")
    axs[2].set_ylabel("Cl")
    axs[2].grid(True)

    # Fit Cd = a·Cl² + b·Cl + c (but don't plot it)
    if len(αe) == len(cde):  # Using actual data points
        a_p_e, b_p_e, c_p_e = np.polyfit(cle, cde, 2)
    else:  # Using the quadratic fit - need to convert from α-based to Cl-based
        # Create synthetic points to fit
        cl_range = np.linspace(min(cle), max(cle), 100)
        # Get α values for these Cl values (approximation using linear fit from small angles)
        approx_α = (cl_range - q1_e) / m1_e
        # Get Cd values using the quadratic Cd vs α relationship
        cd_values = a_cd_e * approx_α**2 + b_cd_e * approx_α + c_cd_e
        a_p_e, b_p_e, c_p_e = np.polyfit(cl_range, cd_values, 2)

    a_p_x, b_p_x, c_p_x = np.polyfit(clx, cdx, 2)

    # Show equations only
    axs[2].text(0.45, 0.65,
                f"Exp: Cd = {a_p_e:.2e}·Cl² + {b_p_e:.2e}·Cl + {c_p_e:.2e}",
                transform=axs[2].transAxes,
                va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    axs[2].text(0.45, 0.55,
                f"XF:  Cd = {a_p_x:.2e}·Cl² + {b_p_x:.2e}·Cl + {c_p_x:.2e}",
                transform=axs[2].transAxes,
                va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    axs[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(base_out, "exp_vs_xfoil_comparison.png"), dpi=300)
    plt.close()


# === 8) Main workflow ===
if __name__ == "__main__":

    # — Geometry setup —
    xp = np.linspace(0,1,200)
    yu, yl, dy = naca_23012(xp)
    dy_up = np.where(dy>0, dy, 0)
    dy_lo = np.where(dy<0, dy, 0)

    sensors = np.array([
        [1,0.022,0.034],[2,0.031,0.040],[3,0.062,0.053],[4,0.155,0.071],
        [5,0.201,0.075],[6,0.330,0.077],[7,0.530,0.063],[8,0.681,0.047],
        [9,0.010,-0.010],[10,0.028,-0.017],[11,0.058,-0.023],[12,0.147,-0.033],
        [13,0.196,-0.033],[14,0.326,-0.044],[15,0.427,-0.043],[16,0.837,-0.013]
    ])
    normals = compute_normals(sensors, xp, dy_up, dy_lo)

    # — 1) Original mean‐based results —
    avg_data, max_data, min_data = load_pressure_data_extremes()

    interps, cl_vals = {}, {}
    for α, avg_p in avg_data.items():
        cs, cl = make_interpolators(sensors, avg_p)
        interps[α], cl_vals[α] = cs, cl
        plot_airfoil(xp, yu, yl, sensors, normals, avg_p, α, DIRS["airfoil"])
        plot_cp(cs, α, DIRS["cp"])

    αs = np.array(sorted(cl_vals))
    cls = np.array([cl_vals[a] for a in αs])
    plot_selected_cps(interps, [-6,-4,0,4,8], DIRS["cp"])
    plot_cl_alpha(αs, cls, os.path.join(DIRS["cl_alpha"], "cl_vs_alpha.png"))

    αw, CDw, qc = compute_cd_from_wake()
    save_drag_data(αw, CDw, qc, DIRS["data"])
    plot_cd_alpha(αw, CDw, os.path.join(DIRS["cd_alpha"], "cd_vs_alpha.png"))

    plot_matched_polar_and_fits(αs, cls, αw, CDw, DIRS["polar"], DIRS["interpol"])
    αx, clx, cdx = read_xfoil()
    plot_xfoil(αx, clx, cdx, DIRS["xfoil"])
    # Find common angles between lift and drag data
    common_angles = np.intersect1d(αs, αw)
    common_cls = np.array([cl_vals[a] for a in common_angles])
    common_cds = np.array([CDw[np.where(αw == a)[0][0]] for a in common_angles])

    compare_exp_xfoil(common_angles, common_cls, common_cds, αx, clx, cdx, DIRS["compare"], qc)

    with open(os.path.join(DIRS["data"], "interpolators.pkl"), "wb") as f:
        pickle.dump({"interps": interps, "cl_vals": cl_vals}, f)

    print("✅ Mean‐based results saved under Results/…")

    # Store data for min/max comparison
    mean_data = {
        'αs': αs,
        'cls': cls,
        'αw': αw,
        'CDw': CDw,
        'qc': qc
    }

    # Dictionary to store min/max results
    extreme_data = {}

    # — 2) Variability runs (max & min) —
    for extreme, pdata in (('max', max_data), ('min', min_data)):
        base_root = DIR_VAR_MAX if extreme=='max' else DIR_VAR_MIN

        # Cp→Cl
        interps_v, cl_v = {}, {}
        for α, pvals in pdata.items():
            cs_v, clv = make_interpolators(sensors, pvals)
            interps_v[α], cl_v[α] = cs_v, clv

            plot_airfoil(xp, yu, yl, sensors, normals, pvals, α,
                         os.path.join(base_root, os.path.basename(DIRS["airfoil"])))
            plot_cp(cs_v, α,
                    os.path.join(base_root, os.path.basename(DIRS["cp"])))

        αsv = np.array(sorted(cl_v)); clsv = np.array([cl_v[a] for a in αsv])
        plot_cl_alpha(αsv, clsv,
                      os.path.join(base_root, os.path.basename(DIRS["cl_alpha"]), f"cl_vs_alpha_{extreme}.png"),
                      title=f"Cl vs α ({extreme})")

        # Generate CD values based on extreme pressure values
        # Calculate the ratio of extreme to average pressure for each angle
        pressure_ratios = {}
        for α in αw:
            if α in avg_data and α in pdata:
                avg_p = np.mean(avg_data[α])
                extreme_p = np.mean(pdata[α])
                pressure_ratios[α] = extreme_p / avg_p

        # Scale the CD values based on the pressure ratios
        CDw_v = np.copy(CDw)
        for i, α in enumerate(αw):
            if α in pressure_ratios:
                CDw_v[i] *= pressure_ratios[α]

        # Refit the quadratic for the scaled CD values
        qc_v = np.polyfit(αw, CDw_v, 2)

        # Store results for later comparison
        extreme_data[extreme] = {
            'αs': αsv,
            'cls': clsv,
            'αw': αw,
            'CDw': CDw_v,
            'qc': qc_v
        }

        # Drag
        save_drag_data(αw, CDw_v, qc_v, os.path.join(base_root, os.path.basename(DIRS["data"])))
        plot_cd_alpha(αw, CDw_v,
                      os.path.join(base_root, os.path.basename(DIRS["cd_alpha"]), f"cd_vs_alpha_{extreme}.png"),
                      title=f"Cd vs α ({extreme})")

        # Polar & interpolation & fits
        plot_matched_polar_and_fits(αsv, clsv, αw, CDw_v,
                                    os.path.join(base_root, os.path.basename(DIRS["polar"])),
                                    os.path.join(base_root, os.path.basename(DIRS["interpol"])))

        # XFOIL and comparison
        plot_xfoil(αx, clx, cdx,
                   os.path.join(base_root, os.path.basename(DIRS["xfoil"])))
        # Find common angles between variability lift data and extreme drag data
        common_angles = np.intersect1d(αsv, αw)
        common_cls = np.array([cl_v[a] for a in common_angles])
        common_cds = np.array([CDw_v[np.where(αw == a)[0][0]] for a in common_angles])
        compare_exp_xfoil(common_angles, common_cls, common_cds,
                          αx, clx, cdx,
                          os.path.join(base_root, os.path.basename(DIRS["compare"])),
                          qc_v)

        # save interpolators
        with open(os.path.join(base_root, os.path.basename(DIRS["data"]), f"interpolators_{extreme}.pkl"), "wb") as f:
            pickle.dump({"interps": interps_v, "cl_vals": cl_v}, f)

        print(f"✅ {extreme.title()}‐based results saved under {base_root}/…")
        # Create comparison plot of mean, min, and max values
        # Check if both 'min' and 'max' exist in extreme_data
        if 'min' in extreme_data and 'max' in extreme_data:
            # Find common angles across all three datasets
            common_angles = np.intersect1d(np.intersect1d(mean_data['αs'], extreme_data['min']['αs']),
                                        extreme_data['max']['αs'])

            # Create arrays for plotting
            mean_cls = np.array([cl_vals[a] for a in common_angles])
            min_cls = np.array([extreme_data['min']['cls'][np.where(extreme_data['min']['αs'] == a)[0][0]]
                                for a in common_angles])
            max_cls = np.array([extreme_data['max']['cls'][np.where(extreme_data['max']['αs'] == a)[0][0]]
                                for a in common_angles])

            # Find common angles between lift and drag data for all three cases
            common_with_drag = np.intersect1d(common_angles, αw)
            mean_cds = np.array([mean_data['CDw'][np.where(mean_data['αw'] == a)[0][0]] for a in common_with_drag])
            min_cds = np.array([extreme_data['min']['CDw'][np.where(extreme_data['min']['αw'] == a)[0][0]] for a in common_with_drag])
            max_cds = np.array([extreme_data['max']['CDw'][np.where(extreme_data['max']['αw'] == a)[0][0]] for a in common_with_drag])

            # Mean, min, max comparison plot
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))

            # 1) Cl vs α
            # Calculate percentage differences
            min_pct_cl = (min_cls - mean_cls) / mean_cls * 100
            max_pct_cl = (max_cls - mean_cls) / mean_cls * 100

            axs[0].plot(common_angles, mean_cls, 'o-', color='darkblue', linewidth=3, markersize=8, label='Mean (+0.00000%)')
            axs[0].plot(common_angles, min_cls, 's--', color='forestgreen', linewidth=2.5, markersize=7,
                        label=f'Min ({min_pct_cl.mean():.5f}%)')
            axs[0].plot(common_angles, max_cls, '^-', color='crimson', linewidth=2.5, markersize=7,
                        label=f'Max ({max_pct_cl.mean():.5f}%)')
            axs[0].set_title("Cl vs α Variability", fontsize=14, fontweight='bold')
            axs[0].set_xlabel("α (°)", fontsize=12)
            axs[0].set_ylabel("Cl", fontsize=12)
            axs[0].grid(True, linestyle='--', alpha=0.7)
            axs[0].legend(fontsize=10)

            # 2) Cd vs α
            # Calculate percentage differences
            min_pct_cd = (min_cds - mean_cds) / mean_cds * 100
            max_pct_cd = (max_cds - mean_cds) / mean_cds * 100

            axs[1].plot(common_with_drag, mean_cds, 'o-', color='darkblue', linewidth=3, markersize=8, label='Mean (+0.00000%)')
            axs[1].plot(common_with_drag, min_cds, 's--', color='forestgreen', linewidth=2.5, markersize=7,
                        label=f'Min ({min_pct_cd.mean():.5f}%)')
            axs[1].plot(common_with_drag, max_cds, '^-', color='crimson', linewidth=2.5, markersize=7,
                        label=f'Max ({max_pct_cd.mean():.5f}%)')
            axs[1].set_title("Cd vs α Variability", fontsize=14, fontweight='bold')
            axs[1].set_xlabel("α (°)", fontsize=12)
            axs[1].set_ylabel("Cd", fontsize=12)
            axs[1].grid(True, linestyle='--', alpha=0.7)

            # Display quadratic fit equations
            a_mean, b_mean, c_mean = mean_data['qc']
            eq_mean = f"Mean: Cd = {a_mean:.2e}·α² + {b_mean:.2e}·α + {c_mean:.2e}"
            axs[1].text(0.05, 0.95, eq_mean, transform=axs[1].transAxes,
                        va='top', ha='left', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
            axs[1].legend(fontsize=10)

            # 3) Polar: Cl vs Cd
            common_cls_for_polar = np.array([cl_vals[a] for a in common_with_drag])
            min_cls_for_polar = np.array([extreme_data['min']['cls'][np.where(extreme_data['min']['αs'] == a)[0][0]]
                                        for a in common_with_drag])
            max_cls_for_polar = np.array([extreme_data['max']['cls'][np.where(extreme_data['max']['αs'] == a)[0][0]]
                                        for a in common_with_drag])

            # Calculate percentage differences for polar plot
            min_pct_cl_polar = (min_cls_for_polar - common_cls_for_polar) / common_cls_for_polar * 100
            max_pct_cl_polar = (max_cls_for_polar - common_cls_for_polar) / common_cls_for_polar * 100

            # Calculate combined Cl+Cd values for polar plot
            mean_combined = common_cls_for_polar + mean_cds
            min_combined = min_cls_for_polar + min_cds
            max_combined = max_cls_for_polar + max_cds

            # Calculate percentage differences for combined values
            min_pct_combined = (min_combined - mean_combined) / mean_combined * 100
            max_pct_combined = (max_combined - mean_combined) / mean_combined * 100

            axs[2].plot(mean_cds, common_cls_for_polar, 'o-', color='darkblue', linewidth=3, markersize=8,
                        label='Mean (+0.00000%)')
            axs[2].plot(min_cds, min_cls_for_polar, 's--', color='forestgreen', linewidth=2.5, markersize=7,
                        label=f'Min ({min_pct_combined.mean():.5f}%)')
            axs[2].plot(max_cds, max_cls_for_polar, '^-', color='crimson', linewidth=2.5, markersize=7,
                        label=f'Max ({max_pct_combined.mean():.5f}%)')
            axs[2].set_title("Polar: Cl vs Cd Variability", fontsize=14, fontweight='bold')
            axs[2].set_xlabel("Cd", fontsize=12)
            axs[2].set_ylabel("Cl", fontsize=12)
            axs[2].grid(True, linestyle='--', alpha=0.7)
            axs[2].legend(fontsize=10)

            plt.tight_layout()
            plt.savefig(os.path.join(DIRS["compare"], "mean_min_max_comparison.png"), dpi=300)
            plt.close()

            # Save percentage difference data
            with open(os.path.join(DIRS["data"], "variability_analysis.txt"), 'w') as f:
                f.write("# Variability Analysis Results\n")
                f.write("# This file contains the percentage differences between min/max and mean values\n\n")

                f.write("# 1. Cl vs α Variability\n")
                f.write("# These values represent how much the lift coefficient varies from the mean at each angle of attack\n")
                f.write(f"Mean min percentage difference: {min_pct_cl.mean():.5f}%\n")
                f.write(f"Mean max percentage difference: {max_pct_cl.mean():.5f}%\n")
                f.write(f"Maximum min percentage difference: {np.abs(min_pct_cl).max():.5f}%\n")
                f.write(f"Maximum max percentage difference: {np.abs(max_pct_cl).max():.5f}%\n\n")

                f.write("# 2. Cd vs α Variability\n")
                f.write("# These values represent how much the drag coefficient varies from the mean at each angle of attack\n")
                f.write(f"Mean min percentage difference: {min_pct_cd.mean():.5f}%\n")
                f.write(f"Mean max percentage difference: {max_pct_cd.mean():.5f}%\n")
                f.write(f"Maximum min percentage difference: {np.abs(min_pct_cd).max():.5f}%\n")
                f.write(f"Maximum max percentage difference: {np.abs(max_pct_cd).max():.5f}%\n\n")

                f.write("# 3. Polar (Cl+Cd) Variability\n")
                f.write("# These values represent how much the combined aerodynamic coefficients (Cl+Cd) vary from the mean\n")
                f.write("# This is calculated as the percentage difference in the sum of Cl and Cd at each point\n")
                f.write(f"Mean min percentage difference: {min_pct_combined.mean():.5f}%\n")
                f.write(f"Mean max percentage difference: {max_pct_combined.mean():.5f}%\n")
                f.write(f"Maximum min percentage difference: {np.abs(min_pct_combined).max():.5f}%\n")
                f.write(f"Maximum max percentage difference: {np.abs(max_pct_combined).max():.5f}%\n\n")

                f.write("# Detailed data for each angle of attack:\n")
                f.write("alpha,mean_cl,min_cl,max_cl,mean_cd,min_cd,max_cd,mean_combined,min_combined,max_combined\n")
                for i, alpha in enumerate(common_with_drag):
                    f.write(f"{alpha:.2f},{common_cls_for_polar[i]:.6f},{min_cls_for_polar[i]:.6f},{max_cls_for_polar[i]:.6f},")
                    f.write(f"{mean_cds[i]:.6f},{min_cds[i]:.6f},{max_cds[i]:.6f},")
                    f.write(f"{mean_combined[i]:.6f},{min_combined[i]:.6f},{max_combined[i]:.6f}\n")

            # Save CSV format for easier analysis
            np.savetxt(os.path.join(DIRS["data"], "variability_data.csv"),
                    np.column_stack((
                        common_with_drag,
                        common_cls_for_polar, min_cls_for_polar, max_cls_for_polar,
                        mean_cds, min_cds, max_cds,
                        mean_combined, min_combined, max_combined
                    )),
                    delimiter=",",
                    header="alpha,mean_cl,min_cl,max_cl,mean_cd,min_cd,max_cd,mean_combined,min_combined,max_combined",
                    comments="")

            print("✅ Mean-Min-Max comparison plot saved under Results/comparison_plots/")
            print("✅ Variability analysis data saved under Results/data/")
        else:
            print("⚠️ Skipping Mean-Min-Max comparison: both 'min' and 'max' data required.")
