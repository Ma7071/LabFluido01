
#!/usr/bin/env python3
import os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from scipy import integrate

# === 0) Constants ===
FREESTREAM_PRESSURE = 190    # Pa
FREESTREAM_VELOCITY = 20.0   # m/s
AIR_DENSITY        = 1.174   # kg/m³
CHORD              = 0.1     # m

# === 1) Setup Results directories ===
RESULTS      = "Results"
DIR_AIRFOIL  = os.path.join(RESULTS, "airfoil_plots")
DIR_INTERPOL = os.path.join(RESULTS, "interpolated_plots")
DIR_CP       = os.path.join(RESULTS, "cp_plots")
DIR_CL_ALPHA = os.path.join(RESULTS, "cl_alpha_plots")
DIR_CD_ALPHA = os.path.join(RESULTS, "cd_alpha_plots")
DIR_POLAR    = os.path.join(RESULTS, "polar_plots")
DIR_XFOIL    = os.path.join(RESULTS, "xfoil_plots")
DIR_COMPARE  = os.path.join(RESULTS, "comparison_plots")
DIR_DATA     = os.path.join(RESULTS, "data")

for d in [DIR_AIRFOIL, DIR_INTERPOL, DIR_CP, DIR_CL_ALPHA, DIR_CD_ALPHA,
          DIR_POLAR, DIR_XFOIL, DIR_COMPARE, DIR_DATA]:
    os.makedirs(d, exist_ok=True)

# === 2) Airfoil geometry & normals ===
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

# === 3) Pressure-sweep handling ===
def load_pressure_data(data_dir="CLDATA", pattern="*_gradi.txt"):
    data = {}
    for fname in sorted(glob.glob(os.path.join(data_dir, pattern))):
        try:
            α = float(os.path.basename(fname).split("_")[0])
            arr = np.loadtxt(fname)[:, :16]
            data[α] = np.mean(arr, axis=0)
        except:
            continue
    return data

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

# === 4) Plot helpers ===
def plot_airfoil(xp, yu, yl, sensors, normals, avg_p, α):
    plt.figure(figsize=(8,6))
    plt.plot(xp, yu, 'b-', xp, yl, 'r-')
    for i,(num,xs,zs) in enumerate(sensors):
        slope = np.interp(xs, xp, np.gradient(yu if zs>=0 else yl, xp))
        v = np.array([-slope,1.0]); v /= np.linalg.norm(v)
        scale = 0.05*(avg_p[i]/avg_p.mean())
        plt.arrow(xs, zs, v[0]*scale, v[1]*scale,
                  head_width=0.1*scale, head_length=0.15*scale,
                  fc='g', ec='g')
    plt.gca().set_aspect('equal','box')
    plt.xlabel("x/c"); plt.ylabel("z/c")
    plt.title(f"Airfoil + Normals @ {α:.1f}°")
    plt.savefig(f"{DIR_AIRFOIL}/airfoil_{α:.1f}deg.png", dpi=300)
    plt.close()

def plot_cp(interps, α):
    x = np.linspace(0,1,1000)
    plt.figure(figsize=(6,4))
    plt.plot(x, interps["upper"](x), 'b-', label='Cp up')
    plt.plot(x, interps["lower"](x), 'r-', label='Cp lo')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel("x/c"); plt.ylabel("Cp")
    plt.title(f"Cp @ {α:.1f}°")
    plt.savefig(f"{DIR_CP}/cp_{α:.1f}deg.png", dpi=300)
    plt.close()

def plot_selected_cps(interps, angles):
    # only keep angles we actually loaded
    available = sorted(interps.keys())
    sel = [α for α in angles if α in interps]
    if not sel:
        print(f"No Cp data for any of {angles}, available angles are {available}")
        return

    x = np.linspace(0,1,1000)
    plt.figure(figsize=(8,5))
    for α in sel:
        cs = interps[α]
        plt.plot(x, cs["upper"](x), '--', label=f'Up {α}°')
        plt.plot(x, cs["lower"](x), '-',  label=f'Lo {α}°')
    plt.gca().invert_yaxis()
    plt.legend(ncol=2, fontsize=8)
    plt.xlabel("x/c"); plt.ylabel("Cp")
    plt.title("Cp Comparison")
    plt.grid(True)
    plt.savefig(f"{DIR_CP}/selected_angles.png", dpi=300)
    plt.close()


def plot_cl_alpha(alphas, cls):
    cs = CubicSpline(alphas, cls)
    af = np.linspace(alphas.min(), alphas.max(), 500)
    plt.figure(figsize=(6,4))
    plt.plot(alphas, cls, 'o', af, cs(af), '-')
    plt.xlabel("α (°)"); plt.ylabel("Cl")
    plt.title("Cl vs α")
    plt.grid(True)
    plt.savefig(f"{DIR_CL_ALPHA}/cl_vs_alpha.png", dpi=300)
    plt.close()

def plot_cd_alpha(alphas, cds):
    plt.figure(figsize=(6,4))
    plt.plot(alphas, cds, 'o-')
    plt.xlabel("α (°)"); plt.ylabel("Cd")
    plt.title("Cd vs α (Wake Data)")
    plt.grid(True)
    plt.savefig(f"{DIR_CD_ALPHA}/cd_vs_alpha.png", dpi=300)
    plt.close()

# === 5+6) Polar using only common α values ===
def plot_matched_polar_and_fits(αs, cls, αw, CDw):
    # 1) find exact common angles
    common = np.intersect1d(αs, αw)

    # Check if we have enough data points
    if len(common) < 4:
        print(f"Warning: Only {len(common)} common angles found. Need at least 4 for proper fitting.")
        if len(common) == 0:
            print("Error: No common angles found. Cannot create polar plots.")
            return

    Cl_c = np.array([cls[np.where(αs==a)[0][0]] for a in common])
    Cd_c = np.array([CDw[np.where(αw==a)[0][0]] for a in common])

    # 2) save matched data
    np.savetxt(os.path.join(DIR_DATA, "polar_matched.csv"),
               np.column_stack((common, Cl_c, Cd_c)),
               delimiter=",",
               header="alpha,Cl,Cd",
               comments="")

    # 3) plot matched polar
    plt.figure(figsize=(6,5))
    plt.plot(Cd_c, Cl_c, 'o-', label='Matched Polar')
    plt.xlabel("Cd"); plt.ylabel("Cl")
    plt.title("Matched Polar Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{DIR_POLAR}/cl_vs_cd_matched.png", dpi=300)
    plt.close()


    # === PART 4: Super-Interpolation + Quadratic Fit ===
    # Sort and deduplicate


    # Sort and deduplicate Cl(α)
    α_cl = sorted(cl_vals.items())
    αc_arr, Cl_arr = map(np.array, zip(*α_cl))
    αc_arr, idx_c = np.unique(αc_arr, return_index=True)
    Cl_arr = Cl_arr[idx_c]

    # Sort and deduplicate Cd(α)
    α_cd = sorted(zip(αw, CDw))
    αd_arr, Cd_arr = map(np.array, zip(*α_cd))
    αd_arr, idx_d = np.unique(αd_arr, return_index=True)
    Cd_arr = Cd_arr[idx_d]

    # Now define the fine α array within the common range
    α_min = max(αc_arr.min(), αd_arr.min())
    α_max = min(αc_arr.max(), αd_arr.max())
    α_fine = np.linspace(α_min, α_max, 1000)

    # Fit splines Cl(α) and Cd(α)
    spline_Cl = UnivariateSpline(αc_arr, Cl_arr, s=1e-8)
    spline_Cd = UnivariateSpline(αd_arr, Cd_arr, s=1e-8)

    # Interpolate Cl and Cd
    Cl_fine = spline_Cl(α_fine)
    Cd_fine = spline_Cd(α_fine)



    # save super-interpolated polar
    #



    plt.figure(figsize=(6,5));
    plt.plot(Cd_fine,Cl_fine,'--',label='Interpolated Polar');
    plt.xlabel("Cd"); plt.ylabel("Cl")
    plt.title("Interpolated Polar Curve")
    plt.grid(True)
    plt.savefig(f"{DIR_INTERPOL}/interpolated_polar.png", dpi=300);
    plt.close()

    plt.figure(figsize=(6,5));
    plt.plot(α_fine,Cd_fine,'--',label='Interpolated CD');
    plt.xlabel("α"); plt.ylabel("Cd")
    plt.title("Interpolated CD Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{DIR_INTERPOL}/interpolated_CD.png", dpi=300);
    plt.close()

    plt.figure(figsize=(6,5))
    plt.plot(α_fine,Cl_fine,'--',label='Interpolated CL')
    plt.xlabel("α"); plt.ylabel("Cl")
    plt.title("Interpolated CL Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{DIR_INTERPOL}/interpolated_CL.png", dpi=300)
    plt.close()


    # quadratic fit on common α
    common=sorted(set(αc_arr).intersection(αd_arr))
    Cl_c=np.array([dict(α_cl)[a] for a in common]); Cd_c=np.array([dict(α_cd)[a] for a in common])
    a,b,c=np.polyfit(Cl_c,Cd_c,2)
    Cl_fit=np.linspace(Cl_c.min(),Cl_c.max(),500); Cd_fit=a*Cl_fit**2+b*Cl_fit+c

    plt.figure(figsize=(6,5))
    plt.plot(Cd_c,Cl_c,'ro',label='Data')
    plt.plot(Cd_fit,Cl_fit,'b--',label=f'Cd={a:.2e}Cl²+{b:.2e}Cl+{c:.2e}')
    plt.xlabel("Cd"); plt.ylabel("Cl")
    plt.title("Quadratic Fit of CD vs CL")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{DIR_POLAR}/cl_vs_cd_quad_fit.png", dpi=300)
    plt.close()

# === 7) Wake-derived drag ===
def compute_cd_from_wake(data_dir):
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
    return alphas[idx], CDs[idx], np.polyfit(alphas[idx], CDs[idx], 2)

def save_drag_data(alphas, CDs, coeffs):
    with open(f"{DIR_DATA}/drag_polar.csv",'w') as f:
        f.write("alpha_deg,alpha2,CD\n")
        for α,cd in zip(alphas,CDs):
            f.write(f"{α:.3f},{α**2:.6f},{cd:.6f}\n")
    with open(f"{DIR_DATA}/drag_polar_coeffs.txt",'w') as f:
        f.write(f"Cd(α)={coeffs[0]:.6e}·α²+{coeffs[1]:.6e}·α+{coeffs[2]:.6e}\n")

# === 8) XFOIL I/O & plots ===
def read_xfoil(fname):
    a_list, cl_list, cd_list = [], [], []
    for L in open(fname):
        ls = L.strip().lower()
        if not ls or ls.startswith('alpha'):
            continue
        a,c1,c2 = map(float, L.split()[:3])
        a_list.append(a); cl_list.append(c1); cd_list.append(c2)
    return np.array(a_list), np.array(cl_list), np.array(cd_list)

def plot_xfoil(α, cl, cd):
    plt.figure(figsize=(6,4))
    plt.plot(α,cl,'o-')
    plt.xlabel("α"); plt.ylabel("Cl"); plt.title("XFOIL Cl vs α")
    plt.grid(True); plt.savefig(f"{DIR_XFOIL}/xfoil_cl_alpha.png", dpi=300); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(α,cd,'o-')
    plt.xlabel("α"); plt.ylabel("Cd"); plt.title("XFOIL Cd vs α")
    plt.grid(True); plt.savefig(f"{DIR_XFOIL}/xfoil_cd_alpha.png", dpi=300); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(cd,cl,'o-')
    plt.xlabel("Cd"); plt.ylabel("Cl"); plt.title("XFOIL Polar")
    plt.grid(True); plt.savefig(f"{DIR_XFOIL}/xfoil_polar.png", dpi=300); plt.close()

# === 9) Comparison ===
def compare_exp_xfoil(αe, cle, cde, αx, clx, cdx):
    me = (αe>=αx.min()) & (αe<=αx.max())
    mx = (αx>=αe.min()) & (αx<=αe.max())
    αe, cle, cde = αe[me], cle[me], cde[me]
    αx, clx, cdx = αx[mx], clx[mx], cdx[mx]
    fig, axs = plt.subplots(1,3, figsize=(15,4))
    axs[0].plot(αe, cle,'-', αx, clx,'--'); axs[0].set_title("Cl vs α"); axs[0].grid(True)
    axs[1].plot(αe, cde,'-', αx, cdx,'--'); axs[1].set_title("Cd vs α"); axs[1].grid(True)
    axs[2].plot(cde, cle,'-', cdx, clx,'--'); axs[2].set_title("Polar"); axs[2].grid(True)
    for ax in axs: ax.legend(['Exp','XFOIL'])
    plt.tight_layout()
    plt.savefig(f"{DIR_COMPARE}/exp_vs_xfoil.png", dpi=300)
    plt.close()

# === Main workflow ===
if __name__ == "__main__":
    # 1) Geometry
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

    # 2) Pressure sweep → Cp & Cl
    p_data = load_pressure_data()
    interps, cl_vals = {}, {}
    for α, avg_p in p_data.items():
        cs, cl = make_interpolators(sensors, avg_p)
        interps[α], cl_vals[α] = cs, cl
        plot_airfoil(xp, yu, yl, sensors, normals, avg_p, α)
        plot_cp(cs, α)

    # 3) Cp comparison & Cl vs α
    plot_selected_cps(interps, [-6,-4,0,4,8])
    αs = np.array(sorted(cl_vals.keys()))
    cls = np.array([cl_vals[a] for a in αs])
    plot_cl_alpha(αs, cls)

    # 4) Wake drag & CD vs α
    αw, CDw, qc = compute_cd_from_wake("CDdata")
    save_drag_data(αw, CDw, qc)
    plot_cd_alpha(αw, CDw)

    # 5+6) Polar and fits using only common α
    plot_matched_polar_and_fits(αs, cls, αw, CDw)

    # 7) XFOIL
    αx, clx, cdx = read_xfoil("rawdata_03.txt")
    plot_xfoil(αx, clx, cdx)

    # 8) Comparison Exp vs XFOIL
    compare_exp_xfoil(αs, cls, np.polyval(qc, αs),
                      αx, clx, cdx)

    # 9) Save interpolators & Cl
    with open(os.path.join(DIR_DATA, "interpolators.pkl"), "wb") as f:
        pickle.dump({"interps": interps, "cl_vals": cl_vals}, f)

    print("✅ All results saved in the `Results/` directory.")
