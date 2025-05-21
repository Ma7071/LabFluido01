import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.interpolate import CubicSpline
from scipy import integrate  # Import the integration routines from SciPy
import pickle

# ---- Freestream and fluid parameters (adjust as needed) ----
FREESTREAM_PRESSURE = 190   # Pa, freestream pressure
FREESTREAM_VELOCITY = 20.0  # m/s, freestream velocity (example value)
AIR_DENSITY = 1.174         # kg/m^3, standard air density
CHORD = 0.1                 # m, chord length (normalized)

# ---- Define the NACA 23012 airfoil equations ----
def naca_23012(x):
    """Given an array of x/c values, compute the upper and lower surface coordinates."""
    t = 0.12  # Thickness (12% of chord)
    m = 0.02  # Maximum camber (2% of chord)
    p = 0.3   # Position of max camber (30% of chord)

    # Thickness distribution
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 +
                  0.2843 * x**3 - 0.1015 * x**4)

    # Camber line and slope
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    for i, xi in enumerate(x):
        if xi < p:
            yc[i] = m * (xi / p**2) * (2*p - xi)
            dyc_dx[i] = (2 * m / p**2) * (p - xi)
        else:
            yc[i] = m * ((1 - xi) / (1 - p)**2) * (1 + xi - 2*p)
            dyc_dx[i] = (2 * m / (1 - p)**2) * (p - xi)

    theta = np.arctan(dyc_dx)
    y_upper = yc + yt * np.cos(theta)
    y_lower = yc - yt * np.cos(theta)

    return y_upper, y_lower

# ---- Generate airfoil profile ----
num_points = 200
x_profile = np.linspace(0, 1, num_points)
y_upper, y_lower = naca_23012(x_profile)

# Compute surface slopes (for sensor normal calculations)
dy_upper_dx = np.gradient(y_upper, x_profile)
dy_lower_dx = np.gradient(y_lower, x_profile)

# ---- Sensor positions (sensor number, x/c, z/c) ----
sensors = np.array([
    [1, 0.022, 0.034], [2, 0.031, 0.04], [3, 0.062, 0.053], [4, 0.155, 0.071],
    [5, 0.201, 0.075], [6, 0.33, 0.077], [7, 0.53, 0.063], [8, 0.681, 0.047],
    [9, 0.01, -0.01], [10, 0.028, -0.017], [11, 0.058, -0.023], [12, 0.147, -0.033],
    [13, 0.196, -0.033], [14, 0.326, -0.044], [15, 0.427, -0.043], [16, 0.837, -0.013]
])

# ---- Compute normal angles for sensors (for plotting) ----
sensor_angles = []
for sensor in sensors:
    sensor_num, x_sensor, z_sensor = sensor
    slope = np.interp(x_sensor, x_profile, dy_upper_dx if z_sensor >= 0 else dy_lower_dx)
    normal_vec = np.array([-slope, 1]) / np.linalg.norm([-slope, 1])
    angle_deg = np.degrees(np.arctan2(normal_vec[1], normal_vec[0]))
    sensor_angles.append((x_sensor, angle_deg))

print("Sensor Normal Angles:")
for x_sensor, angle in sensor_angles:
    print(f"x/c = {x_sensor:.3f}: Normal Angle = {angle:.2f} degrees")

# ---- Read pressure data from files ----
pressure_files = sorted(glob.glob("*_gradi.txt"))  # Find all matching files
pressure_data_all = {}

for file in pressure_files:
    try:
        angle_str = file.split("_")[0]
        angle = float(angle_str)
        pressure_data = np.loadtxt(file)[:, :16]  # Read first 16 columns
        avg_pressures = np.mean(pressure_data, axis=0)  # Average per sensor
        pressure_data_all[angle] = avg_pressures
    except Exception as e:
        print(f"Error reading {file}: {e}")

# ---- Create output directory ---
output_directory = "airfoil_plots"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Dictionaries to store interpolators and Cl values for each angle
pressure_interpolators = {}
cl_values = {}

# ---- Process each angle and generate plots ----
for angle in sorted(pressure_data_all.keys()):
    avg_pressures = pressure_data_all[angle]

    pressures_upper = avg_pressures[:8]
    pressures_lower = avg_pressures[8:]

    x_upper = sensors[:8, 1]
    x_lower = sensors[8:16, 1]

    q_inf = 0.5 * AIR_DENSITY * FREESTREAM_VELOCITY**2
    cp_up_sensor = pressures_upper / q_inf
    cp_low_sensor = pressures_lower / q_inf

    cp_convergence = (cp_up_sensor[-1] + cp_low_sensor[-1]) / 2.0

    new_x_upper = np.append(x_upper, 1.15)
    new_cp_upper = np.append(cp_up_sensor, cp_convergence)
    new_x_lower = np.append(x_lower, 1.15)
    new_cp_lower = np.append(cp_low_sensor, cp_convergence)

    cs_upper = CubicSpline(new_x_upper, new_cp_upper, extrapolate=True)
    cs_lower = CubicSpline(new_x_lower, new_cp_lower, extrapolate=True)

    pressure_interpolators[angle] = {"upper": cs_upper, "lower": cs_lower}

    x_integration = np.linspace(0, 1, 1000)
    cp_upper_eval = cs_upper(x_integration)
    cp_lower_eval = cs_lower(x_integration)

    cl = integrate.simpson(cp_lower_eval - cp_upper_eval, x_integration)
    cl_values[angle] = cl

    x_fine_upper = np.linspace(x_upper.min(), x_upper.max(), 1000)
    x_fine_lower = np.linspace(x_lower.min(), x_lower.max(), 1000)
    cs_upper_orig = CubicSpline(x_upper, pressures_upper)
    cs_lower_orig = CubicSpline(x_lower, pressures_lower)
    interp_upper = cs_upper_orig(x_fine_upper)
    interp_lower = cs_lower_orig(x_fine_lower)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(x_profile, y_upper, 'b-', label='Upper Surface')
    ax1.plot(x_profile, y_lower, 'r-', label='Lower Surface')
    for idx, (sensor, (x_sensor, normal_angle)) in enumerate(zip(sensors, sensor_angles)):
        x_sensor_val, z_sensor = sensor[1], sensor[2]
        ax1.plot(x_sensor_val, z_sensor, 'ko')
        pressure = avg_pressures[idx]
        slope = np.interp(x_sensor_val, x_profile, dy_upper_dx if z_sensor >= 0 else dy_lower_dx)
        normal_vec = np.array([-slope, 1]) / np.linalg.norm([-slope, 1])
        scale = 0.05 * (pressure / np.mean(avg_pressures))
        head_width = 0.1 * scale
        head_length = 0.15 * scale
        points_toward_profile = (z_sensor > 0 and normal_vec[1] < 0) or (z_sensor < 0 and normal_vec[1] > 0)
        if points_toward_profile:
            ax1.arrow(x_sensor_val, z_sensor, normal_vec[0] * scale, normal_vec[1] * scale,
                      head_width=head_width, head_length=head_length, fc='black', ec='g')
        else:
            ax1.arrow(x_sensor_val, z_sensor, normal_vec[0] * scale, normal_vec[1] * scale,
                      head_width=head_width, head_length=head_length, fc='g', ec='g')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.legend()
    ax1.set_title(f"Airfoil with Pressure-Proportional Normals (Angle: {angle:.1f}°)")
    ax1.set_xlabel("x/c")
    ax1.set_ylabel("z/c")
    ax1.axis("equal")

    ax2.plot(x_upper, pressures_upper, 'bo', label='Upper Data')
    ax2.plot(x_lower, pressures_lower, 'ro', label='Lower Data')
    ax2.plot(x_fine_upper, interp_upper, 'b-', label='Upper Interpolation')
    ax2.plot(x_fine_lower, interp_lower, 'r-', label='Lower Interpolation')
    ax2.set_xlabel("Sensor x/c")
    ax2.set_ylabel("Pressure (Pa)")
    ax2.set_title("Pressure Interpolation on Two Sides")
    ax2.legend()
    ax2.grid(True)
    ax2.text(0.05, 0.95, f"Cl = {cl:.4f}", transform=ax2.transAxes, fontsize=14,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    filename = f"airfoil_plot_{angle:.1f}deg.png"
    plot_path = os.path.join(output_directory, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path} | Cl = {cl:.4f}")
    plt.close()

    cp_directory = "cp_plots"
    if not os.path.exists(cp_directory):
        os.makedirs(cp_directory)

    plt.figure(figsize=(8, 6))
    plt.plot(x_integration, cs_upper(x_integration), 'b-', label='Cp Upper')
    plt.plot(x_integration, cs_lower(x_integration), 'r-', label='Cp Lower')
    plt.xlabel("x/c")
    plt.ylabel("Coefficient of Pressure (Cp)")
    plt.title(f"Pressure Coefficient Distribution (Angle: {angle:.1f}°)")
    plt.gca().invert_yaxis()
    plt.legend()
    cp_plot_path = os.path.join(cp_directory, f'cp_distribution_{angle:.1f}deg.png')
    plt.savefig(cp_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved Cp distribution plot: {cp_plot_path}")
    plt.close()

# ---- Create a Cl vs. Angle (α) plot ----
angles = np.array(sorted(cl_values.keys()))
cl_arr = np.array([cl_values[a] for a in angles])
cs_cl = CubicSpline(angles, cl_arr)
angles_fine = np.linspace(angles.min(), angles.max(), 1000)
cl_interpolated = cs_cl(angles_fine)

plt.figure(figsize=(8, 6))
plt.plot(angles, cl_arr, 'o', label='Original Data')
plt.plot(angles_fine, cl_interpolated, '-', label='Interpolated Curve')
plt.xlabel("Angle of Attack (°)")
plt.ylabel("Coefficient of Lift (Cl)")
plt.title("Cl vs. Angle of Attack (Interpolated)")
plt.grid(True)
plt.legend()

cl_alfa_dir = "cl_alfa_plots"
if not os.path.exists(cl_alfa_dir):
    os.makedirs(cl_alfa_dir)

cl_alfa_interp_plot_path = os.path.join(cl_alfa_dir, 'cl_vs_alpha_interpolated.png')
plt.savefig(cl_alfa_interp_plot_path, dpi=300, bbox_inches='tight')
print(f"Saved Cl vs. Angle Interpolation plot: {cl_alfa_interp_plot_path}")
plt.close()

# ---- Store the interpolation functions and Cl values for later use ----
with open(os.path.join(output_directory, 'pressure_interpolators.pkl'), 'wb') as f:
    pickle.dump({"interpolators": pressure_interpolators, "cl_values": cl_values}, f)

print("\nAll plots have been generated and saved in the 'airfoil_plots' directory.")
print("Pressure interpolation functions and Cl values have been stored in 'pressure_interpolators.pkl'.")

# === PART 2: Compute & Save Profile Drag Coefficient (C_D) ===

# Constants — make sure to define these properly
rho = 1.174                     # kg/m³, air density
V_inf = FREESTREAM_VELOCITY    # m/s, freestream velocity (define or substitute)
d = CHORD                      # m, chord length (define or substitute)

data_dir = 'CDdata'  # folder with your wake-survey .txt files


alpha_list = []
CD_list    = []

for fname in os.listdir(data_dir):
    if not fname.endswith('.txt'):
        continue

    # 1) Extract α from filename
    alpha = float(fname.replace('.txt', ''))
    alpha_list.append(alpha)

    # 2) Load data: z[mm], q_inf_measurements[Pa], q_loc_measurements[Pa]
    data = np.loadtxt(os.path.join(data_dir, fname),
                      delimiter='\t', skiprows=1)
    z_mm        = data[:, 0]
    q_inf_all   = data[:, 1]
    q_loc_all   = data[:, 2]

    # 3) Determine a single freestream dynamic pressure q∞
    q_inf = q_inf_all.mean()   # scalar Pa

    # 4) Average q_loc at each unique z
    unique_z   = np.unique(z_mm)
    avg_q_loc  = np.array([q_loc_all[z_mm == z].mean()
                           for z in unique_z])
    z_m        = unique_z / 1000.0  # to meters

    # 5) Build the wake-deficit integrand: Δq(z) = q∞ – q(z)
    integrand = q_inf - avg_q_loc   # [Pa]

    # 6) Integrate Δq over z to get momentum deficit (Pa·m)
    momentum_deficit = np.trapz(integrand, z_m)

    # 7) Drag per span D' = 2 * ∫(q∞ – q) dz
    D_prime = 2.0 * momentum_deficit

    # 8) 2D drag coefficient: C_D = D' / (q∞·c)
    CD = D_prime / (q_inf * d)

    CD_list.append(CD)

# Now alpha_list and CD_list are parallel lists of floats
# e.g. convert to array if you like:
alpha_array = np.array(alpha_list)
CD_array    = np.array(CD_list)


# Sort by angle of attack
alpha_array = np.array(alpha_list)
CD_array = np.array(CD_list)
sort_idx = np.argsort(alpha_array)
alpha_array = alpha_array[sort_idx]
CD_array = CD_array[sort_idx]

# Fit C_D = C_D0 + k α²
coeffs = np.polyfit(alpha_array**2, CD_array, 1)
k, CD0 = coeffs

# Fit quadratic: C_D = C_D0 + k₁α + k₂α²
quad_coeffs = np.polyfit(alpha_array, CD_array, 2)
k1, k2, CD0_quad = quad_coeffs

print("\n=== Profile Drag Results ===")
print(f"CD0 (zero-lift) = {CD0:.6f}")
print(f"k (parabolic coeff) = {k:.6f}")
print(f"\nQuadratic Fit Coefficients:")
print(f"CD0 = {CD0_quad:.6f}")
print(f"k₁ = {k1:.6f}")
print(f"k₂ = {k2:.6f}")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# First plot: Drag Polar (CD vs α²)
ax1.plot(alpha_array**2, CD_array, 'bo', label='Data points')
ax1.plot(alpha_array**2, CD0 + k*alpha_array**2, 'r--',
         label=f'Parabolic fit: C_D0={CD0:.4f}, k={k:.2e}')
ax1.set_xlabel("α² (deg²)")
ax1.set_ylabel("C_D")
ax1.set_title("Drag Polar")
ax1.legend()
ax1.grid(True)

# Second plot: CD vs angle of attack
ax2.plot(alpha_array, CD_array, 'bo', label='Data points')
ax2.set_xlabel("Angle of Attack (°)")
ax2.set_ylabel("Drag Coefficient (C_D)")
ax2.set_title("Drag Coefficient vs Angle of Attack")
ax2.legend()
ax2.grid(True)

cd_alfa_dir = "cd_alfa_plots"
if not os.path.exists(cd_alfa_dir):
    os.makedirs(cd_alfa_dir)

cd_alfa_interp_plot_path = os.path.join(cd_alfa_dir, 'cd_vs_alpha^2_regression.png')
plt.savefig(cd_alfa_interp_plot_path, dpi=300, bbox_inches='tight')
print(f"Saved CD vs. Angle Regression plot: {cd_alfa_interp_plot_path}")
plt.close()

# Save drag polar data and coefficients
os.makedirs(output_directory, exist_ok=True)

# Raw data CSV
csv_path = os.path.join(output_directory, 'drag_polar.csv')
with open(csv_path, 'w') as f:
    f.write('alpha_deg,alpha2_deg2,CD\n')
    for a, cd in zip(alpha_array, CD_array):
        f.write(f'{a:.3f},{a**2:.6f},{cd:.6f}\n')
print(f"Saved drag polar data → {csv_path}")

# Coefficients TXT
coeff_path = os.path.join(output_directory, 'drag_polar_coeffs.txt')
with open(coeff_path, 'w') as f:
    f.write(f'CD0: {CD0:.6f}\n')
    f.write(f'k:   {k:.6f}\n')
    f.write(f'k₁:   {k1:.6f}\n')
    f.write(f'k₂:   {k2:.6f}\n')
print(f"Saved drag polar coefficients → {coeff_path}")


# === PART 3: Cl vs Cd Plot ===


# Ensure Cl and Cd data are aligned by angle of attack
cl_cd_data = []
for alpha_cd, cd in zip(alpha_list, CD_list):
    if alpha_cd in cl_values:
        cl = cl_values[alpha_cd]
        cl_cd_data.append((cl, cd))

# Sort by Cl for smooth plotting
cl_cd_data = sorted(cl_cd_data, key=lambda x: x[0])
cl_arr_plot = np.array([c[0] for c in cl_cd_data])
cd_arr_plot = np.array([c[1] for c in cl_cd_data])

# Fit spline for smooth polar curve
cs_cl_cd = CubicSpline(cl_arr_plot, cd_arr_plot)
cl_fine = np.linspace(cl_arr_plot.min(), cl_arr_plot.max(), 1000)
cd_fine = cs_cl_cd(cl_fine)

# Plot Cl vs Cd
plt.figure(figsize=(8, 6))
plt.plot(cd_arr_plot, cl_arr_plot, 'o', label='Original Data')
plt.plot(cd_fine, cl_fine, '-', label='Interpolated Curve')
plt.xlabel("Coefficient of Drag (Cd)")
plt.ylabel("Coefficient of Lift (Cl)")
plt.title("Polar Curve: Cl vs. Cd")
plt.grid(True)
plt.legend()

polar_dir = "polar_plot"
if not os.path.exists(polar_dir):
    os.makedirs(polar_dir)

polar_path = os.path.join(polar_dir, 'cl_vs_cd_polar.png')
plt.savefig(polar_path, dpi=300, bbox_inches='tight')
print(f"Saved Cl vs. Cd polar plot: {polar_path}")
plt.close()


# === PART 4: Super-Interpolated Cl and Cd vs Alpha + Polar ===



# Sort alpha and Cl/Cd data
alpha_cl = sorted(cl_values.items())
alpha_cd = sorted(zip(alpha_list, CD_list))

alpha_cl_arr = np.array([a for a, _ in alpha_cl])
cl_arr = np.array([cl for _, cl in alpha_cl])

alpha_cd_arr = np.array([a for a, _ in alpha_cd])
cd_arr = np.array([cd for _, cd in alpha_cd])

# Interpolate Cl and Cd vs alpha
cl_interp = CubicSpline(alpha_cl_arr, cl_arr)
cd_interp = CubicSpline(alpha_cd_arr, cd_arr)

# Define common fine alpha range for super interpolation
alpha_fine = np.linspace(
    max(min(alpha_cl_arr.min(), alpha_cd_arr.min()), -8),
    min(max(alpha_cl_arr.max(), alpha_cd_arr.max()), 10),
    2000
)
cl_fine = cl_interp(alpha_fine)
cd_fine = cd_interp(alpha_fine)


# Plot super-interpolated Cl vs Cd polar curve
plt.figure(figsize=(8, 6))
plt.plot(cd_fine, cl_fine, '-', label='Super-Interpolated Polar')
plt.xlabel("Coefficient of Drag (Cd)")
plt.ylabel("Coefficient of Lift (Cl)")
plt.title("Super-Interpolated Polar Curve: Cl vs. Cd")
plt.grid(True)
plt.legend()
# Create a new directory for super-interpolated plots
super_polar_dir = "super_interpolated_plots"
if not os.path.exists(super_polar_dir):
    os.makedirs(super_polar_dir)

super_polar_dir = os.path.join(super_polar_dir, 'cl_vs_cd_polar_super_inter.png')
plt.savefig(super_polar_dir, dpi=300, bbox_inches='tight')
plt.close()

print("Super-interpolated plotsaved ")
