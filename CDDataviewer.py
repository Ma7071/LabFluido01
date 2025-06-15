import os
import numpy as np
import matplotlib.pyplot as plt

# Constants for drag coefficient computation
FREESTREAM_VELOCITY = 20.0  # m/s
AIR_DENSITY = 1.174         # kg/m³
CHORD = 0.1                 # m

# Directories
cd_data_dir = 'CDdata'
raw_plots_dir = 'raw_cd_data_plots'
os.makedirs(raw_plots_dir, exist_ok=True)

# Process each wake-survey file
for fname in sorted(os.listdir(cd_data_dir)):
    if not fname.endswith('.txt'):
        continue

    # Extract angle of attack
    alpha = float(fname.replace('.txt', ''))
    filepath = os.path.join(cd_data_dir, fname)

    # Load raw data: z [mm], q_inf [Pa], q_loc [Pa]
    data = np.loadtxt(filepath, delimiter='\t', skiprows=1)
    z_mm = data[:, 0]
    q_inf_all = data[:, 1]
    q_loc_all = data[:, 2]

    # Average values at each unique z
    unique_z = np.unique(z_mm)
    avg_q_inf = np.array([np.mean(q_inf_all[z_mm == z]) for z in unique_z])
    avg_q_loc = np.array([np.mean(q_loc_all[z_mm == z]) for z in unique_z])
    z_m = unique_z / 1000.0

    # Plot raw dynamic pressures vs z
    plt.figure(figsize=(8, 6))
    plt.plot(z_m, avg_q_inf, 'b-o', label='Upstream (q_inf)')
    plt.plot(z_m, avg_q_loc, 'r-s', label='Downstream (q_loc)')
    plt.xlabel('Distance from wall, z (m)')
    plt.ylabel('Dynamic Pressure (Pa)')
    plt.title(f'Raw Wake Pressures at α = {alpha:.1f}°')
    plt.legend()
    plt.grid(True)

    # Save figure
    plot_filename = f'raw_pressures_{alpha:.1f}deg.png'
    plt.savefig(os.path.join(raw_plots_dir, plot_filename), dpi=300, bbox_inches='tight')
    plt.close()

print(f"All raw pressure plots saved in '{raw_plots_dir}' directory.")
