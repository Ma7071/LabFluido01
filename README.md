# LabFluido2.0

This repository contains a Python-based workflow for airfoil pressure-sweep analysis, interpolation, and comparison with XFOIL data. It automates geometry generation, data loading, Cp/Cl/Cd plotting, polar curve fitting, wake-derived drag computation, and experimental vs. XFOIL comparisons.

---

## Features

* **NACA 23012 Geometry**: Computes upper/lower surface coordinates and normals.
* **Pressure Sweep Handling**: Loads pressure measurement files from `CLDATA/`, computes average pressures, and generates Cp distributions.
* **Force Coefficient Computation**:

  * Integrates Cp curves to compute Cl.
  * Computes drag from wake data in `CDdata/`.
* **Plot Generation**:

  * Airfoil geometry with sensor normals.
  * Cp distributions per angle of attack.
  * Cp comparison for selected angles.
  * Cl vs. α and Cd vs. α curves.
  * Matched and interpolated polar curves.
  * Quadratic fit of Cd vs. Cl.
* **XFOIL I/O**: Reads XFOIL raw data and plots Cl vs. α, Cd vs. α, and polar curve.
* **Experimental vs. XFOIL**: Generates comparative plots of Cl, Cd, and polar curves.

---

## Repository Structure

```
LabFluido2.0/
├── CLDATA/                     # Input pressure data (*.txt)
├── CDdata/                     # Wake-derived drag data (*.txt)
├── rawdata_03.txt              # XFOIL output file
├── Final2.0Var.py              # Main analysis script
├── Results/                    # Generated outputs
│   ├── airfoil_plots/          # Airfoil + normals images
│   ├── cp_plots/               # Cp distribution images
│   ├── cl_alpha_plots/         # Cl vs α plot
│   ├── cd_alpha_plots/         # Cd vs α plot
│   ├── polar_plots/            # Polar curve images
│   ├── interpolated_plots/     # Interpolated polar/CD/CL curves
│   ├── xfoil_plots/            # XFOIL comparison plots
│   ├── comparison_plots/       # Experimental vs XFOIL comparisons
│   └── data/                   # CSV and pickle data exports
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## Requirements

* Python 3.8+
* `numpy`
* `scipy`
* `matplotlib`

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage

1. **Place your pressure data** in `CLDATA/`, named as `<alpha>_gradi.txt` (e.g., `0_gradi.txt`, `4_gradi.txt`). First column: sensor pressures.

2. **Place wake data** in `CDdata/`, named as `<alpha>.txt`, with tab-delimited columns: `z_mm`, `unused`, `qloc`.

3. **Ensure XFOIL output** is in `rawdata_03.txt`.

4. **Run the analysis**:

   ```bash
   python3 Final2.0Var.py
   ```

5. **Results** will be in the `Results/` directory, organized by plot type and data exports.

---

## Customization

* **Sensor Layout**: Modify the `sensors` array in the main script to change sensor positions.
* **Airfoil Definition**: Adjust `naca_23012()` for different NACA profiles.
* **Angle Selection**: Change the list `[-6, -4, 0, 4, 8]` in the `plot_selected_cps` call to your desired angles.

---

## Contact

For questions or issues, please contact *Giulio Mastromartino* at [giuliomastromartino@polimi.it](mailto:giuliomastromartino@polimi.it).