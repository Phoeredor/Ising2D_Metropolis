# 2D Ising Model: MCMC Simulation & Critical Behavior Analysis

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Linux](https://img.shields.io/badge/Linux-FCC624.svg?style=flat&logo=linux&logoColor=black)
![C](https://img.shields.io/badge/C-00599C.svg?style=flat&logo=c&logoColor=white)
![Python](https://img.shields.io/badge/Python-%E2%89%A5%203.8-3776AB.svg?style=flat&logo=python&logoColor=white)
![GCC](https://img.shields.io/badge/GCC-00599C.svg?style=flat&logo=gnu&logoColor=white)

> **Module 1** — *Numerical Methods for Physics (Metodi Numerici per la Fisica)*, University of Pisa

A high-performance C implementation of the 2D Ising model on a square lattice using MCMC with the Metropolis algorithm. This project implements an automated, rigorous pipeline designed to study phase transitions and Finite-Size Scaling (FSS).

---

## ✨ Key Features

- **Single-Shot FSS Pipeline:** Automated generation of simulation points based on the universal scaling variable $x = (\beta - \beta_c)L^{1/\nu}$, ensuring optimal sampling density near criticality.
- **Smart Dynamics Analysis:** Automatic estimation of integrated autocorrelation time ($\tau_{int}$) via windowing method to dimension simulation length dynamically.
- **Ergodicity Check:** Simultaneous random/up/down spin initialization runs to detect hysteresis and ensure proper thermalization.
- **Server-Ready Execution:** Scripts include built-in job control (max concurrent jobs) and process prioritization (`nice`) for shared environments.
- **Robust Data Handling:** Binary I/O with header validation, drift detection, and "resume capability" (skips already completed runs).
- **High-Quality RNG:** Uses the PCG32 generator for superior statistical properties.

---

## 📁 Directory Structure

```text
.
├── src/                      # Core C source files
│   ├── main_pilot.c          # Entry point for Pilot/Diagnostic runs
│   ├── main_prod.c           # Entry point for Production runs
│   ├── ising.c               # Metropolis algorithm & observables
│   ├── lattice.c             # Geometry & neighbor handling
│   └── pcg32.c               # PCG32 Random Number Generator
├── include/                  # Header files (ising.h, lattice.h, etc.)
├── scripts/                  # Workflow orchestration
│   ├── pilot/
│   │   ├── run_pilot.sh              # Runs FSS pilot grid
│   │   ├── run_pilot_analysis.sh     # Computes tau_int from pilots
│   │   ├── analysis/                 # Analysis scripts
│   │   │   ├── run_equilibration_diagnostics.sh
│   │   │   ├── analyze_pilots.py
│   │   │   └── summarize_pilots.py
│   │   ├── plotting/                 # Visualization
│   │   │   └── plot_equilibration_dynamics.py
│   │   └── setup/                    # Grid generation
│   │       └── gen_pilot_grid.py
│   └── production/
│       ├── run_prod.sh               # Executes production runs
│       ├── analysis/                 # Data analysis
│       │   ├── parse_final.py
│       │   └── check_integrity_prod.py
│       ├── plotting/                 # Plotting scripts
│       │   └── plot_observables.py
│       └── setup/                    # Plan generation
│           └── generate_plan.py
├── tests/                    # Unit tests
└── results/                  # Output storage (Gitignored)
    ├── pilot/MC_convergence/ # Data for equilibration plots
    ├── pilot/L{L}/bin/       # Raw pilot data
    └── production/L{L}/bin/  # Final production binary data
```
---

## 🚀 Getting Started

### 1. Compilation

Build the high-performance binaries using the provided Makefile.

```bash
make clean && make

```

---

### 2. Preliminar Phase (Calibration)

This phase probes the system dynamics to estimate $\tau_{int}$ and plan the production run.

#### Step 1: Validation & Diagnostics test
Before running the grid, we validate the algorithm dynamics. This step generates time-series data for three Ising 2D systems with random initial configuration $(0)$, $m=1$ $(+)$, $m=-1$ $(-)$ (one run per core).
```bash
# Generates time-series data (L >= 128 recommended)
./scripts/pilot/analysis/run_equilibration_diagnostics.sh

# Produces "Convergence to Equilibrium" and "Relaxation" plots
python3 scripts/pilot/plotting/plot_equilibration_dynamics.py
```

#### Step 2: Run Pilot Simulations
Generates a sparse grid based on scaling variable $x$ and runs short simulations (`rng` & `up` starts).

**Local Mode (Full Power):**
Use nearly all CPU cores with normal priority.
```bash
./scripts/pilot/run_pilot.sh local
```
**Server Mode (Shared Resource):**
Restricts to 6 concurrent jobs and sets low process priority (nice -n 19).
```bash
./scripts/pilot/run_pilot.sh server
```

#### Step 3: Analyze Pilots
Calculates autocorrelation times and checks for ergodicity breaking. Aggregates results into a JSON database.
```bash
./scripts/pilot/run_pilot_analysis.sh

```

### 3. Production Phase (High Statistics)
This phase generates a dense grid of points and executes long simulations required for high-precision FSS.

#### Step 1:Generate Production Plan
Creates `production_plan.dat` using interpolated $\tau_{\text{int}}$​ from pilots. It applies a rigorous filter (e.g., $0.35\leq\beta\leq0.50$) and targets $\sim 20\text{k}$ independent measurements in the critical region.
```bash
python3 scripts/production/setup/generate_plan.py
```
#### Step 2: Launch Production & integrity check
Executes the plan. This script is robust: it handles job throttling (max 6 cores), process priority, and resumes interrupted runs automatically.
```bash
./scripts/production/run_prod.sh
```

Ensures binary files are not corrupted before analysis and then generates the final plots.
```bash
# Check for corrupted binary records
python3 scripts/production/analysis/check_integrity_prod.py

# Generate FSS plots (Magnetization, Binder Cumulant, etc.)
# Example for L=64
python3 scripts/production/plotting/plot_observables.py 64
```

#### Step 3: Final Parsing & Error Analysis
Parses binary files, performs blocking and Jackknife analysis to compute observables ($\chi$, $C_v$, $U_4$, $|m|$) and their statistical errors.
```bash
# Example for L=64
python3 scripts/production/analysis/parse_final.py 64
```

## 🎯 Physics Objectives

* **Phase Transition:** Plot Magnetization $M$, Susceptibility $\chi$, and Specific Heat $C_V$ vs $\beta$.
* **Finite-Size Scaling:** Verification of the universal scaling hypothesis: $\chi \sim L^{\gamma/\nu} \tilde{\chi}(t L^{1/\nu})$.
* **Critical Exponents:** High-precision extraction of $\beta$, $\gamma$, $\nu$ via data collapse.
* **Critical Slowing Down:** Measurement of the dynamic exponent $z \sim 2.17$

---

## 📊 Results

### Summary of Critical Exponents

| Quantity | Measured | Exact | $\Delta/\sigma$ | Method |
|:---:|:---:|:---:|:---:|---|
| $\beta_c$ | $0.440751 \pm 0.000060$ | $0.440687^a$ | $1.08\sigma$ | Binder crossing |
| $U^*$ | $0.6116 \pm 0.0006$ | $0.6107^b$ | $1.6\sigma$ | Universal value at crossing |
| $\nu$ | $0.9818 \pm 0.0166$ | $1.0000^a$ | $1.09\sigma$ | Correlated $\chi^2$ fit of $S(L)$ |
| $\gamma/\nu$ | $1.7487 \pm 0.0130$ | $1.7500^a$ | $0.10\sigma$ | 3-param fit of $\chi'_{\max}(L)$ |
| $\beta/\nu$ | $0.12473 \pm 0.00336$ | $0.12500^a$ | $0.08\sigma$ | 3-param fit of $M(\beta_{pc}, L)$ |
| $\eta$ | $0.2513 \pm 0.0130$ | $0.2500^a$ | $0.10\sigma$ | $\eta = 2 - \gamma/\nu$ |
| $\alpha$ | $0.007 \pm 0.023$ | $0^a$ | $0.33\sigma$ | Free $g = \alpha/\nu$ in $C_{\max}(L)$ |
| $\delta$ | $15.02 \pm 0.41$ | $15^a$ | $0.05\sigma$ | $\delta = 1 + (\gamma/\nu)/(\beta/\nu)$ |
| $z$ | $2.18 \pm 0.08$ | $2.1667^b$ | $0.17\sigma$ | Dynamic scaling $\tau_{int} \sim L^z$ |
| **Hyperscaling** | $1.9981 \pm 0.0147$ | $2.0000$ | $0.13\sigma$ | $2(\beta/\nu)+\gamma/\nu$, $d=2$ |

> $^a$ Exact theoretical value. $^b$ Numerical reference estimate.

<details>
<summary><b>⏱ Preliminary Simulations & Diagnostics</b></summary>
<br>

| Convergence to Equilibrium | MC Relaxation |
|:---:|:---:|
| <img src="plots/fig1_convergence.png" width="400"> | <img src="plots/fig2_relaxation.png" width="400"> |

| Tunneling Dynamics ($L=16$, $\beta \simeq 0.50$) |
|:---:|
| <img src="plots/dynamics_L16_beta0.500.png" width="500"> |

</details>

<details>
<summary><b>📉 Autocorrelation & Critical Slowing Down</b></summary>
<br>

| Autocorrelation (temperature dependence) | Autocorrelation at $\beta_c$ |
|:---:|:---:|
| <img src="plots/autocorr_temp_dependence.png" width="400"> | <img src="plots/autocorr_critical_all.png" width="400"> |
| **$\tau_{int}$ Convergence (Madras–Sokal)** | **Dynamic Exponent $z$ fit** |
| <img src="plots/tau_convergence_critical_all.png" width="400"> | <img src="plots/dynamic_scaling_z.png" width="400"> |

</details>

<details>
<summary><b>🔥 Thermodynamic Observables & Phase Transition</b></summary>
<br>

| $P(m)$ — Disordered Phase | $P(m)$ — Ordered Phase |
|:---:|:---:|
| <img src="plots/hist_disordered_gaussian.png" width="400"> | <img src="plots/hist_ordered_bimodal.png" width="400"> |
| **Magnetization $\langle \|m\| \rangle$** | **Susceptibility $\chi'$** |
| <img src="plots/magnetization_abs.png" width="400"> | <img src="plots/susceptibility.png" width="400"> |
| **$\langle m \rangle$ (signed, zoom)** | **Exponent fits** |
| <img src="plots/magnetization_signed_zoom.png" width="400"> | <img src="plots/exponent_fits_magnetic.png" width="400"> |
| **Energy Density $\langle e \rangle$** | **Specific Heat $C_v$** |
| <img src="plots/energy_density.png" width="400"> | <img src="plots/specific_heat.png" width="400"> |
| **Binder Cumulant $U_4$** | **$U_4$ vs $L$ (Gaussian limit)** |
| <img src="plots/binder_cumulant.png" width="400"> | <img src="plots/binder_vs_L.png" width="400"> |

</details>

<details>
<summary><b>🔍 Error Analysis (Data Blocking)</b></summary>
<br>

| $\sigma_{\text{mean}}$ vs block size $k$ | $\langle \|m\| \rangle$ stability vs $k$ |
|:---:|:---:|
| <img src="plots/error_saturation.png" width="400"> | <img src="plots/magnetization_stability.png" width="400"> |

</details>

<details>
<summary><b>📐 Finite-Size Scaling & Critical Exponents</b></summary>
<br>

| Binder Crossing | Convergence to $\beta_{pc}$ |
|:---:|:---:|
| <img src="plots/binder_crossing_with_inset.png" width="400"> | <img src="plots/beta_pc_vs_L_convergence.png" width="400"> |
| **$\nu$ extraction (log-log)** | **Magnetic exponents $\gamma/\nu$, $\beta/\nu$** |
| <img src="plots/nu_extraction_loglog.png" width="400"> | <img src="plots/exponent_fits_magnetic.png" width="400"> |
| **$\chi'$ parabolic fit example** | **$C_v$ parabolic fit example** |
| <img src="plots/chi_parabolic_fit_example.png" width="400"> | <img src="plots/C_parabolic_fit_example.png" width="400"> |
| **$C_v$ log divergence** | **$C_{\max}(L)$ scaling fit** |
| <img src="plots/specific_heat_log.png" width="400"> | <img src="plots/specific_heat_scaling_fit.png" width="400"> |

| **Robustness vs $L_{\min}$** |
| :---: |
| <img src="plots/robustness_vs_Lmin.png" width="800"> |

</details>

<details>
<summary><b>🔄 Data Collapse Validation</b></summary>
<br>

| Susceptibility $\chi'$ Collapse | Magnetization $M$ Collapse |
|:---:|:---:|
| <img src="plots/collapse_susceptibility_inset.png" width="400"> | <img src="plots/collapse_magnetization_inset.png" width="400"> |
| **Binder $U_4$ Collapse** | **$C_v / \ln L$ Collapse** |
| <img src="plots/collapse_binder_inset.png" width="400"> | <img src="plots/collapse_cv_log.png" width="400"> |

</details>

---

## 🔧 Build System

The project uses a professional Makefile supporting incremental builds.

**Key Targets:**

* `make`: Compiles everything (Pilot + Production).
* `make pilot`: Compiles only the pilot executable.
* `make prod`: Compiles only the production executable.
* `make clean`: Removes `obj/` directory and executables.

**Compilation Flags:**

* `CFLAGS`: `-Wall -Wextra -O3 -march=native -flto -MMD -MP`
* `DUSE_BINARY_OUTPUT`: Automatically defined for binary I/O efficiency.

---

## 📚 Bibliography

### Books

* **Barone, Luciano Maria, Marinari, Enzo, Organtini, Giovanni, Ricci‑Tersenghi, Federico** *Scientific Programming: C‑Language, Algorithms and Models in Science*.
World Scientific, 2013.
ISBN: 978‑981‑4513‑40‑1.
* **Landau, David & Binder, Kurt** *A Guide to Monte Carlo Simulations in Statistical Physics* (5th ed.).
Cambridge: Cambridge University Press, 2021.
* **Newman, M. E. J. & Barkema, G. T.** *Monte Carlo Methods in Statistical Physics*.
Oxford: Clarendon Press, 1999.
ISBN: 0198517971

### Conference Proceedings

* **Sokal, Alan D.** *Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms*.
In: DeWitt-Morette, C., Cartier, P., Folacci, A. (eds.), *Functional Integration, NATO ASI B: Physics*, vol. 361, pp. 131–192.
Springer, Boston, MA (1997).
DOI: 10.1007/978-1-4899-0319-8_6

### Lecture Notes

* **Bonati, Claudio** *Some notes for “Metodi Numerici per la Fisica / Computational Physics Laboratory”*.
Lecture notes, Università di Pisa, 2025.
*Last updated: December 19, 2025.*
* **Bonati, Claudio** *Numerical Methods*.
GitHub repository: [https://github.com/claudio-bonati/NumericalMethods](https://github.com/claudio-bonati/NumericalMethods)

---

## 📝 License

This project is distributed under the MIT License.

See [LICENSE](LICENSE) for details.
