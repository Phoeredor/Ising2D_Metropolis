# Finite-Size Scaling Analysis — `full_fss_analysis.py`

Self-consistent FSS analysis for the 2D Ising model (square lattice, PBC, Metropolis).
All critical exponents are extracted as **free parameters** from MC data; exact Onsager values serve only as validation.

## Quick start

```bash
pip install numpy scipy matplotlib seaborn tqdm
cd scripts/production/analysis
python3 full_fss_analysis.py
```

**Input:** JSON files from `parse_final.py` at `results/production/L{L}/parsed/observables_L{L}.json`
(keys: `beta`, `U4`, `U4_err`, `M_abs`, `M_abs_err`, `chi`, `chi_err`, `C`, `C_err`).

**Lattice sizes:** L = 24, 32, 48, 64, 96, 128.

---

## Analysis pipeline

### Phase 1 — Pseudo-critical temperature β_pc

Binder cumulant U₄(β, L) crossings between consecutive L pairs.
Cubic splines (weighted, k = 3) + Brent root-finding; bootstrap (N = 250) for errors.
Inverse-variance weighted mean → β_pc ± σ and U* ± σ.

### Phase 2 — Correlation length exponent ν

Binder slope S(L) = |dU₄/dβ|_{β_pc} with parametric bootstrap errors per L.

Two fits are performed:
- **Uncorrelated** (per-phase): S(L) = A · L^{1/ν} · [1 + B · L^{−ω}], ω = 2 fixed (CFT), L_min cuts {24, 32, 48}.
- **Correlated** (main): global bootstrap builds full covariance matrix C_ij of slopes → SVD pseudo-inverse → correlated χ². The leading-only 2-parameter fit (S = A · L^{1/ν}) gives **ν_final**, used in all subsequent phases.

### Phase 3 — Magnetic exponents γ/ν and β/ν (free)

- **χ_max(L):** 3-point local parabolic fit (Rummukainen FSS standard), errors from pcov; spline cross-check.
- **M(β_pc, L):** spline interpolation + parametric bootstrap.
- Fits: A · L^{γ/ν} · [1 + B · L^{−2}] and A · L^{−β/ν} · [1 + B · L^{−2}] with exponents free.
- Derived: γ = (γ/ν)·ν, β = (β/ν)·ν, η = 2 − γ/ν. Hyperscaling check: 2β/ν + γ/ν = 2.

### Phase 3b — Sub-leading corrections

Two-step procedure + F-test for each observable (χ_max, M, S_Binder, C_max):
1. Simple power law O = A · L^α (2 params)
2. α fixed → O = A · L^α · [1 + B · L^{−ω}] (2 params)
3. F-test of 2-param vs 3-param (all free) for model selection

C_max uses logarithmic variant: a + b·ln(L) → a + b·ln(L) + D·L^{−ω}.

### Phase 4 — Specific heat exponent α

C_max via 3-point parabolic fit (same as χ_max).
Primary fit: C_max = A · L^g · (1 + q · ln L) with g = α/ν free.
Cross-check: C_max = a + b · ln(L). F-test for model selection.

### Phase 5 — Data collapse

All collapses use **measured** exponents (not Onsager):
χ/L^{γ/ν}, M·L^{β/ν}, U₄, C_v/ln(L) vs scaling variable (β − β_pc)·L^{1/ν}.

### Robustness

Three-panel stability plot of β_pc, 1/ν, γ/ν as a function of L_min cutoff.

---

## Output

All output goes to `results/production/`:

| Type | Path | Content |
|------|------|---------|
| Plots (14 PDF) | `plots/fss/` | See table below |
| Results | `analysis/fss_results_complete.dat` | All numerical results (structured text) |
| LaTeX tables | `analysis/fss_tables_complete.tex` | 5 `booktabs` tables (see below) |

### Plots

| # | File | Content |
|---|------|---------|
| 1 | `binder_crossing_with_inset.pdf` | U₄ crossing + zoom inset |
| 2 | `beta_pc_vs_L_convergence.pdf` | β_pc convergence vs L |
| 3 | `nu_extraction_loglog.pdf` | ν extraction (log-log) |
| 4 | `exponent_fits_magnetic.pdf` | γ/ν and β/ν (two-panel) |
| 5 | `specific_heat_log.pdf` | C_max vs ln(L) |
| 6–9 | `collapse_*.pdf` | χ, M, U₄, C_v data collapse |
| 10 | `robustness_vs_Lmin.pdf` | Exponent stability vs L_min |
| S1 | `chi_parabolic_fit_example.pdf` | χ(β) parabolic fit, L = 96 |
| S2 | `C_parabolic_fit_example.pdf` | C(β) parabolic fit, L = 96 |
| S3 | `specific_heat_scaling_fit.pdf` | C_max(L) generalized fit |
| S4 | `magnetization_vs_L.pdf` | M(β_pc) vs L (log-log) |

### LaTeX tables

| Label | Content |
|-------|---------|
| `tab:fss_crossings` | Binder crossings (L₁×L₂, β_cross, U*) |
| `tab:fss_amplitudes` | Leading amplitudes and correction terms |
| `tab:fss_corrections` | F-test results per observable |
| `tab:fss_exponents_summary` | All exponents vs Onsager + hyperscaling |
| `tab:fss_onsager_comparison` | Deviation analysis (abs, rel, σ) |

---

## Configuration

Key parameters at the top of the script:

```python
L_LIST         = [24, 32, 48, 64, 96, 128]
BETA_RANGE_FIT = (0.43, 0.45)
N_BOOTSTRAP    = 250
OMEGA_EXACT    = 2.0       # CFT correction-to-scaling exponent (fixed)
```

---

## References

- Onsager, L. (1944). *Phys. Rev.* **65**, 117.
- Binder, K. (1981). *Z. Phys. B* **43**, 119.
- Newman, M. E. J. & Barkema, G. T. (1999). *Monte Carlo Methods in Statistical Physics*. Oxford.
- Amit, D. J. & Martín-Mayor, V. (2005). *Field Theory, the Renormalization Group, and Critical Phenomena*. World Scientific.
