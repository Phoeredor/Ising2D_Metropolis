# Final finite-size scaling analysis

`full_fss_analysis.py` is the official, deterministic FSS pipeline for the
square-lattice 2D Ising model with periodic boundary conditions. It consumes
the parsed production JSON files and uses the reduced-observable convention of
`parse_final.py`:

\[
\chi=L^2(\langle m^2\rangle-\langle|m|\rangle^2),\qquad
C=L^2(\langle e^2\rangle-\langle e\rangle^2).
\]

No factors of \(\beta\) or \(\beta^2\) are introduced.

## Run

From the repository root:

```bash
python3 scripts/production/analysis/full_fss_analysis.py
```

The joint cache is stored under
`results/production/analysis/fss_joint_cache/`. On first use, a compatible
`tmp/cache/` is copied only after every run has passed source-size, source-mtime
and NPZ validation; this avoids rereading the production binaries. Later runs
validate and reuse the official cache. Invalid runs cause an explicit error.
Raw reconstruction is possible only by explicit opt-in:

```bash
python3 scripts/production/analysis/full_fss_analysis.py --rebuild-joint-cache
```

The rebuild flag reconstructs invalidated runs only; it does not launch Monte
Carlo simulations.

## Official estimators

- Sizes: \(L=24,32,48,64,96,128\); primary cutoff \(L_{\min}=32\).
- Binder window: \(0.43\le\beta\le0.45\).
- \(\beta_{pc}\) and \(U^*\): weighted cubic Binder splines, consecutive-size
  crossings, the original 250-replica crossing bootstrap and the original
  inverse-variance average.
- \(1/\nu\): \(S(L)=|dU_4/d\beta|_{\beta_{pc}}\), the validated bootstrap
  covariance with SVD inversion, and the correlated leading-only model
  \(S=A L^{1/\nu}\). The reported \(\nu\) is \(1/(1/\nu)\).
- \(\chi_{\max}\): the exact local three-point parabolic estimator with
  Jacobian propagation through `pcov`. A spline peak is diagnostic only.
- \(M(\beta_{pc},L)\): the validated weighted spline and original parametric
  bootstrap, using one global \(\beta_{pc}\).
- Magnetic exponents: leading-only weighted fits
  \(\chi_{\max}=A_\chi L^{\gamma/\nu}\) and
  \(M=A_M L^{-\beta/\nu}\), five points, two parameters, three degrees of
  freedom, and `absolute_sigma=True`.
- The \(B L^{-2}\) term is diagnostic only. The original nested-model F-test
  is recorded for \(\nu\), \(\gamma/\nu\) and \(\beta/\nu\); it is never
  promoted when not significant.
- \(C_{\max}\) uses the same three-point parabolic estimator. The generalized
  specific-heat fit gives \(\alpha\), with the logarithmic fit and F-test as
  cross-checks.

Exact Onsager/CFT values are used only in final comparisons, never as fallback
values or fit inputs. A failed primary fit terminates the program.

## Joint correlations and derived quantities

The helper `fss_joint_correlations.py` implements the validated synchronized
block bootstrap solely to estimate the correlation structure of
\([\nu,\gamma/\nu,\beta/\nu]\):

- 500 replicas, seed `20260713`, block multiplier 1x;
- synchronized indices for \(|m|,m^2,m^4,e,e^2\) within each run;
- independent resampling between runs;
- the same replica index and the same estimators for all three exponents.

The raw bootstrap correlation matrix is calibrated to the central values and
marginal errors produced in the same official execution. Calibration verifies
means, standard deviations and correlations, as well as symmetry, eigenvalues,
positive definiteness and condition number. No PSD projection is applied; a
non-positive matrix is a hard error.

The calibrated replicas propagate correlations replica by replica to
\(\gamma\), \(\beta\), \(\eta\), \(\delta\), and
\(d_{\rm hyper}=\gamma/\nu+2\beta/\nu\). The output records the complete
hyperscaling variance identity, including the covariance term and the error
that would result from incorrectly assuming independence.

## Outputs

Numeric outputs:

- `results/production/analysis/fss_results_complete.dat`
- `results/production/analysis/fss_results_complete.json`
- `results/production/analysis/fss_joint_replicas.npz`

Exactly these 14 PDFs are produced, with the established plotting style:

1. `binder_crossing_with_inset.pdf`
2. `beta_pc_vs_L_convergence.pdf`
3. `nu_extraction_loglog.pdf`
4. `exponent_fits_magnetic.pdf`
5. `specific_heat_log.pdf`
6. `collapse_susceptibility_inset.pdf`
7. `collapse_magnetization_inset.pdf`
8. `collapse_binder_inset.pdf`
9. `collapse_cv_log.pdf`
10. `robustness_vs_Lmin.pdf`
11. `chi_parabolic_fit_example.pdf`
12. `C_parabolic_fit_example.pdf`
13. `specific_heat_scaling_fit.pdf`
14. `magnetization_vs_L.pdf`

The robustness panel is explicitly diagnostic but applies the same weighted
crossing, correlated Binder-slope and three-point-peak estimators at each
\(L_{\min}\).

## Current official marginal results

| Quantity | Result | Fit quality |
|---|---:|---:|
| \(\beta_{pc}\) | \(0.44075115\pm0.00005958\) | Binder crossings |
| \(U^*\) | \(0.6115767\pm0.0005662\) | Binder crossings |
| \(\nu\) | \(0.9818157\pm0.0166128\) | \(\chi^2_{red}=0.405540,\ p=0.749021\) |
| \(\gamma/\nu\) | \(1.7425443\pm0.0042751\) | \(\chi^2_{red}=0.325987,\ p=0.806584\) |
| \(\beta/\nu\) | \(0.1235445\pm0.0011327\) | \(\chi^2_{red}=0.280342,\ p=0.839631\) |
| \(\alpha\) | \(0.00735\pm0.02274\) | generalized specific heat |

Derived errors in the DAT and JSON outputs are joint errors and include the
measured exponent correlations.
## Current joint-derived results

| Quantity | Result |
|---|---:|
| $\gamma$ | $1.71086\pm0.02924$ |
| $\beta$ | $0.121298\pm0.002329$ |
| $\eta$ | $0.25746\pm0.00428$ |
| $\delta$ | $15.1046\pm0.1315$ |
| $d_{\rm hyper}$ | $1.98963\pm0.00497$ |

The diagnostic $L^{-2}$ correction to the Binder-slope fit is not significant
($p_F=0.075864$); the leading-only result remains official.  The hyperscaling
value lies about $2.09\,\sigma$ below $d=2$.

