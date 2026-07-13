from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "production" / "analysis"
REFERENCE = json.loads((ROOT / "tests" / "fixtures" / "fss_regression_reference.json").read_text())
SOURCE_PATH = ROOT / "scripts" / "production" / "analysis" / "full_fss_analysis.py"

EXPECTED_PLOTS = {
    "binder_crossing_with_inset.pdf", "beta_pc_vs_L_convergence.pdf",
    "nu_extraction_loglog.pdf", "exponent_fits_magnetic.pdf", "specific_heat_log.pdf",
    "collapse_susceptibility_inset.pdf", "collapse_magnetization_inset.pdf",
    "collapse_binder_inset.pdf", "collapse_cv_log.pdf", "robustness_vs_Lmin.pdf",
    "chi_parabolic_fit_example.pdf", "C_parabolic_fit_example.pdf",
    "specific_heat_scaling_fit.pdf", "magnetization_vs_L.pdf",
}


def require_official_outputs(*names: str) -> None:
    missing = [name for name in names if not (OUTPUT / name).is_file()]
    if missing:
        pytest.skip(
            "local production outputs are not versioned: " + ", ".join(missing)
        )


def test_json_and_npz_regression() -> None:
    require_official_outputs("fss_results_complete.json", "fss_joint_replicas.npz")
    result = json.loads((OUTPUT / "fss_results_complete.json").read_text())
    comparisons = {
        "beta_pc": result["beta_pc"]["value"], "U_star": result["U_star"]["value"],
        "nu": result["nu"]["value"], "nu_error": result["nu"]["error"],
        "gamma_nu": result["gamma_nu"]["value"], "gamma_nu_error": result["gamma_nu"]["error"],
        "beta_nu": result["beta_nu"]["value"], "beta_nu_error": result["beta_nu"]["error"],
    }
    tolerances = {"beta_pc": 2e-10, "U_star": 2e-9, "nu": 2e-9, "nu_error": 2e-9,
                  "gamma_nu": 2e-10, "gamma_nu_error": 2e-10,
                  "beta_nu": 2e-10, "beta_nu_error": 2e-10}
    for name, actual in comparisons.items():
        assert abs(actual - REFERENCE[name]) <= tolerances[name]
    for prefix in ("nu", "gamma_nu", "beta_nu"):
        quality = result["fit_quality"][prefix]
        assert round(quality["chi2_red"], 6) == round(REFERENCE[f"{prefix}_chi2_red"], 6)
        assert round(quality["p_value"], 6) == round(REFERENCE[f"{prefix}_p_value"], 6)
    np.testing.assert_allclose(result["correlation_matrix"], REFERENCE["correlation_matrix"], atol=2e-9)

    for name, expected in REFERENCE["derived"].items():
        actual = result["derived"][name]
        assert abs(actual["central"] - expected["central"]) <= 2e-9
        assert abs(actual["error"] - expected["error"]) <= 2e-9
    assert abs(result["F_tests"]["nu"]["p_value"] - REFERENCE["nu_f_test_p_value"]) <= 2e-9

    with np.load(OUTPUT / "fss_joint_replicas.npz") as archive:
        assert set(archive.files) == {"raw", "valid", "calibrated", "derived", "names",
                                      "derived_names", "R_boot", "C_cal"}
        assert archive["raw"].shape == (500, 3)
        assert archive["calibrated"].shape == (500, 3)
        assert archive["derived"].shape == (500, 5)
        assert archive["raw"].dtype == np.float64
        assert archive["valid"].all()
        assert np.isfinite(archive["raw"]).all()


def test_exact_plot_inventory() -> None:
    plot_dir = ROOT / "results" / "production" / "plots" / "fss"
    if not plot_dir.is_dir():
        pytest.skip("local production plots are not versioned")
    actual = {path.name for path in plot_dir.iterdir()}
    assert actual == EXPECTED_PLOTS


def test_primary_fits_have_no_exact_value_fallback() -> None:
    source = SOURCE_PATH.read_text()
    tree = ast.parse(source)
    primary_names = {"extract_nu_official", "extract_magnetic_exponents"}
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in primary_names:
        segment = ast.get_source_segment(source, functions[name])
        assert "= NU_EXACT" not in segment
        assert "= GAMMA_NU_EXACT" not in segment
        assert "= BETA_NU_EXACT" not in segment
    assert "popt = [1.0, beta_nu]" not in ast.get_source_segment(source, functions["plot_magnetization_vs_L"])

def test_no_provisional_independent_derived_errors() -> None:
    source = SOURCE_PATH.read_text()
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    segment = ast.get_source_segment(source, functions["extract_magnetic_exponents"])
    assert "gamma_err = np.sqrt" not in segment
    assert "delta_err = np.sqrt" not in segment
    assert "hyperscaling_err = np.sqrt" not in segment
    assert "report_joint_derived_quantities(joint_result)" in source
