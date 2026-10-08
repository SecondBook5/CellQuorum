"""Pure-Python tests for structured factorial contrasts (no R, no backend).

Contrasts are declared in factor-level terms and built from hypothetical design rows
under the SAME reference-aware coding as the observed design matrix, so the tests pin
the weight vectors, their invariance to the choice of reference level, and the failure
modes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError

from cellquorum.config.design import (
    ContrastsConfig,
    FactorialContrast,
    FactorialDesignConfig,
)
from cellquorum.config.factorial_contrasts import (
    ContrastNotEstimable,
    FactorialDesignEncoder,
    build_factorial_contrasts,
)
from cellquorum.config.factorial_design import resolve_factorial_columns
from cellquorum.core.exceptions import CellQuorumConfigError

_ADAPTER = TypeAdapter(list[FactorialContrast])


def _contrasts(*specs: dict) -> list:
    return _ADAPTER.validate_python(list(specs))


def _simple(name, factor, num, den, **at):
    return {
        "kind": "simple_effect",
        "name": name,
        "factor": factor,
        "numerator": num,
        "denominator": den,
        "at": at,
    }


def _dod(name, a, na, da, b, nb, db, **at):
    return {
        "kind": "difference_of_differences",
        "name": name,
        "factor_a": a,
        "numerator_a": na,
        "denominator_a": da,
        "factor_b": b,
        "numerator_b": nb,
        "denominator_b": db,
        "at": at,
    }


def _specimens(sites=("colon", "jej")) -> pd.DataFrame:
    rows = []
    for subject, arm in {
        "C1": "ctl",
        "C2": "ctl",
        "C3": "ctl",
        "P2": "trt",
        "P3": "trt",
        "P4": "trt",
    }.items():
        for site in sites:
            rows.append(
                {"specimen": f"{subject}_{site}", "subject": subject, "arm": arm, "site": site}
            )
    return pd.DataFrame(rows).set_index("specimen")


def _design(
    arm_ref="ctl", site_ref="colon", interactions=(("arm", "site"),)
) -> FactorialDesignConfig:
    return FactorialDesignConfig.model_validate(
        {
            "subject_col": "subject",
            "factors": {
                "arm": {"reference": arm_ref, "scope": "between_subject"},
                "site": {"reference": site_ref, "scope": "within_subject"},
            },
            "interactions": [list(i) for i in interactions],
            "random_effects": ["subject"],
        }
    )


FIVE = [
    _simple("trt_vs_ctl_colon", "arm", "trt", "ctl", site="colon"),
    _simple("trt_vs_ctl_jej", "arm", "trt", "ctl", site="jej"),
    _simple("jej_vs_colon_ctl", "site", "jej", "colon", arm="ctl"),
    _simple("jej_vs_colon_trt", "site", "jej", "colon", arm="trt"),
    _dod("arm_x_site", "arm", "trt", "ctl", "site", "jej", "colon"),
]


def _weights(result_set, name) -> list[float]:
    return list(next(c for c in result_set.contrasts if c.name == name).weights)


# --- worked 2x2 contrast matrix ---------------------------------------------------------


def test_balanced_2x2_contrast_matrix():
    cs = build_factorial_contrasts(_contrasts(*FIVE), design=_design(), sample_meta=_specimens())

    assert cs.columns == ("Intercept", "arm_trt", "site_jej", "arm_trt:site_jej")
    assert _weights(cs, "trt_vs_ctl_colon") == [0, 1, 0, 0]
    assert _weights(cs, "trt_vs_ctl_jej") == [0, 1, 0, 1]
    assert _weights(cs, "jej_vs_colon_ctl") == [0, 0, 1, 0]
    assert _weights(cs, "jej_vs_colon_trt") == [0, 0, 1, 1]
    assert _weights(cs, "arm_x_site") == [0, 0, 0, 1]

    matrix = cs.matrix()
    assert list(matrix.index) == list(cs.columns)  # fixed coefficients x contrasts
    assert list(matrix.columns) == cs.names
    assert matrix.shape == (4, 5)
    # A pairwise contrast carries no intercept component.
    assert (matrix.loc["Intercept"] == 0).all()


def test_dod_equals_difference_of_the_two_simple_effects():
    cs = build_factorial_contrasts(_contrasts(*FIVE), design=_design(), sample_meta=_specimens())
    diff = np.array(_weights(cs, "trt_vs_ctl_jej")) - np.array(_weights(cs, "trt_vs_ctl_colon"))
    assert diff.tolist() == _weights(cs, "arm_x_site")
    diff_site = np.array(_weights(cs, "jej_vs_colon_trt")) - np.array(
        _weights(cs, "jej_vs_colon_ctl")
    )
    assert diff_site.tolist() == _weights(cs, "arm_x_site")


def test_sign_reversal_of_numerator_and_denominator():
    cs = build_factorial_contrasts(
        _contrasts(
            _dod("fwd", "arm", "trt", "ctl", "site", "jej", "colon"),
            _dod("rev_a", "arm", "ctl", "trt", "site", "jej", "colon"),
            _dod("rev_b", "arm", "trt", "ctl", "site", "colon", "jej"),
            _dod("rev_both", "arm", "ctl", "trt", "site", "colon", "jej"),
            _simple("fwd_s", "arm", "trt", "ctl", site="jej"),
            _simple("rev_s", "arm", "ctl", "trt", site="jej"),
        ),
        design=_design(),
        sample_meta=_specimens(),
    )
    fwd = np.array(_weights(cs, "fwd"))
    assert (np.array(_weights(cs, "rev_a")) == -fwd).all()
    assert (np.array(_weights(cs, "rev_b")) == -fwd).all()
    assert (np.array(_weights(cs, "rev_both")) == fwd).all()  # two sign flips cancel
    assert (np.array(_weights(cs, "rev_s")) == -np.array(_weights(cs, "fwd_s"))).all()


# --- one coding for observed matrix, design rows and weights -----------------------------


def _cell_means_y() -> pd.DataFrame:
    """Deterministic data with known cell means (and per-subject offsets that cancel)."""
    meta = _specimens()
    cell = {("ctl", "colon"): 1.0, ("ctl", "jej"): 2.0, ("trt", "colon"): 1.5, ("trt", "jej"): 5.0}
    meta["y"] = [cell[(a, s)] for a, s in zip(meta["arm"], meta["site"], strict=True)]
    return meta


@pytest.mark.parametrize(
    ("arm_ref", "site_ref"), [("ctl", "colon"), ("trt", "colon"), ("ctl", "jej"), ("trt", "jej")]
)
def test_changing_references_changes_parameterization_not_the_estimand(arm_ref, site_ref):
    meta = _cell_means_y()
    design = _design(arm_ref=arm_ref, site_ref=site_ref)
    cs = build_factorial_contrasts(_contrasts(*FIVE), design=design, sample_meta=meta)
    encoder = FactorialDesignEncoder.fit(meta, design)
    x = encoder.matrix(meta).to_numpy()
    beta, *_ = np.linalg.lstsq(x, meta["y"].to_numpy(), rcond=None)

    est = {c.name: float(np.dot(c.weights, beta)) for c in cs.contrasts}
    assert est["trt_vs_ctl_colon"] == pytest.approx(0.5)
    assert est["trt_vs_ctl_jej"] == pytest.approx(3.0)
    assert est["jej_vs_colon_ctl"] == pytest.approx(1.0)
    assert est["jej_vs_colon_trt"] == pytest.approx(3.5)
    assert est["arm_x_site"] == pytest.approx(2.5)  # (5-1.5) - (2-1)


def test_reference_change_does_change_the_weight_vectors():
    a = build_factorial_contrasts(_contrasts(*FIVE), design=_design(), sample_meta=_specimens())
    b = build_factorial_contrasts(
        _contrasts(*FIVE), design=_design(arm_ref="trt", site_ref="jej"), sample_meta=_specimens()
    )
    assert a.columns != b.columns
    assert b.columns == ("Intercept", "arm_ctl", "site_colon", "arm_ctl:site_colon")
    assert _weights(a, "trt_vs_ctl_colon") != _weights(b, "trt_vs_ctl_colon")


def test_encoder_rows_reproduce_the_observed_design_matrix():
    meta = _specimens()
    design = _design(arm_ref="trt")
    encoder = FactorialDesignEncoder.fit(meta, design)
    observed = encoder.matrix(meta)
    for specimen, row in meta.iterrows():
        assert encoder.encode({"arm": row["arm"], "site": row["site"]}).tolist() == (
            observed.loc[specimen].tolist()
        )


def test_encoder_rejects_unknown_factor_level_and_unobserved_reference():
    meta = _specimens()
    encoder = FactorialDesignEncoder.fit(meta, _design())
    with pytest.raises(CellQuorumConfigError, match="Unknown factor"):
        encoder.encode({"nope": "x"})
    with pytest.raises(CellQuorumConfigError, match="no level"):
        encoder.encode({"arm": "zzz"})
    with pytest.raises(CellQuorumConfigError, match="not observed"):
        FactorialDesignEncoder.fit(meta, _design(arm_ref="WT"))


# --- three-level factor ----------------------------------------------------------------------


def test_three_level_factor_supports_a_chosen_pairwise_contrast():
    meta = _specimens(sites=("colon", "ileum", "jej"))
    cell = {
        ("ctl", "colon"): 1.0, ("ctl", "ileum"): 4.0, ("ctl", "jej"): 2.0,
        ("trt", "colon"): 1.5, ("trt", "ileum"): 6.0, ("trt", "jej"): 5.0,
    }  # fmt: skip
    meta["y"] = [cell[(a, s)] for a, s in zip(meta["arm"], meta["site"], strict=True)]
    cs = build_factorial_contrasts(
        _contrasts(
            _simple("jej_vs_colon_ctl", "site", "jej", "colon", arm="ctl"),
            _simple(
                "ileum_vs_jej_trt", "site", "ileum", "jej", arm="trt"
            ),  # neither is the reference
            _dod("arm_x_jej_vs_ileum", "arm", "trt", "ctl", "site", "jej", "ileum"),
        ),
        design=_design(),
        sample_meta=meta,
    )
    assert cs.columns == (
        "Intercept", "arm_trt", "site_ileum", "site_jej", "arm_trt:site_ileum", "arm_trt:site_jej",
    )  # fmt: skip
    encoder = FactorialDesignEncoder.fit(meta, _design())
    beta, *_ = np.linalg.lstsq(encoder.matrix(meta).to_numpy(), meta["y"].to_numpy(), rcond=None)
    est = {c.name: float(np.dot(c.weights, beta)) for c in cs.contrasts}
    assert est["jej_vs_colon_ctl"] == pytest.approx(1.0)
    assert est["ileum_vs_jej_trt"] == pytest.approx(1.0)
    assert est["arm_x_jej_vs_ileum"] == pytest.approx((5.0 - 2.0) - (6.0 - 4.0))


# --- failure modes -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (_simple("c", "bogus", "a", "b"), "unknown factor 'bogus'"),
        (_simple("c", "arm", "zzz", "ctl", site="colon"), "no level 'zzz'"),
        (_simple("c", "arm", "trt", "zzz", site="colon"), "no level 'zzz'"),
        (_simple("c", "arm", "trt", "ctl", nope="x"), "unknown factor 'nope'"),
        (_simple("c", "arm", "trt", "ctl", site="zzz"), "no level 'zzz'"),
        (_dod("c", "arm", "trt", "ctl", "site", "jej", "zzz"), "no level 'zzz'"),
    ],
)
def test_unknown_factor_or_level_fails(spec, match):
    with pytest.raises(CellQuorumConfigError, match=match):
        build_factorial_contrasts(_contrasts(spec), design=_design(), sample_meta=_specimens())


def test_simple_effect_under_an_interaction_requires_at():
    with pytest.raises(CellQuorumConfigError, match="ambiguous.*site"):
        build_factorial_contrasts(
            _contrasts(_simple("c", "arm", "trt", "ctl")),
            design=_design(),
            sample_meta=_specimens(),
        )


def test_without_an_interaction_at_is_optional():
    cs = build_factorial_contrasts(
        _contrasts(_simple("c", "arm", "trt", "ctl")),
        design=_design(interactions=()),
        sample_meta=_specimens(),
    )
    assert _weights(cs, "c") == [0, 1, 0]


def test_difference_of_differences_without_the_interaction_term_fails():
    with pytest.raises(CellQuorumConfigError, match="identically zero"):
        build_factorial_contrasts(
            _contrasts(_dod("c", "arm", "trt", "ctl", "site", "jej", "colon")),
            design=_design(interactions=()),
            sample_meta=_specimens(),
        )


def test_duplicate_names_fail_in_builder_and_config():
    with pytest.raises(CellQuorumConfigError, match="Duplicate"):
        build_factorial_contrasts(
            _contrasts(
                _simple("c", "arm", "trt", "ctl", site="jej"),
                _simple("c", "arm", "trt", "ctl", site="colon"),
            ),
            design=_design(),
            sample_meta=_specimens(),
        )
    with pytest.raises(ValidationError, match="more than once"):
        ContrastsConfig.model_validate(
            {
                "factorial": [
                    _simple("c", "arm", "trt", "ctl", site="jej"),
                    _simple("c", "arm", "trt", "ctl", site="colon"),
                ]
            }
        )
    with pytest.raises(ValidationError, match="more than once"):
        ContrastsConfig.model_validate(
            {
                "contrasts": [{"name": "c", "case": "a", "control": "b"}],
                "factorial": [_simple("c", "arm", "trt", "ctl", site="jej")],
            }
        )


@pytest.mark.parametrize(
    "spec",
    [
        _simple("c", "arm", "trt", "trt", site="jej"),
        _simple("c", "arm", "trt", "ctl", arm="x"),
        _dod("c", "arm", "trt", "ctl", "arm", "jej", "colon"),
        _dod("c", "arm", "trt", "trt", "site", "jej", "colon"),
        _dod("c", "arm", "trt", "ctl", "site", "jej", "jej"),
        {"kind": "mystery", "name": "c"},
    ],
)
def test_structurally_invalid_contrast_declarations_fail_at_config_time(spec):
    with pytest.raises(ValidationError):
        _contrasts(spec)


def test_non_estimable_contrast_in_an_observed_design_with_an_empty_cell():
    meta = _specimens()
    meta = meta[~((meta["arm"] == "trt") & (meta["site"] == "jej"))]
    cs = build_factorial_contrasts(
        _contrasts(_simple("ok", "arm", "trt", "ctl", site="colon")),
        design=_design(),
        sample_meta=meta,
    )
    assert _weights(cs, "ok") == [0, 1, 0, 0]  # still estimable from the populated cells
    for bad in (
        _dod("dod", "arm", "trt", "ctl", "site", "jej", "colon"),
        _simple("at_jej", "arm", "trt", "ctl", site="jej"),
    ):
        with pytest.raises(ContrastNotEstimable, match="not estimable"):
            build_factorial_contrasts(_contrasts(bad), design=_design(), sample_meta=meta)
    # An un-estimable contrast is a config-class error, but distinguishable for skips.
    assert issubclass(ContrastNotEstimable, CellQuorumConfigError)


# --- tested-effect text and support ------------------------------------------------------------


def test_tested_effect_text_uses_configured_levels_as_written():
    cs = build_factorial_contrasts(_contrasts(*FIVE), design=_design(), sample_meta=_specimens())
    by = {c.name: c.tested_effect for c in cs.contrasts}
    assert by["trt_vs_ctl_jej"] == "arm: trt vs ctl at site=jej"
    assert by["jej_vs_colon_trt"] == "site: jej vs colon at arm=trt"
    assert by["arm_x_site"] == "arm x site: (trt jej - trt colon) - (ctl jej - ctl colon)"


def test_contrast_support_counts_specimens_subjects_and_complete_blocks():
    meta = _specimens().drop(index="P3_jej")
    cs = build_factorial_contrasts(_contrasts(*FIVE), design=_design(), sample_meta=meta)
    sup = next(c for c in cs.contrasts if c.name == "jej_vs_colon_trt").support(meta, "subject")
    assert sup["n_subjects_in_every_component"] == 2  # P3 lacks jej
    by_site = {c["assignment"]["site"]: c for c in sup["components"]}
    assert by_site["colon"]["n_subjects"] == 3 and by_site["jej"]["n_subjects"] == 2
    dod = next(c for c in cs.contrasts if c.name == "arm_x_site").support(meta, "subject")
    assert [c["n_specimens"] for c in dod["components"]] == [2, 3, 3, 3]


def test_result_set_is_json_safe():
    import json

    cs = build_factorial_contrasts(_contrasts(*FIVE), design=_design(), sample_meta=_specimens())
    payload = json.loads(json.dumps(cs.to_dict()))
    assert payload["columns"][0] == "Intercept" and len(payload["contrasts"]) == 5


# --- config + intentional resolver precedence ---------------------------------------------------


def test_contrasts_config_keeps_legacy_list_and_adds_factorial_union():
    cfg = ContrastsConfig.model_validate(
        {"contrasts": [{"name": "LE_vs_N", "case": "LE", "control": "N"}], "factorial": FIVE}
    )
    assert [c.name for c in cfg.contrasts] == ["LE_vs_N"]
    assert [type(c).__name__ for c in cfg.factorial][-1] == "DifferenceOfDifferencesContrast"
    assert ContrastsConfig().factorial == []


def test_factorial_resolver_precedence_is_intentionally_opposite_to_the_legacy_helper():
    # PINNED ON PURPOSE. For the factorial model the explicit design value beats the
    # cohort default (specific > general). The legacy resolve_cohort_key lets the cohort
    # value win; that is historical behaviour and is deliberately left unchanged.
    from types import SimpleNamespace

    from cellquorum.config.cohort import resolve_cohort_key

    config = SimpleNamespace(
        cohort=SimpleNamespace(sample_key="cohort_lib", donor_key="cohort_donor")
    )
    fac = FactorialDesignConfig.model_validate(
        {
            "sample_col": "design_lib",
            "subject_col": "design_subject",
            "factors": {"arm": {"reference": "ctl"}},
        }
    )
    resolved = resolve_factorial_columns(config, fac)
    assert (resolved.sample_col, resolved.subject_col) == ("design_lib", "design_subject")
    assert resolve_cohort_key(config, attr="sample_key", stage_value="design_lib") == "cohort_lib"
