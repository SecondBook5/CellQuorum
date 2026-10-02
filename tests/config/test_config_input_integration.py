"""Tests for top-level CellQuorum input configuration."""

from __future__ import annotations

# Import Path for config path assertions and YAML fixture paths.
from pathlib import Path

# Import pytest for exception assertions.
import pytest

# Import configuration loading utilities.
from cellquorum.config.loader import ConfigLoadError, load_config, validate_config_dict

# Import top-level configuration models.
from cellquorum.config.models import CellQuorumConfig, InputConfig


def test_input_config_defaults_to_no_h5ad() -> None:
    """
    Verify InputConfig defaults to no file input.

    Programmatic workflows may provide AnnData directly in PipelineContext, so the
    input h5ad path should be optional.
    """

    # Build a default input config.
    config = InputConfig()

    # Confirm no h5ad path is required by default.
    assert config.h5ad is None

    # Confirm no counts layer is required by default.
    assert config.counts_layer is None


def test_input_config_accepts_h5ad_path() -> None:
    """
    Verify InputConfig accepts a valid h5ad path.

    The config model checks path shape, while file existence is checked later by
    the AnnData I/O layer.
    """

    # Build an input config with an h5ad path.
    config = InputConfig(h5ad="data/example.h5ad")

    # Confirm the string path was coerced to Path.
    assert config.h5ad == Path("data/example.h5ad")


def test_input_config_rejects_non_h5ad_suffix() -> None:
    """
    Verify InputConfig rejects unsupported input suffixes.

    The first CellQuorum input mode is intentionally h5ad-only.
    """

    # Confirm unsupported suffixes fail validation.
    with pytest.raises(ValueError, match="must point to a '.h5ad' file"):
        InputConfig(h5ad="data/example.csv")


def test_input_config_strips_counts_layer() -> None:
    """
    Verify InputConfig strips harmless whitespace from counts_layer.

    YAML values sometimes include accidental whitespace; preserving the cleaned
    value avoids downstream layer lookup surprises.
    """

    # Build an input config with padded layer text.
    config = InputConfig(counts_layer=" counts ")

    # Confirm the layer name was stripped.
    assert config.counts_layer == "counts"


def test_input_config_rejects_empty_counts_layer() -> None:
    """
    Verify InputConfig rejects empty counts layer names.

    Empty strings should not be treated as meaningful AnnData layer names.
    """

    # Confirm empty layer names fail validation.
    with pytest.raises(ValueError, match="input.counts_layer cannot be empty"):
        InputConfig(counts_layer="   ")


def test_cellquorum_config_default_includes_input_config() -> None:
    """
    Verify CellQuorumConfig includes input settings by default.

    This creates the top-level config slot needed before pipeline execution can
    load AnnData into context.adata.
    """

    # Build a default top-level config.
    config = CellQuorumConfig()

    # Confirm input config exists.
    assert isinstance(config.input, InputConfig)

    # Confirm default input values are empty.
    assert config.input.h5ad is None
    assert config.input.counts_layer is None


def test_cellquorum_config_accepts_input_mapping_directly() -> None:
    """
    Verify CellQuorumConfig coerces an input mapping into InputConfig.

    This protects programmatic construction of top-level configs.
    """

    # Build a config with an input mapping.
    config = CellQuorumConfig(
        input={
            "h5ad": "data/project.h5ad",
            "counts_layer": "counts",
        }
    )

    # Confirm the input block was parsed into InputConfig.
    assert isinstance(config.input, InputConfig)

    # Confirm input values were retained.
    assert config.input.h5ad == Path("data/project.h5ad")
    assert config.input.counts_layer == "counts"


def test_validate_config_dict_accepts_input_block() -> None:
    """
    Verify validate_config_dict accepts a top-level input block.

    This is the dictionary/API path for configuring h5ad input files.
    """

    # Validate a dictionary with input settings.
    config = validate_config_dict(
        {
            "project": {
                "name": "input_dict_project",
            },
            "input": {
                "h5ad": "data/input_dict_project.h5ad",
                "counts_layer": "raw_counts",
            },
        }
    )

    # Confirm top-level config was returned.
    assert isinstance(config, CellQuorumConfig)

    # Confirm input settings were parsed.
    assert isinstance(config.input, InputConfig)
    assert config.input.h5ad == Path("data/input_dict_project.h5ad")
    assert config.input.counts_layer == "raw_counts"


def test_validate_config_dict_rejects_unknown_input_keys() -> None:
    """
    Verify unknown nested input keys still fail strict validation.

    Adding input config must not weaken strict config validation.
    """

    # Confirm unknown input keys fail.
    with pytest.raises(ConfigLoadError, match="Invalid CellQuorum configuration"):
        validate_config_dict(
            {
                "project": {
                    "name": "bad_input_project",
                },
                "input": {
                    "not_a_real_input_key": True,
                },
            }
        )


def test_validate_config_dict_rejects_bad_h5ad_suffix() -> None:
    """
    Verify validate_config_dict rejects non-h5ad input paths.

    Suffix errors should surface during config validation before runtime I/O.
    """

    # Confirm unsupported input suffixes fail through the loader wrapper.
    with pytest.raises(ConfigLoadError, match="Invalid CellQuorum configuration"):
        validate_config_dict(
            {
                "project": {
                    "name": "bad_suffix_project",
                },
                "input": {
                    "h5ad": "data/not_h5ad.txt",
                },
            }
        )


def test_load_config_accepts_yaml_input_block(tmp_path: Path) -> None:
    """
    Verify YAML config files can include an input block.

    This is the future CLI path for specifying the h5ad file to load.
    """

    # Build a config file path.
    config_path = tmp_path / "config.yaml"

    # Write a minimal YAML config with input settings.
    config_path.write_text(
        """
project:
  name: yaml_input_project
compute:
  backend: cpu
  prefer_gpu: false
input:
  h5ad: data/yaml_input_project.h5ad
  counts_layer: counts
""",
        encoding="utf-8",
    )

    # Load the config file.
    config = load_config(config_path)

    # Confirm the top-level config loaded.
    assert isinstance(config, CellQuorumConfig)

    # Confirm input settings loaded.
    assert isinstance(config.input, InputConfig)
    assert config.project.name == "yaml_input_project"
    assert config.input.h5ad == Path("data/yaml_input_project.h5ad")
    assert config.input.counts_layer == "counts"


def test_load_config_rejects_empty_yaml_counts_layer(tmp_path: Path) -> None:
    """
    Verify YAML config loading rejects empty counts_layer.

    Empty layer names should fail during config validation.
    """

    # Build a config file path.
    config_path = tmp_path / "bad_config.yaml"

    # Write config with invalid empty counts layer.
    config_path.write_text(
        """
project:
  name: bad_yaml_input_project
input:
  h5ad: data/example.h5ad
  counts_layer: "   "
""",
        encoding="utf-8",
    )

    # Confirm invalid YAML config fails.
    with pytest.raises(ConfigLoadError, match="Invalid CellQuorum configuration"):
        load_config(config_path)


def test_input_config_accepts_an_exclusion_rule() -> None:
    """
    Verify InputConfig carries an exclusion rule alongside the inclusion one.

    An inclusion list cannot express "everything except": dropping one artifact
    cluster from a 39-cluster partition by subset means naming the other 38, which
    is unreadable and silently incomplete the next time the object is re-clustered.
    """

    config = InputConfig(
        h5ad="data/example.h5ad",
        exclude={"column": "leiden", "values": ["22"]},
    )

    assert config.exclude is not None
    assert config.exclude.column == "leiden"
    assert config.exclude.values == ["22"]

    # The two rules are independent: an exclusion does not imply a subset.
    assert config.subset is None


def test_input_config_rejects_an_empty_exclusion() -> None:
    """
    Verify an exclusion with no values is refused.

    A rule that names a column and nothing to drop from it reads like a filter and
    removes nothing, which is the failure mode the loader's vocabulary guard also
    exists to prevent.
    """

    with pytest.raises(ValueError, match="values"):
        InputConfig(h5ad="data/example.h5ad", exclude={"column": "leiden", "values": []})


def test_load_config_accepts_a_yaml_exclusion_block(tmp_path: Path) -> None:
    """
    Verify input.exclude round-trips through YAML config loading.

    The artifact-cluster mask has to be a declared, reviewable part of the run
    config rather than a hand-edit in a driver script, or the next run silently
    analyses the debris.
    """

    config_path = tmp_path / "exclude_config.yaml"
    config_path.write_text(
        """
project:
  name: yaml_exclude_project
input:
  h5ad: data/example.h5ad
  exclude:
    column: leiden
    values:
      - "22"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.input.exclude is not None
    assert config.input.exclude.column == "leiden"
    assert config.input.exclude.values == ["22"]
