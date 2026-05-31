"""Tests for CellQuorum configuration loading utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cellquorum.config.loader import (
    ConfigLoadError,
    load_config,
    save_resolved_config,
    validate_config_dict,
    validate_omegaconf,
)
from cellquorum.config.models import CellQuorumConfig


def test_validate_config_dict_accepts_valid_minimal_config() -> None:
    """
    Verify that a plain Python dictionary can be validated into CellQuorumConfig.

    This protects the programmatic API path, where users may construct config
    dictionaries directly instead of loading YAML files through OmegaConf or
    Hydra.
    """

    # Build a minimal valid configuration dictionary.
    config_dict = {
        "project": {
            "name": "minimal_project",
            "organism": "human",
            "species_id": 9606,
        },
        "run": {
            "profile": "standard",
            "random_seed": 42,
        },
    }

    # Validate the dictionary into a CellQuorumConfig object.
    config = validate_config_dict(config_dict)

    # Confirm the validated object has the expected type.
    assert isinstance(config, CellQuorumConfig)

    # Confirm project metadata was retained.
    assert config.project.name == "minimal_project"

    # Confirm run metadata was retained.
    assert config.run.random_seed == 42


def test_validate_config_dict_rejects_invalid_config() -> None:
    """
    Verify that invalid dictionaries raise ConfigLoadError.

    Pydantic validation errors should be wrapped in CellQuorum-specific errors so
    CLI and API users receive consistent configuration failure messages.
    """

    # Build an invalid configuration dictionary with an unsupported profile.
    config_dict = {
        "run": {
            "profile": "not_a_real_profile",
        }
    }

    # Confirm invalid config raises a CellQuorum-specific loading error.
    with pytest.raises(ConfigLoadError, match="Invalid CellQuorum configuration"):
        validate_config_dict(config_dict)


def test_validate_config_dict_rejects_unknown_fields() -> None:
    """
    Verify that unknown configuration fields fail validation.

    Strict configuration validation prevents misspelled YAML keys from silently
    changing pipeline behavior.
    """

    # Build a config dictionary containing an unsupported top-level key.
    config_dict = {
        "project": {
            "name": "test_project",
        },
        "not_a_valid_key": True,
    }

    # Confirm unknown fields raise a CellQuorum-specific loading error.
    with pytest.raises(ConfigLoadError, match="Invalid CellQuorum configuration"):
        validate_config_dict(config_dict)


def test_validate_omegaconf_accepts_valid_dict_config() -> None:
    """
    Verify that an OmegaConf DictConfig can be validated.

    This is the path Hydra will use after composing config groups. The resolved
    OmegaConf object must become the strict Pydantic CellQuorumConfig before
    execution.
    """

    # Build an OmegaConf configuration object.
    omega_config = OmegaConf.create(
        {
            "project": {
                "name": "omegaconf_project",
            },
            "compute": {
                "backend": "cpu",
                "prefer_gpu": False,
            },
        }
    )

    # Validate the OmegaConf object.
    config = validate_omegaconf(omega_config)

    # Confirm the validated object has the expected type.
    assert isinstance(config, CellQuorumConfig)

    # Confirm the project name was retained.
    assert config.project.name == "omegaconf_project"

    # Confirm the compute backend was retained.
    assert config.compute.backend == "cpu"

    # Confirm the GPU preference was retained.
    assert config.compute.prefer_gpu is False


def test_validate_omegaconf_resolves_interpolations() -> None:
    """
    Verify that OmegaConf interpolations are resolved before validation.

    Hydra/OmegaConf composition often uses interpolation for paths and project
    metadata. CellQuorum should validate the resolved values, not unresolved
    placeholders.
    """

    # Build an OmegaConf object with interpolation.
    omega_config = OmegaConf.create(
        {
            "project": {
                "name": "resolved_project",
            },
            "paths": {
                "run_root": "/tmp/${project.name}",
            },
        }
    )

    # Validate the OmegaConf object.
    config = validate_omegaconf(omega_config)

    # Confirm the interpolation was resolved before path validation.
    assert config.paths.run_root == Path("/tmp/resolved_project")


def test_validate_omegaconf_rejects_non_dict_config() -> None:
    """
    Verify that validate_omegaconf rejects non-DictConfig objects.

    This prevents accidental calls with plain dictionaries, lists, or other
    objects that should use a different validation helper.
    """

    # Confirm a plain dictionary raises a CellQuorum-specific error.
    with pytest.raises(ConfigLoadError, match="expected an OmegaConf DictConfig"):
        validate_omegaconf({"project": {"name": "bad"}})  # type: ignore[arg-type]


def test_load_config_loads_valid_yaml_file(tmp_path: Path) -> None:
    """
    Verify that load_config reads and validates a YAML configuration file.

    This is the main file-based configuration entry point used by the CLI and
    future project templates.
    """

    # Create a temporary YAML config file.
    config_path = tmp_path / "config.yaml"

    # Write a valid minimal CellQuorum config.
    config_path.write_text(
        """
project:
  name: yaml_project
run:
  profile: publication
  random_seed: 7
compute:
  backend: cpu
  prefer_gpu: false
""",
        encoding="utf-8",
    )

    # Load and validate the config file.
    config = load_config(config_path)

    # Confirm the returned object has the expected type.
    assert isinstance(config, CellQuorumConfig)

    # Confirm project metadata was loaded.
    assert config.project.name == "yaml_project"

    # Confirm run profile was loaded.
    assert config.run.profile == "publication"

    # Confirm random seed was loaded.
    assert config.run.random_seed == 7

    # Confirm compute backend was loaded.
    assert config.compute.backend == "cpu"


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    """
    Verify that load_config rejects missing files with a clear error.

    Missing config files should fail before OmegaConf attempts to parse them.
    """

    # Build a path that does not exist.
    missing_path = tmp_path / "missing.yaml"

    # Confirm loading a missing file raises a CellQuorum-specific error.
    with pytest.raises(ConfigLoadError, match="does not exist"):
        load_config(missing_path)


def test_load_config_rejects_directory_path(tmp_path: Path) -> None:
    """
    Verify that load_config rejects directory paths.

    Users should pass a YAML file, not a directory containing configuration
    groups.
    """

    # Confirm loading a directory raises a clear error.
    with pytest.raises(ConfigLoadError, match="not a file"):
        load_config(tmp_path)


def test_load_config_rejects_unsupported_suffix(tmp_path: Path) -> None:
    """
    Verify that load_config only accepts YAML suffixes.

    CellQuorum config loading should be explicit about supported file formats.
    """

    # Create a config path with an unsupported suffix.
    config_path = tmp_path / "config.json"

    # Write placeholder content so the file exists.
    config_path.write_text("{}", encoding="utf-8")

    # Confirm unsupported config suffixes are rejected.
    with pytest.raises(ConfigLoadError, match="must use '.yaml' or '.yml'"):
        load_config(config_path)


def test_load_config_rejects_invalid_yaml_content(tmp_path: Path) -> None:
    """
    Verify that load_config wraps YAML parsing failures.

    Malformed YAML should produce a CellQuorum-specific loading error so CLI
    users do not get an unhandled OmegaConf stack trace.
    """

    # Create a temporary malformed YAML file.
    config_path = tmp_path / "bad.yaml"

    # Write invalid YAML content.
    config_path.write_text("project:\n  name: [unterminated\n", encoding="utf-8")

    # Confirm malformed YAML raises a CellQuorum-specific error.
    with pytest.raises(ConfigLoadError, match="Failed to load configuration file"):
        load_config(config_path)


def test_load_config_rejects_yaml_that_fails_pydantic_validation(tmp_path: Path) -> None:
    """
    Verify that load_config wraps Pydantic validation failures.

    YAML files can parse successfully but still be semantically invalid. Those
    failures should still become ConfigLoadError.
    """

    # Create a temporary YAML config file.
    config_path = tmp_path / "invalid.yaml"

    # Write a YAML file with an unsupported run profile.
    config_path.write_text(
        """
run:
  profile: impossible_profile
""",
        encoding="utf-8",
    )

    # Confirm semantic validation errors are wrapped clearly.
    with pytest.raises(ConfigLoadError, match="Invalid CellQuorum configuration"):
        load_config(config_path)


def test_save_resolved_config_writes_validated_config_json(tmp_path: Path) -> None:
    """
    Verify that save_resolved_config writes a validated config as JSON.

    The validated config should be stored in run provenance so the exact runtime
    configuration can be inspected and reproduced.
    """

    # Build a validated CellQuorum config object.
    config = CellQuorumConfig(project={"name": "saved_project"})

    # Define the output JSON path.
    output_path = tmp_path / "provenance" / "resolved_config.json"

    # Save the config to disk.
    written_path = save_resolved_config(config, output_path)

    # Confirm the returned path is the target path.
    assert written_path == output_path

    # Confirm the output file exists.
    assert output_path.exists()

    # Confirm the saved JSON contains the project name.
    assert '"name": "saved_project"' in output_path.read_text(encoding="utf-8")


def test_save_resolved_config_rejects_non_config_object(tmp_path: Path) -> None:
    """
    Verify that save_resolved_config rejects non-CellQuorumConfig objects.

    This prevents unrelated dictionaries or partially validated objects from
    being written as authoritative provenance.
    """

    # Define an output path.
    output_path = tmp_path / "resolved_config.json"

    # Confirm non-config objects raise a clear type error.
    with pytest.raises(TypeError, match="expected a CellQuorumConfig object"):
        save_resolved_config({"project": "bad"}, output_path)  # type: ignore[arg-type]


def test_save_resolved_config_requires_json_suffix(tmp_path: Path) -> None:
    """
    Verify that save_resolved_config requires a .json output path.

    The resolved Pydantic config is serialized as JSON, so the filename should
    make that format explicit.
    """

    # Build a validated config.
    config = CellQuorumConfig()

    # Define a non-JSON output path.
    output_path = tmp_path / "resolved_config.yaml"

    # Confirm non-JSON suffixes are rejected.
    with pytest.raises(ValueError, match="must use a '.json' suffix"):
        save_resolved_config(config, output_path)
