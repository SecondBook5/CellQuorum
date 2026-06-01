"""Configuration loading utilities for CellQuorum."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.exceptions import CellQuorumConfigError


class ConfigLoadError(CellQuorumConfigError):
    """
    Report a configuration loading or validation failure.

    CellQuorum uses OmegaConf for flexible YAML loading and Pydantic for strict
    runtime validation. This exception wraps file loading, conversion, and
    validation failures with messages that are clearer for CLI and API users.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize a configuration loading error.

        Args:
            message: User-facing error message.
        """

        # Initialize the CellQuorumConfigError base class with the user-facing message.
        super().__init__(message)


def load_config(config_path: str | Path) -> CellQuorumConfig:
    """
    Load and validate a CellQuorum configuration file.

    This is the main file-based configuration entry point. It loads a YAML file
    through OmegaConf, resolves interpolations, converts the result to plain
    Python containers, and validates the final structure with Pydantic.

    Args:
        config_path: Path to a CellQuorum YAML configuration file.

    Returns:
        Validated CellQuorumConfig object.

    Raises:
        ConfigLoadError: If the file path is invalid, loading fails, or Pydantic
            validation fails.
    """

    # Normalize the config path to a Path object.
    path = Path(config_path).expanduser()

    # Validate that the config file exists.
    if not path.exists():
        raise ConfigLoadError(f"Configuration file does not exist: {path}")

    # Validate that the config path points to a file.
    if not path.is_file():
        raise ConfigLoadError(f"Configuration path is not a file: {path}")

    # Validate that the config file has a supported suffix.
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigLoadError(
            "CellQuorum configuration files must use '.yaml' or '.yml'. " f"Received: {path.name}"
        )

    # Try loading the YAML file through OmegaConf.
    try:
        # Load the YAML configuration as an OmegaConf object.
        raw_config = OmegaConf.load(path)

    # Convert OmegaConf loading failures into a CellQuorum-specific error.
    except Exception as error:
        raise ConfigLoadError(f"Failed to load configuration file '{path}': {error}") from error

    # Validate the loaded OmegaConf object.
    return validate_omegaconf(raw_config)


def validate_omegaconf(config: DictConfig) -> CellQuorumConfig:
    """
    Validate an OmegaConf configuration object.

    This helper is useful for both file-based config loading and future Hydra
    composition. Hydra can compose config groups into an OmegaConf object, and
    this function converts that resolved object into the strict CellQuorum
    Pydantic model.

    Args:
        config: OmegaConf DictConfig object.

    Returns:
        Validated CellQuorumConfig object.

    Raises:
        ConfigLoadError: If the input is not a DictConfig or validation fails.
    """

    # Validate that the incoming object is an OmegaConf DictConfig.
    if not isinstance(config, DictConfig):
        raise ConfigLoadError(
            "validate_omegaconf expected an OmegaConf DictConfig. "
            f"Received: {type(config).__name__}"
        )

    # Try converting the resolved OmegaConf object into plain Python containers.
    try:
        # Resolve interpolations and convert nested OmegaConf containers.
        container = OmegaConf.to_container(config, resolve=True)

    # Convert OmegaConf resolution failures into a CellQuorum-specific error.
    except Exception as error:
        raise ConfigLoadError(f"Failed to resolve OmegaConf configuration: {error}") from error

    # Validate that the resolved container is a mapping.
    if not isinstance(container, Mapping):
        raise ConfigLoadError(
            "Resolved configuration must be a mapping at the top level. "
            f"Received: {type(container).__name__}"
        )

    # Validate the resolved mapping through Pydantic.
    return validate_config_dict(container)


def validate_config_dict(config: Mapping[str, Any]) -> CellQuorumConfig:
    """
    Validate a plain Python configuration mapping.

    This helper supports programmatic API usage and tests. It is also the final
    validation point after YAML or Hydra/OmegaConf composition.

    Args:
        config: Plain Python mapping containing CellQuorum configuration values.

    Returns:
        Validated CellQuorumConfig object.

    Raises:
        ConfigLoadError: If Pydantic validation fails.
    """

    # Try validating the mapping with the strict top-level Pydantic model.
    try:
        # Return the validated CellQuorum configuration model.
        return CellQuorumConfig.model_validate(dict(config))

    # Convert Pydantic validation errors into a CellQuorum-specific error.
    except ValidationError as error:
        raise ConfigLoadError(f"Invalid CellQuorum configuration:\n{error}") from error


def save_resolved_config(config: CellQuorumConfig, output_path: str | Path) -> Path:
    """
    Save a validated CellQuorum configuration as JSON.

    The resolved, validated config should be stored in run provenance so every
    analysis can be reproduced. JSON is used here because Pydantic models can
    serialize it directly and downstream tools can read it easily.

    Args:
        config: Validated CellQuorumConfig object.
        output_path: Destination JSON path.

    Returns:
        Path to the written configuration file.

    Raises:
        TypeError: If config is not a CellQuorumConfig.
        ValueError: If output_path does not end in .json.
    """

    # Validate that the caller provided a CellQuorumConfig object.
    if not isinstance(config, CellQuorumConfig):
        raise TypeError(
            "save_resolved_config expected a CellQuorumConfig object. "
            f"Received: {type(config).__name__}"
        )

    # Normalize the output path.
    path = Path(output_path).expanduser()

    # Validate that the output path uses a JSON suffix.
    if path.suffix.lower() != ".json":
        raise ValueError(
            "Resolved configuration output must use a '.json' suffix. " f"Received: {path.name}"
        )

    # Create the parent directory if needed.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write the validated config as pretty JSON.
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    # Return the written path.
    return path
