"""A ``methods:`` entry must name its method; a wrong key fails at config load."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.config.models import CellQuorumConfig


def _config(**stage: object) -> dict:
    return {"project": {"name": "t"}, **stage}


def test_an_entry_without_a_method_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="needs a `method:` key"):
        CellQuorumConfig.model_validate(
            _config(differential_abundance={"methods": [{"name": "proportion_ttest"}]})
        )


def test_the_error_names_the_entry_and_its_keys() -> None:
    with pytest.raises(ValidationError, match=r"methods\[1\].*\['name'\]"):
        CellQuorumConfig.model_validate(
            _config(
                integration={"methods": [{"method": "harmony"}, {"name": "scvi"}]},
            )
        )


def test_a_blank_method_is_rejected() -> None:
    with pytest.raises(ValidationError, match="needs a `method:` key"):
        CellQuorumConfig.model_validate(_config(embeddings={"methods": [{"method": " "}]}))


def test_entries_that_name_their_method_are_accepted_with_their_own_settings() -> None:
    config = CellQuorumConfig.model_validate(
        _config(
            differential_abundance={
                "methods": [{"method": "proportion_ttest", "paired": True}, {"method": "milo"}]
            }
        )
    )
    assert [m["method"] for m in config.differential_abundance.methods] == [
        "proportion_ttest",
        "milo",
    ]


def test_an_empty_methods_list_still_means_use_the_stage_default() -> None:
    config = CellQuorumConfig.model_validate(_config(differential_abundance={"methods": []}))
    assert config.differential_abundance.methods == []
