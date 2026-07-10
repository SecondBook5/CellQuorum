"""Abstract strategy base for interchangeable analysis methods.

Every tool that implements a stage (SoupX/DecontX/CellBender for ambient
correction; LIANA/CellPhoneDB for communication; ...) subclasses
``AnalysisMethod``. The base owns the shared control flow — skip-guards,
input-contract validation — so concrete methods only implement their contract
and their computation. This is the Strategy pattern: the stage holds a method,
and the method is chosen by config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import anndata as ad

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult


@dataclass(frozen=True)
class MethodSkip:
    """
    Explain why a method declined to run (a recorded, non-silent skip).

    Args:
        reason: Human-readable skip reason.
        details: Structured details for provenance and reports.
    """

    # Store the human-readable skip reason.
    reason: str

    # Store structured skip details.
    details: dict[str, object] = field(default_factory=dict)


class AnalysisMethod(ABC):
    """
    Abstract base for one interchangeable analysis method (a strategy).

    Subclasses set the class attributes ``name`` (snake_case), ``stage_category``
    (which stage they implement), and ``backend`` (execution backend), and
    implement ``input_contract`` and ``_run``. The concrete ``run`` template
    method wires in skip-guards and contract validation.
    """

    # Stable method name (snake_case); set by subclasses.
    name: str

    # Stage category this method implements (e.g. 'ambient_correction').
    stage_category: str

    # Execution backend name (python/r/rscript/gpu/rapids).
    backend: str

    @abstractmethod
    def input_contract(self, config: dict) -> DataContract:
        """Return the DataContract this method requires on its input."""

    @abstractmethod
    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult:
        """Execute the method. Implementations assume the contract has passed."""

    def min_donors(self) -> int:
        """Minimum distinct donors required; 0 means no donor guard."""

        # Default: no donor requirement.
        return 0

    def requires_layers(self) -> list[str]:
        """Layers that must exist for the method to run; [] means none."""

        # Default: no extra layer requirement beyond the contract.
        return []

    def run(
        self,
        adata: ad.AnnData,
        config: dict,
        context: object,
        *,
        donor_col: str | None = None,
    ) -> StageResult | MethodSkip:
        """
        Template method: evaluate skip-guards, validate contract, then run.

        Args:
            adata: Input AnnData.
            config: Resolved method configuration.
            context: Pipeline context (opaque here; passed through to ``_run``).
            donor_col: obs column identifying donors, for the min_donors guard.

        Returns:
            A StageResult on success, or a MethodSkip when a guard trips.

        Raises:
            CellQuorumContractError: If the input contract is violated.
        """

        # ---- Skip-guard: minimum donors ---- #
        min_d = self.min_donors()
        if min_d > 0:
            if donor_col is None or donor_col not in adata.obs.columns:
                return MethodSkip(
                    reason=f"min_donors={min_d} requested but donor column is unavailable",
                    details={"method": self.name, "donor_col": donor_col},
                )
            n_donors = int(adata.obs[donor_col].nunique())
            if n_donors < min_d:
                return MethodSkip(
                    reason=(
                        f"method '{self.name}' skipped: n_donors={n_donors} < min_donors={min_d}"
                    ),
                    details={"method": self.name, "n_donors": n_donors, "min_donors": min_d},
                )

        # ---- Skip-guard: required layers ---- #
        missing = [ly for ly in self.requires_layers() if ly not in adata.layers]
        if missing:
            return MethodSkip(
                reason=f"method '{self.name}' skipped: required layers absent {missing}",
                details={"method": self.name, "missing_layers": missing},
            )

        # ---- Contract validation (raises on violation) ---- #
        self.input_contract(config).validate(adata)

        # ---- Execute the concrete method ---- #
        return self._run(adata, config, context)


__all__ = ["AnalysisMethod", "MethodSkip"]
