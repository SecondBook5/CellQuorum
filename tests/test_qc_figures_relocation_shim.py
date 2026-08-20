"""Move 2: qc figure modules -> visualization/qc relocation keeps old import paths valid."""

from __future__ import annotations


def test_old_publication_path_resolves_to_moved_objects():
    from cellquorum.qc.publication import QCPublicationFigureError as OldErr
    from cellquorum.qc.publication import write_publication_qc_figures as old_fn
    from cellquorum.visualization.qc.publication import QCPublicationFigureError as NewErr
    from cellquorum.visualization.qc.publication import write_publication_qc_figures as new_fn

    assert OldErr is NewErr
    assert old_fn is new_fn


def test_old_visualization_path_resolves_to_moved_objects():
    from cellquorum.qc.visualization import QCVisualizationError as OldErr
    from cellquorum.qc.visualization import QCVisualizationResult as OldResult
    from cellquorum.qc.visualization import write_qc_figures as old_fn
    from cellquorum.visualization.qc.diagnostics import QCVisualizationError as NewErr
    from cellquorum.visualization.qc.diagnostics import QCVisualizationResult as NewResult
    from cellquorum.visualization.qc.diagnostics import write_qc_figures as new_fn

    assert OldErr is NewErr
    assert OldResult is NewResult
    assert old_fn is new_fn
