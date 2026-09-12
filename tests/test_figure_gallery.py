"""Gallery outputs must remain complete and agree with their source fixtures."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def test_gallery_exports_and_source_contracts(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_figure_gallery.py"
    spec = importlib.util.spec_from_file_location("build_figure_gallery", script)
    gallery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gallery)
    paths = gallery.build_gallery(tmp_path)
    names = {"enrichment", "embedding", "paired", "running_score", "volcano", "qc_summary"}
    assert set(paths) == {
        tmp_path / f"{name}.{fmt}" for name in names for fmt in ("png", "pdf", "svg")
    }
    for path in paths:
        assert path.stat().st_size > 100
        if path.suffix == ".pdf":
            assert path.read_bytes().startswith(b"%PDF-")
        elif path.suffix == ".png":
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            if os.environ.get("CELLQUORUM_CHECK_FIGURE_BASELINES") == "1":
                from matplotlib.testing.compare import compare_images

                baseline = script.parents[1] / "docs" / "gallery" / "assets" / path.name
                difference = compare_images(str(baseline), str(path), tol=0)
                assert difference is None, difference
        else:
            from xml.etree import ElementTree

            root = ElementTree.parse(path).getroot()
            assert root.tag.endswith("svg")
            assert any(element.tag.endswith("text") for element in root.iter())
    embedding = pd.read_csv(tmp_path / "embedding_source.csv")
    assert sorted(embedding.population.value_counts()) == [30, 140, 180]
    qc = pd.read_csv(tmp_path / "qc_summary_source.csv")
    assert len(qc) == 400 and qc.keep.sum() == 320
    assert (qc["rule:high_mito"] == ~qc.keep).all()
    paired = pd.read_csv(tmp_path / "paired_source.csv")
    result = json.loads((tmp_path / "paired_comparison.json").read_text())
    wide = paired.pivot(index="donor", columns="condition", values="value")
    assert result["n_group1"] == result["n_group2"] == len(wide)
    assert np.isclose(result["effect"], np.median(wide.Case - wide.Control))
    walk = pd.read_csv(tmp_path / "running_score_source.csv")
    assert np.isclose(walk.running_es.iloc[-1], 0)
    assert walk.hit.sum() == 12
