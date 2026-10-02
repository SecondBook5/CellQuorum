"""Run provenance includes optional analysis stacks without importing them."""

from importlib.metadata import PackageNotFoundError

from cellquorum.core.pipeline import _environment_stamp


def test_environment_stamp_records_gpu_and_rendering_distributions(monkeypatch):
    versions = {
        "torch": "test-torch",
        "cupy-cuda12x": "test-cupy",
        "rapids-singlecell": "test-rapids",
        "matplotlib": "test-matplotlib",
    }

    def version(name):
        if name not in versions:
            raise PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr("cellquorum.core.pipeline.importlib.metadata.version", version)
    stamp = _environment_stamp()
    for name, expected in versions.items():
        assert stamp["dependencies"][name] == expected
    assert stamp["dependencies"]["scvi-tools"] is None
    assert stamp["dependencies"]["cupy"] is None
    assert stamp["python_version"]
    assert stamp["platform"]
