"""Run conda-lock 4.0.5 with complete manylinux tags through its glibc ceiling.

Upstream pypi_solver.MANYLINUX_TAGS omits intermediate PEP 600 tags, including
2.27 used by JAX and NVIDIA wheels. Keep its existing glibc filtering and ceiling;
only fill gaps. Recheck upstream before changing the tool version.

A second correction runs after the solve. A pip package can ask for a Python package that conda
already provides under another name (``palantir`` asks for ``igraph``; conda ships it as
``python-igraph``). The pip solver does not see the conda package and adds the PyPI wheel as well,
leaving two providers of one module. Those pip records are dropped when the conda package that
provides them is present, so each package has exactly one owner.
"""

import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

#: PyPI package -> the conda package that provides the same module.
PIP_PROVIDED_BY_CONDA = {"igraph": "python-igraph"}


def drop_pip_packages_provided_by_conda(lockfile: Path) -> list[str]:
    """Remove pip records that a conda package in the same lock already provides.

    Returns the names removed. Works on the text so the generated file keeps its header and
    layout; each package is one ``- name: ...`` block.
    """
    text = lockfile.read_text()
    head, sep, body = text.partition("\npackage:\n")
    if not sep:
        return []
    blocks = re.split(r"(?m)^(?=- name: )", body)
    conda_names = {
        re.match(r"- name: (\S+)", block).group(1)  # type: ignore[union-attr]
        for block in blocks
        if block.startswith("- name: ") and re.search(r"(?m)^  manager: conda$", block)
    }
    kept, dropped = [], []
    for block in blocks:
        match = re.match(r"- name: (\S+)", block)
        is_pip = bool(re.search(r"(?m)^  manager: pip$", block))
        if match and is_pip and PIP_PROVIDED_BY_CONDA.get(match.group(1)) in conda_names:
            dropped.append(match.group(1))
            continue
        kept.append(block)
    if dropped:
        lockfile.write_text(head + sep + "".join(kept))
    return dropped


def complete_manylinux_tags(tags: list[str]) -> list[str]:
    ceiling = max(int(tag.removeprefix("_2_")) for tag in tags if tag.startswith("_2_"))
    legacy = [tag for tag in tags if not tag.startswith("_2_")]
    return legacy + [f"_2_{minor}" for minor in range(5, ceiling + 1)]


def main() -> None:
    try:
        installed = version("conda-lock")
    except PackageNotFoundError:
        raise SystemExit("Install conda-lock==4.0.5 in this Python environment.") from None
    if installed != "4.0.5":
        raise SystemExit(
            f"Expected conda-lock 4.0.5, found {installed}; review the manylinux workaround."
        )

    from conda_lock import pypi_solver
    from conda_lock.conda_lock import main as conda_lock_main

    pypi_solver.MANYLINUX_TAGS = complete_manylinux_tags(pypi_solver.MANYLINUX_TAGS)
    conda_lock_main(standalone_mode=False)

    arguments = sys.argv[1:]
    for index, argument in enumerate(arguments):
        if argument == "--lockfile" and index + 1 < len(arguments):
            removed = drop_pip_packages_provided_by_conda(Path(arguments[index + 1]))
            if removed:
                print(f"Dropped pip records already provided by conda: {', '.join(removed)}")


if __name__ == "__main__":
    main()
