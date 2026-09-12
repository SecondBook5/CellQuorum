"""Run conda-lock 4.0.2 with complete manylinux tags through its glibc ceiling.

Upstream pypi_solver.MANYLINUX_TAGS omits intermediate PEP 600 tags, including
2.27 used by JAX and NVIDIA wheels. Keep its existing glibc filtering and ceiling;
only fill gaps. Recheck upstream before changing the tool version.
"""

from importlib.metadata import PackageNotFoundError, version


def complete_manylinux_tags(tags: list[str]) -> list[str]:
    ceiling = max(int(tag.removeprefix("_2_")) for tag in tags if tag.startswith("_2_"))
    legacy = [tag for tag in tags if not tag.startswith("_2_")]
    return legacy + [f"_2_{minor}" for minor in range(5, ceiling + 1)]


def main() -> None:
    try:
        installed = version("conda-lock")
    except PackageNotFoundError:
        raise SystemExit("Install conda-lock==4.0.2 in this Python environment.") from None
    if installed != "4.0.2":
        raise SystemExit(
            f"Expected conda-lock 4.0.2, found {installed}; review the manylinux workaround."
        )

    from conda_lock import pypi_solver
    from conda_lock.conda_lock import main as conda_lock_main

    pypi_solver.MANYLINUX_TAGS = complete_manylinux_tags(pypi_solver.MANYLINUX_TAGS)
    conda_lock_main()


if __name__ == "__main__":
    main()
