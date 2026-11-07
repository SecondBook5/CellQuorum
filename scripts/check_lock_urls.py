#!/usr/bin/env python3
"""Check that every package pinned in the conda-lock files can still be downloaded.

A lock file fixes exact package builds, but it only reproduces an environment while those
builds still exist on the channel. Channels remove old builds (the ``rapidsai`` channel removed a
pinned ``libxgboost`` build, which broke the GPU image), so a lock can rot silently. This sends a
HEAD request for each pinned URL and reports any that no longer resolve, so rot is found by a
scheduled check instead of by a failed build.

Usage:
    python scripts/check_lock_urls.py                      # every envs/*.conda-lock.yml
    python scripts/check_lock_urls.py envs/cellquorum-gpu.conda-lock.yml
Exit status is 1 when any pinned URL is gone.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def pinned_urls(lock: Path) -> list[tuple[str, str]]:
    """(manager, url) for every package in a conda-lock file, linux-64 only."""
    data = yaml.safe_load(lock.read_text())
    out = []
    for package in data.get("package", []):
        if package.get("platform", "linux-64") != "linux-64":
            continue
        # A git+https pin (a package installed from a commit) is not a file a HEAD request can test.
        if str(package["url"]).startswith("git+"):
            continue
        out.append((package.get("manager", "conda"), package["url"]))
    return out


def reachable(url: str) -> tuple[str, bool, str]:
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "cellquorum-lock-check"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed https URLs from a lock file
            return url, response.status < 400, str(response.status)
    except urllib.error.HTTPError as err:
        return url, False, str(err.code)
    except Exception as err:  # noqa: BLE001 - a network error is reported, not hidden
        return url, False, type(err).__name__


def main() -> int:
    locks = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "envs").glob("*.conda-lock.yml"))
    missing_total = 0
    for lock in locks:
        urls = pinned_urls(lock)
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(lambda item: reachable(item[1]), urls))
        gone = [(url, status) for url, ok, status in results if not ok]
        print(f"{lock.name}: {len(urls) - len(gone)} of {len(urls)} pinned packages still resolve")
        for url, status in gone:
            print(f"  GONE ({status}) {url}")
        missing_total += len(gone)
    return 1 if missing_total else 0


if __name__ == "__main__":
    sys.exit(main())
