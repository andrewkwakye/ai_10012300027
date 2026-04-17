# CS4241 - Introduction to Artificial Intelligence
# Author: Andrew Kofi Kwakye
# Index: 10012300027
"""
Downloads the two source datasets specified in the exam question:

    1) Ghana_Election_Result.csv  (GitHub raw URL)
    2) 2025 Budget Statement PDF  (mofep.gov.gh)

Idempotent: skips files that already exist with non-zero size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

# Make the src package importable when running this file directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CSV_PATH, CSV_URL, PDF_PATH, PDF_URL  # noqa: E402
from src.logger import get_logger  # noqa: E402

log = get_logger("download")


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log.info(f"Skipping {dest.name} (already present, {dest.stat().st_size} bytes)")
        return
    log.info(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # stream to avoid loading big PDFs into RAM
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
    log.info(f"Saved {dest} ({dest.stat().st_size} bytes)")


def main() -> None:
    _download(CSV_URL, CSV_PATH)
    _download(PDF_URL, PDF_PATH)


if __name__ == "__main__":
    main()
