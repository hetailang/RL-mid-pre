#!/usr/bin/env python3
"""Download and validate DSRL datasets for validation experiments."""

import argparse
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import h5py


DATASETS = {
    "OfflineBallRun-v0": "http://data.offline-saferl.org/download/SafetyBallRun-v0-80-940.hdf5",
    "OfflineCarRun-v0": "http://data.offline-saferl.org/download/SafetyCarRun-v0-40-651.hdf5",
    "OfflineBallCircle-v0": "http://data.offline-saferl.org/download/SafetyBallCircle-v0-80-886.hdf5",
    "OfflineCarCircle-v0": "http://data.offline-saferl.org/download/SafetyCarCircle-v0-100-1450.hdf5",
}


def validate_hdf5(path: Path) -> None:
    with h5py.File(path, "r") as h5file:
        keys = list(h5file.keys())
    if not keys:
        raise RuntimeError(f"{path} opened as HDF5 but contains no top-level keys")
    print(f"validated {path} with keys: {keys[:10]}")


def download(url: str, output_dir: Path, retries: int, sleep_seconds: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rsplit("/", 1)[-1]
    output_path = output_dir / filename
    tmp_path = output_dir / f"{filename}.part"

    if output_path.exists():
        print(f"found existing file: {output_path}")
        validate_hdf5(output_path)
        return output_path

    if tmp_path.exists():
        tmp_path.unlink()

    last_error = None
    for attempt in range(1, retries + 1):
        print(f"attempt {attempt}/{retries}: {url}")
        try:
            urllib.request.urlretrieve(url, tmp_path)
            tmp_path.replace(output_path)
            print(f"downloaded {output_path} ({output_path.stat().st_size} bytes)")
            validate_hdf5(output_path)
            return output_path
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
            print(f"failed: {type(exc).__name__}: {exc}")
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < retries:
                time.sleep(sleep_seconds)

    raise SystemExit(f"download failed after {retries} attempts: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(DATASETS), default="OfflineBallRun-v0")
    parser.add_argument("--url", default=None)
    parser.add_argument("--output-dir", default=os.environ.get("DSRL_DATASET_DIR", "datasets/dsrl"))
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=int, default=20)
    args = parser.parse_args()

    url = args.url or DATASETS[args.task]
    download(url, Path(args.output_dir), args.retries, args.sleep_seconds)


if __name__ == "__main__":
    main()
