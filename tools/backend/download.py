#!/usr/bin/env python3
"""Download and verify the exact EasyFHE wheel used by the Couplet experiments."""
from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
WHEEL_NAME = "easyfhe-0.1.1+cu128.git949c3bd.u2204-cp312-cp312-linux_x86_64.whl"
WHEEL_SHA256 = "c0bc34862cdbceb614556ac196849e5844e492f938448d9fdd8d8549a3bc3653"
WHEEL_BYTES = 56863015
RELEASE_URL = "https://github.com/Azathoth13/couplet/releases/download/v0.2.0/"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_download(url: str, destination: Path, expected_sha: str, expected_size: int) -> Path:
    if destination.exists():
        if destination.stat().st_size == expected_size and file_digest(destination) == expected_sha:
            return destination
        raise ValueError(f"Existing file does not match the pinned wheel: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=60) as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    output.write(block)
        if temporary.stat().st_size != expected_size or file_digest(temporary) != expected_sha:
            raise ValueError("Downloaded wheel failed its pinned byte-count / SHA-256 verification")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest-dir", type=Path, default=ROOT / "tools/third_party/wheels")
    parser.add_argument("--url", default=RELEASE_URL + quote(WHEEL_NAME),
                        help="optional mirror URL; the pinned size and SHA-256 are still required")
    args = parser.parse_args()
    try:
        path = verified_download(args.url, args.dest_dir.expanduser().resolve() / WHEEL_NAME,
                                 WHEEL_SHA256, WHEEL_BYTES)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"EasyFHE download failed: {exc}\n")
    print(f"Verified {path}\nSHA-256: {WHEEL_SHA256}")
    print(f'export EASYFHE_WHEEL="{path}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
