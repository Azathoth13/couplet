#!/usr/bin/env python3
"""Download a pinned, official Llama base checkpoint and verify its weight blobs.

Uses HF_TOKEN / the Hugging Face login cache. Credentials are never written to
the download manifest. A historical public mirror is an explicit alternative;
its tensors must have the pinned official SHA-256 values too.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("1b", "3b", "8b"), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", choices=("official", "frozen-mirror"), default="official")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    profile = json.loads((ROOT / "configs/models" / f"{args.model}.json").read_text())
    checkpoint = profile["checkpoint"]
    prefix = "official" if args.source == "official" else "frozen_mirror"
    repository, revision = checkpoint[f"{prefix}_repo"], checkpoint[f"{prefix}_revision"]
    output = (args.output or args.data_root / profile["paths"]["raw"]).expanduser().resolve()
    description = {"model": args.model, "repo_id": repository, "revision": revision,
                   "output": str(output), "verify_against": checkpoint["official_repo"],
                   "official_revision": checkpoint["official_revision"]}
    print(json.dumps(description, indent=2), flush=True)
    if args.dry_run:
        return 0

    from huggingface_hub import HfApi, snapshot_download

    if args.source == "frozen-mirror":
        info = HfApi().model_info(repository, revision=revision, files_metadata=True)
        blobs = {item.rfilename: {"sha256": item.lfs.sha256, "bytes": item.size}
                 for item in info.siblings if item.rfilename.endswith(".safetensors") and item.lfs}
        if blobs != checkpoint["official_weight_blobs"]:
            raise ValueError("historical mirror blobs differ from this official revision; use --source official")

    snapshot_download(repo_id=repository, revision=revision, local_dir=str(output),
                      max_workers=args.workers,
                      allow_patterns=["*.json", "*.safetensors", "*.md", "LICENSE*", "USE_POLICY*"])
    checked = {}
    for name, expected in checkpoint["official_weight_blobs"].items():
        path = output / name
        actual = {"sha256": sha256(path), "bytes": path.stat().st_size}
        if actual != expected:
            raise ValueError(f"{name} differs from the pinned official weight blob")
        checked[name] = actual
    config = json.loads((output / "config.json").read_text())
    expected_geometry = (profile["geometry"]["native_hidden"], profile["num_layers"],
                         profile["geometry"]["native_query_heads"], profile["geometry"]["kv_heads"])
    actual_geometry = tuple(int(config[key]) for key in
                            ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads"))
    if actual_geometry != expected_geometry:
        raise ValueError(f"checkpoint geometry {actual_geometry} != {expected_geometry}")
    description.update({"files": checked, "config_sha256": sha256(output / "config.json"),
                        "weight_blobs_match_pinned_official": True})
    (output / "download_manifest.json").write_text(json.dumps(description, indent=2) + "\n")
    print(json.dumps({"verified_weight_files": len(checked), "manifest": str(output / "download_manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
