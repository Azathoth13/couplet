# Frozen EasyFHE Release assets

The companion GitHub Release distributes the exact EasyFHE backend used by
the published Blackwell measurements, plus corresponding source and build
instructions. `backend_manifest.json` records the file names, hashes, wheel
metadata, native ABI inspection and complete source dependency pins.

Release attachments:

- `easyfhe-0.1.1+cu128.git949c3bd.u2204-cp312-cp312-linux_x86_64.whl`;
- `easyfhe-949c3bd-source.tar.gz` (EasyFHE, pinned submodule/NVTX sources,
  notices, and the compatibility build recipe);
- `BACKEND-MANIFEST.json`;
- `SHA256SUMS`.

See [the portable recipe](../../backend/build/README.md). Downloaded wheels
and source archives are Release attachments, not Git blobs. No new native
build or GPU experiment was performed to publish this preserved backend.
