"""The release wheel is installed only after content verification."""
import hashlib

import pytest

from tools.backend.download import verified_download


def test_download_verifies_content_and_reuses_only_matching_file(tmp_path):
    payload = b"release wheel fixture"
    source = tmp_path / "source.whl"
    source.write_bytes(payload)
    destination = tmp_path / "cache/backend.whl"
    digest = hashlib.sha256(payload).hexdigest()
    verified_download(source.as_uri(), destination, digest, len(payload))
    assert destination.read_bytes() == payload
    source.unlink()
    assert verified_download(source.as_uri(), destination, digest, len(payload)) == destination


@pytest.mark.parametrize("size,digest", [(1, "0" * 64), (7, "0" * 64)])
def test_bad_download_is_not_published(tmp_path, size, digest):
    source = tmp_path / "source.whl"
    source.write_bytes(b"payload")
    destination = tmp_path / "cache/backend.whl"
    with pytest.raises(ValueError):
        verified_download(source.as_uri(), destination, digest, size)
    assert not destination.exists()
    assert not list(destination.parent.glob("*.part"))
