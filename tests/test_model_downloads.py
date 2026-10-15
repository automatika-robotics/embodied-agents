"""Model downloads are verified against the hash pinned in their URL."""

import hashlib
import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import agents.utils.utils as utils
from agents.utils.utils import download, load_model, load_model_archive


def _serve(monkeypatch, payload: bytes):
    """httpx.stream answering with `payload`, recording the URL it was asked"""
    response = MagicMock()
    response.headers = {"content-length": str(len(payload))}
    response.iter_bytes = lambda chunk_size: iter([
        payload[i : i + chunk_size] for i in range(0, len(payload), chunk_size)
    ])
    stream = MagicMock()
    stream.return_value.__enter__.return_value = response
    monkeypatch.setattr(utils.httpx, "stream", stream)
    return stream


def _pinned(url: str, payload: bytes) -> str:
    return f"{url}#sha256={hashlib.sha256(payload).hexdigest()}"


class TestDownload:
    PAYLOAD = b"weights" * 20_000  # spans several chunks

    def test_a_matching_pin_keeps_the_file(self, tmp_path, monkeypatch):
        stream = _serve(monkeypatch, self.PAYLOAD)
        target = tmp_path / "m.onnx"

        download(_pinned("https://host/m.onnx", self.PAYLOAD), target, "m")

        assert target.read_bytes() == self.PAYLOAD
        # the pin is for us, not for the server
        assert stream.call_args.args[1] == "https://host/m.onnx"

    def test_a_mismatch_removes_the_file_and_says_so(self, tmp_path, monkeypatch):
        _serve(monkeypatch, self.PAYLOAD)
        target = tmp_path / "m.onnx"

        with pytest.raises(ValueError, match="SHA-256"):
            download("https://host/m.onnx#sha256=" + "0" * 64, target, "m")

        assert not target.exists()

    def test_an_unpinned_url_is_fetched_as_is(self, tmp_path, monkeypatch):
        _serve(monkeypatch, self.PAYLOAD)
        target = tmp_path / "m.onnx"

        download("https://host/m.onnx", target, "m")

        assert target.read_bytes() == self.PAYLOAD


class TestTheModelLoaders:
    """Both loaders fetch through `download`, so both honor a pin"""

    def test_load_model_caches_a_verified_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("platformdirs.user_cache_dir", lambda name: str(tmp_path))
        payload = b"onnx model"
        _serve(monkeypatch, payload)

        path = load_model("vad", _pinned("https://host/silero_vad.onnx", payload))

        assert Path(path).read_bytes() == payload and path.endswith("vad.onnx")

    def test_load_model_archive_names_the_bundle_without_the_pin(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("platformdirs.user_cache_dir", lambda name: str(tmp_path))
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            info = tarfile.TarInfo("kws-bundle/tokens.txt")
            info.size = len(b"tokens")
            tar.addfile(info, io.BytesIO(b"tokens"))
        payload = buffer.getvalue()
        _serve(monkeypatch, payload)

        path = load_model_archive(
            "kws", _pinned("https://host/kws-bundle.tar.gz", payload)
        )

        assert Path(path).name == "kws-bundle"
        assert (Path(path) / "tokens.txt").read_bytes() == b"tokens"
