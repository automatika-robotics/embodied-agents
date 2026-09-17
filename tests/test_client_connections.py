"""What clients say when they would send data unencrypted off this machine."""

from unittest.mock import MagicMock

import pytest

from agents.clients.generic import GenericHTTPClient
from agents.utils import plain_text_warning


class TestPlainTextWarning:
    @pytest.mark.parametrize(
        "host",
        [
            None,
            "127.0.0.1",
            "127.0.0.2",
            "localhost",
            "[::1]",
            "http://127.0.0.1:8000",
            "ws://localhost:8000",
            "http://[::1]:8000",
            "https://gpu-box:8443",
            "wss://gpu-box/infer",
            "https://api.openai.com/v1",
        ],
    )
    def test_loopback_and_tls_hosts_are_quiet(self, host):
        assert plain_text_warning(host) is None

    @pytest.mark.parametrize(
        "host",
        [
            "10.0.0.5",
            "gpu-box",
            "gpu-box.local",
            "fe80::1",
            "http://10.0.0.5:8000",
            "ws://gpu-box:8000",
        ],
    )
    def test_plain_text_to_another_host_names_it(self, host):
        assert host in plain_text_warning(host)


class TestClientsWarnWhenStarted:
    """The check lives in the client base classes, so every client gets it. It
    runs when the client starts, which happens once, in the process that uses
    the connection; construction also happens in the recipe process"""

    MODEL = {
        "model_type": "GenericLLM",
        "model_name": "m",
        "init_timeout": 10,
        "model_init_params": {},
    }

    def _started(self, monkeypatch, host):
        logger = MagicMock()
        monkeypatch.setattr(
            "agents.clients.model_base.logging.get_logger", lambda name: logger
        )
        client = GenericHTTPClient(self.MODEL, host=host)
        assert not logger.warning.called, "construction alone must stay quiet"
        client.init_on_activation = False  # no server to reach in a test
        client.initialize()
        return logger

    def test_a_server_on_the_lan_gets_one_warning(self, rclpy_init, monkeypatch):
        logger = self._started(monkeypatch, "10.0.0.5")

        logger.warning.assert_called_once()
        assert "10.0.0.5" in logger.warning.call_args[0][0]

    def test_the_default_host_is_quiet(self, rclpy_init, monkeypatch):
        assert not self._started(monkeypatch, "127.0.0.1").warning.called

    def test_a_tls_endpoint_is_quiet(self, rclpy_init, monkeypatch):
        assert not self._started(monkeypatch, "https://gpu-box:8443").warning.called
