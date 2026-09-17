"""What clients say when they would send data unencrypted off this machine."""

import ssl
import sys
from unittest.mock import MagicMock

import httpx
import ollama
import pytest

from agents.clients.chroma import ChromaClient
from agents.clients.generic import GenericHTTPClient
from agents.clients.ollama import OllamaClient
from agents.clients.roboml import RoboMLHTTPClient, RoboMLWSClient
from agents.utils import plain_text_warning, tls_verify


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


class TestTheGenericClientsKey:
    """The generic client also sends an API key with every request, so a
    plain-text connection off this machine exposes the key as well"""

    MODEL = TestClientsWarnWhenStarted.MODEL

    def _warnings(self, monkeypatch, host, api_key):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("LLAMA_KEY", api_key or "")
        logger = MagicMock()
        monkeypatch.setattr(
            "agents.clients.model_base.logging.get_logger", lambda name: logger
        )
        client = GenericHTTPClient(
            self.MODEL, host=host, api_key_env="LLAMA_KEY" if api_key else None
        )
        # the key check sits in _initialize, so run that without a server
        client._validate_model_availability = lambda: None
        client.initialize()
        return [call.args[0] for call in logger.warning.call_args_list]

    def test_a_key_over_plain_text_to_the_lan_is_called_out(
        self, rclpy_init, monkeypatch
    ):
        warnings = self._warnings(monkeypatch, "10.0.0.5", api_key="sk-test")

        assert len(warnings) == 2
        assert "API key" in warnings[1]

    def test_without_a_key_only_the_connection_is_warned_about(
        self, rclpy_init, monkeypatch
    ):
        warnings = self._warnings(monkeypatch, "10.0.0.5", api_key=None)

        assert len(warnings) == 1 and "API key" not in warnings[0]

    @pytest.mark.parametrize("host", ["127.0.0.1", "https://api.openai.com/v1"])
    def test_a_key_stays_quiet_on_loopback_and_over_tls(
        self, rclpy_init, monkeypatch, host
    ):
        assert self._warnings(monkeypatch, host, api_key="sk-test") == []


class TestWhereTheKeyComesFrom:
    """A key is read from a named environment variable in the process that
    runs the component. Recipes and process arguments carry the variable's
    name, never the key"""

    MODEL = TestClientsWarnWhenStarted.MODEL

    def test_the_named_variable_is_the_key(self, monkeypatch):
        monkeypatch.setenv("LLAMA_KEY", "sk-test")

        client = GenericHTTPClient(self.MODEL, api_key_env="LLAMA_KEY")

        assert client.api_key == "sk-test"
        assert client.client.headers["Authorization"] == "Bearer sk-test"

    def test_the_default_is_the_openai_convention(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

        client = GenericHTTPClient(self.MODEL)

        assert client.client.headers["Authorization"] == "Bearer sk-openai"

    def test_the_default_may_be_unset(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        client = GenericHTTPClient(self.MODEL)

        assert client.api_key == "" and "Authorization" not in client.client.headers

    def test_a_named_variable_that_is_unset_is_an_error(self, monkeypatch):
        monkeypatch.delenv("LLAMA_KEY", raising=False)

        with pytest.raises(ValueError, match="LLAMA_KEY"):
            GenericHTTPClient(self.MODEL, api_key_env="LLAMA_KEY")

    def test_serialization_carries_the_name_and_survives_a_round_trip(
        self, monkeypatch
    ):
        monkeypatch.setenv("LLAMA_KEY", "sk-test")
        client = GenericHTTPClient(self.MODEL, host="10.0.0.5", api_key_env="LLAMA_KEY")

        serialized = client.serialize()
        rebuilt = GenericHTTPClient(**serialized)

        assert serialized["api_key_env"] == "LLAMA_KEY"
        assert "sk-test" not in str(serialized)
        assert rebuilt.api_key == "sk-test" and rebuilt.host == "10.0.0.5"


@pytest.fixture
def ca_cert(tmp_path):
    """A self-signed certificate, as a server serving its own would present"""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "gpu-box")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509
        .CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    path = tmp_path / "gpu-box.pem"
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(path)


class TestTrustingAServersOwnCertificate:
    """A server with a self-signed certificate is trusted through `ca_cert`,
    a PEM path, on every client that connects over TLS"""

    MODEL = TestClientsWarnWhenStarted.MODEL
    HOST = "https://gpu-box:8443"

    def test_the_system_store_is_the_default(self):
        assert tls_verify(None) is True

    def test_a_pem_file_becomes_a_context_trusting_it(self, ca_cert):
        context = tls_verify(ca_cert)

        assert isinstance(context, ssl.SSLContext)
        assert context.cert_store_stats()["x509"] == 1

    def _clients(self, monkeypatch):
        """One constructor per TLS-capable client, none reaching a server"""
        for cls in (RoboMLHTTPClient, ChromaClient, OllamaClient):
            monkeypatch.setattr(cls, "_check_connection", lambda self: None)
        # Chroma's local embeddings import a heavy package at construction
        monkeypatch.setitem(sys.modules, "sentence_transformers", MagicMock())
        ollama_model = {**self.MODEL, "model_type": "OllamaModel"}
        db = {
            "db_type": "ChromaDB",
            "init_timeout": 10,
            "db_init_params": {"embeddings": "default"},
        }
        return {
            "generic": lambda **kw: GenericHTTPClient(self.MODEL, **kw),
            "roboml": lambda **kw: RoboMLHTTPClient(self.MODEL, **kw),
            "chroma": lambda **kw: ChromaClient(db, **kw),
            "ollama": lambda **kw: OllamaClient(ollama_model, **kw),
        }

    @pytest.mark.parametrize("name", ["generic", "roboml", "chroma", "ollama"])
    def test_the_certificate_reaches_the_connection(self, monkeypatch, ca_cert, name):
        made = MagicMock()
        monkeypatch.setattr(httpx, "Client", made)
        monkeypatch.setattr(ollama, "Client", made)

        self._clients(monkeypatch)[name](host=self.HOST, ca_cert=ca_cert)

        assert isinstance(made.call_args.kwargs["verify"], ssl.SSLContext)

    @pytest.mark.parametrize("name", ["generic", "roboml", "chroma", "ollama"])
    def test_without_one_the_system_store_is_used(self, monkeypatch, name):
        made = MagicMock()
        monkeypatch.setattr(httpx, "Client", made)
        monkeypatch.setattr(ollama, "Client", made)

        self._clients(monkeypatch)[name](host=self.HOST)

        assert made.call_args.kwargs["verify"] is True

    def test_the_path_survives_serialization(self, monkeypatch, ca_cert):
        clients = self._clients(monkeypatch)

        model_client = clients["generic"](host=self.HOST, ca_cert=ca_cert)
        db_client = clients["chroma"](host=self.HOST, ca_cert=ca_cert)

        assert model_client.serialize()["ca_cert"] == ca_cert
        assert db_client.serialize()["ca_cert"] == ca_cert
        assert GenericHTTPClient(**model_client.serialize()).ca_cert == ca_cert


class TestTheWebSocketClientsUrl:
    """The RoboML WebSocket client sets its model up over HTTP and runs
    inference over a WebSocket to the same server, so one host must serve
    both, over TLS or not"""

    MODEL = TestClientsWarnWhenStarted.MODEL

    def _client(self, monkeypatch, **kwargs):
        monkeypatch.setattr(RoboMLHTTPClient, "_check_connection", lambda self: None)
        return RoboMLWSClient(self.MODEL, **kwargs)

    @pytest.mark.parametrize(
        "host, endpoint",
        [
            ("127.0.0.1", "ws://127.0.0.1:8000/m/ws_inference"),
            ("http://gpu-box:8000", "ws://gpu-box:8000/m/ws_inference"),
            ("https://gpu-box:8443", "wss://gpu-box:8443/m/ws_inference"),
        ],
    )
    def test_the_websocket_follows_the_http_scheme(self, monkeypatch, host, endpoint):
        client = self._client(monkeypatch, host=host)

        assert client.url.startswith("http")
        assert client.websocket_endpoint == endpoint

    def test_a_websocket_scheme_in_the_host_is_refused(self, monkeypatch):
        with pytest.raises(ValueError, match="http:// or https://"):
            self._client(monkeypatch, host="wss://gpu-box:8443")

    def test_a_certificate_applies_to_the_secure_websocket_only(
        self, monkeypatch, ca_cert
    ):
        secure = self._client(monkeypatch, host="https://gpu-box:8443", ca_cert=ca_cert)
        plain = self._client(monkeypatch, host="http://gpu-box:8000", ca_cert=ca_cert)

        assert isinstance(secure._ws_ssl, ssl.SSLContext)
        assert plain._ws_ssl is None
