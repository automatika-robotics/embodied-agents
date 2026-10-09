"""Live tests of the model and vector DB clients against running servers.

The servers are not started by these tests. Each client's tests are skipped when
its server cannot be reached, and all of them are marked local_only, so CI
deselects them. Hosts, ports and models are set with environment variables:

- Ollama, for OllamaClient and for GenericHTTPClient against Ollama's
  OpenAI-compatible endpoint: OLLAMA_HOST (127.0.0.1), OLLAMA_PORT (11434),
  OLLAMA_MODEL (qwen2.5vl:latest, a vision model that does not think, so its
  reply is all content)
- RoboML (``roboml``), for the HTTP and WebSocket clients: ROBOML_HOST
  (127.0.0.1), ROBOML_PORT (8000)
- RoboML RESP (``roboml-resp``): ROBOML_RESP_PORT (6379)
- RoboML models: ROBOML_LLM (Qwen/Qwen3-0.6B), ROBOML_MLLM
  (Qwen/Qwen2.5-VL-3B-Instruct)
- Chroma (``chroma run --path <dir> --port 8001``): CHROMA_HOST (127.0.0.1),
  CHROMA_PORT (8001), with embeddings from Ollama's CHROMA_EMBEDDINGS
  (bge-large:latest)
- A decision model on llama.cpp's llama-server (``llama-server -m <model.gguf>
  --alias lev --port 8090``), for GenericHTTPClient's decisions: DECISION_HOST
  (127.0.0.1), DECISION_PORT (8090), DECISION_MODEL (lev)

RoboML's tests run first: on one GPU, a model Ollama has loaded stays in memory
for a few minutes after its last use, and can leave too little for RoboML's.
"""

import os
import queue
import threading
from pathlib import Path

import cv2
import pytest

from agents.clients import (
    ChromaClient,
    GenericHTTPClient,
    OllamaClient,
    RoboMLHTTPClient,
    RoboMLRESPClient,
    RoboMLWSClient,
)
from agents.models import (
    GenericDecisionModel,
    GenericLLM,
    OllamaModel,
    TransformersLLM,
    TransformersMLLM,
)
from agents.vectordbs import ChromaDB

pytestmark = pytest.mark.local_only

HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", 11434))
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:latest")
ROBOML_HOST = os.environ.get("ROBOML_HOST", "127.0.0.1")
ROBOML_PORT = int(os.environ.get("ROBOML_PORT", 8000))
ROBOML_RESP_PORT = int(os.environ.get("ROBOML_RESP_PORT", 6379))
ROBOML_LLM = os.environ.get("ROBOML_LLM", "Qwen/Qwen3-0.6B")
ROBOML_MLLM = os.environ.get("ROBOML_MLLM", "Qwen/Qwen2.5-VL-3B-Instruct")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8001))
CHROMA_EMBEDDINGS = os.environ.get("CHROMA_EMBEDDINGS", "bge-large:latest")
DECISION_HOST = os.environ.get("DECISION_HOST", "127.0.0.1")
DECISION_PORT = int(os.environ.get("DECISION_PORT", 8090))
DECISION_MODEL = os.environ.get("DECISION_MODEL", "lev")

IMAGE = Path(__file__).parents[1] / "agents" / "resources" / "test.jpeg"


def started(client):
    """Connect and initialize a client, skipping when its server is not up"""
    try:
        client.check_connection()
    except Exception as e:
        pytest.skip(f"{type(client).__name__}: server not reachable ({e})")
    client.initialize()
    return client


def chat(text: str, stream: bool = False, **params) -> dict:
    # Room for a thinking model to finish thinking before it answers
    return {
        "query": [{"role": "user", "content": f"/no_think {text}"}],
        "max_new_tokens": 512,
        "temperature": 0.1,
        "stream": stream,
        **params,
    }


def initialized_again(client):
    """Deinitialize a client and initialize it again, as its component does when
    it reconfigures it or switches back to it"""
    client.deinitialize()
    client.initialize()
    return client


def text_of(chunk) -> str:
    """Text of a streamed chunk, from Ollama or from an OpenAI-compatible server"""
    if "message" in chunk:
        return chunk["message"].get("content") or ""
    return chunk["choices"][0]["delta"].get("content") or ""


@pytest.fixture(scope="class")
def ollama():
    client = started(OllamaClient(OllamaModel(name="llm", checkpoint=OLLAMA_MODEL)))
    yield client
    client.deinitialize()


@pytest.fixture(scope="class")
def generic():
    model = GenericLLM(name="llm", checkpoint=OLLAMA_MODEL)
    client = started(GenericHTTPClient(model, host=HOST, port=OLLAMA_PORT))
    yield client
    client.deinitialize()


@pytest.fixture(scope="class")
def decision():
    model = GenericDecisionModel(name="decision", checkpoint=DECISION_MODEL)
    client = started(GenericHTTPClient(model, host=DECISION_HOST, port=DECISION_PORT))
    yield client
    client.deinitialize()


@pytest.fixture(scope="class")
def roboml_http():
    model = TransformersLLM(name="agents_test_llm_http", checkpoint=ROBOML_LLM)
    client = started(RoboMLHTTPClient(model, host=ROBOML_HOST, port=ROBOML_PORT))
    yield client
    client.deinitialize()


@pytest.fixture(scope="class")
def roboml_http_mllm():
    model = TransformersMLLM(name="agents_test_mllm_http", checkpoint=ROBOML_MLLM)
    client = started(RoboMLHTTPClient(model, host=ROBOML_HOST, port=ROBOML_PORT))
    yield client
    client.deinitialize()


@pytest.fixture(scope="class")
def roboml_ws():
    model = TransformersLLM(name="agents_test_llm_ws", checkpoint=ROBOML_LLM)
    client = started(RoboMLWSClient(model, host=ROBOML_HOST, port=ROBOML_PORT))
    # Run the websocket loop on its own thread, fed through its queues, as the
    # component does
    client.stop_event = threading.Event()
    client.request_queue = queue.Queue()
    client.response_queue = queue.Queue()
    thread = threading.Thread(target=client._inference, daemon=True)
    thread.start()
    yield client
    client.stop_event.set()
    thread.join(timeout=10)
    client.deinitialize()


@pytest.fixture(scope="class")
def roboml_resp():
    model = TransformersLLM(name="agents_test_llm_resp", checkpoint=ROBOML_LLM)
    client = started(RoboMLRESPClient(model, host=ROBOML_HOST, port=ROBOML_RESP_PORT))
    yield client
    client.deinitialize()


@pytest.fixture(scope="class")
def chroma():
    db = ChromaDB(
        embeddings="ollama",
        checkpoint=CHROMA_EMBEDDINGS,
        ollama_host=HOST,
        ollama_port=OLLAMA_PORT,
    )
    client = ChromaClient(db, host=CHROMA_HOST, port=CHROMA_PORT)
    try:
        client.check_connection()
    except Exception as e:
        pytest.skip(f"ChromaClient: server not reachable ({e})")
    client.initialize()
    yield client
    client.deinitialize()


class TestRoboMLHTTPClient:
    def test_inference(self, roboml_http):
        result = roboml_http.inference(chat("Say hello in five words."))
        assert result and result["output"].strip()

    def test_streamed_inference(self, roboml_http):
        result = roboml_http.inference(chat("Say hello in five words.", stream=True))
        assert "".join(result["output"]).strip()

    def test_inference_after_initializing_again(self, roboml_http):
        result = initialized_again(roboml_http).inference(
            chat("Say hello in five words.")
        )
        assert result and result["output"].strip()

    def test_inference_with_an_image(self, roboml_http_mllm):
        image = cv2.cvtColor(cv2.imread(str(IMAGE)), cv2.COLOR_BGR2RGB)
        result = roboml_http_mllm.inference(
            chat("What do you see in this image?", images=[image])
        )
        assert result and result["output"].strip()


class TestRoboMLWSClient:
    def test_inference(self, roboml_ws):
        roboml_ws.request_queue.put(chat("Say hello in five words."))
        output = roboml_ws.response_queue.get(timeout=120)
        assert output.strip()


class TestRoboMLRESPClient:
    def test_inference(self, roboml_resp):
        result = roboml_resp.inference(chat("Say hello in five words."))
        assert result and result["output"].strip()

    def test_inference_after_initializing_again(self, roboml_resp):
        result = initialized_again(roboml_resp).inference(
            chat("Say hello in five words.")
        )
        assert result and result["output"].strip()


class TestChromaClient:
    COLLECTION = "agents_client_test"
    DOCS = {
        "ids": ["kitchen", "garden"],
        "metadatas": [{"room": "kitchen"}, {"room": "garden"}],
        "documents": ["a red mug on the kitchen table", "a tree in the garden"],
    }

    def test_add_then_query(self, chroma):
        added = chroma.add({
            **self.DOCS,
            "collection_name": self.COLLECTION,
            "reset_collection": True,
        })
        assert added

        result = chroma.query({
            "query": "where is the mug?",
            "collection_name": self.COLLECTION,
            "n_results": 1,
        })
        assert result["output"]["ids"][0] == ["kitchen"]

    def test_metadata_query(self, chroma):
        result = chroma.metadata_query({
            "metadatas": [{"room": "garden"}],
            "collection_name": self.COLLECTION,
        })
        assert result and result["output"]

    def test_query_after_deinitialize(self, chroma):
        """As after its component restarts"""
        chroma.deinitialize()

        result = chroma.query({
            "query": "where is the mug?",
            "collection_name": self.COLLECTION,
            "n_results": 1,
        })
        assert result["output"]["ids"][0] == ["kitchen"]


class TestOllamaClient:
    def test_inference(self, ollama):
        result = ollama.inference(chat("Say hello in five words."))
        assert result and result["output"].strip()

    def test_streamed_inference(self, ollama):
        result = ollama.inference(chat("Say hello in five words.", stream=True))
        assert "".join(text_of(chunk) for chunk in result["output"]).strip()

    def test_inference_with_an_image(self, ollama):
        image = cv2.cvtColor(cv2.imread(str(IMAGE)), cv2.COLOR_BGR2RGB)
        result = ollama.inference(
            chat("What do you see in this image?", images=[image])
        )
        assert result and result["output"].strip()

    def test_inference_after_initializing_again(self, ollama):
        result = initialized_again(ollama).inference(chat("Say hello in five words."))
        assert result and result["output"].strip()


class TestGenericHTTPClient:
    def test_inference(self, generic):
        result = generic.inference(chat("Say hello in five words."))
        assert result and result["output"].strip()

    def test_streamed_inference(self, generic):
        result = generic.inference(chat("Say hello in five words.", stream=True))
        assert "".join(text_of(chunk) for chunk in result["output"]).strip()

    def test_inference_after_initializing_again(self, generic):
        result = initialized_again(generic).inference(chat("Say hello in five words."))
        assert result and result["output"].strip()


class TestGenericHTTPClientDecisions:
    QUESTIONS = {
        "stop": {
            "type": "noul",
            "instructions": "Is the person telling the robot to stop?",
        },
        "route": {
            "type": "choice",
            "instructions": "Which route should handle this input?",
            "criteria": {
                "goto": "go somewhere, or fetch and bring an object",
                "chat": "a general question or chit-chat",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["can wait", "soon", "right now"],
        },
    }

    def test_inference(self, decision):
        result = decision.inference({
            "state": "Speech heard by the robot: 'Stop right now!'",
            "questions": self.QUESTIONS,
        })
        answers = result["output"]
        assert set(answers) == set(self.QUESTIONS)
        assert answers["stop"]["noul"] > 0.5
        assert answers["route"]["choice"] in ("goto", "chat")
        assert sum(answers["route"]["probabilities"].values()) == pytest.approx(1.0)
        assert 0.0 <= answers["urgency"]["score"] <= 2.0

    def test_inference_after_initializing_again(self, decision):
        result = initialized_again(decision).inference({
            "state": {"speech": "Please continue"},
            "questions": {"stop": self.QUESTIONS["stop"]},
        })
        assert result["output"]["stop"]["noul"] < 0.5
