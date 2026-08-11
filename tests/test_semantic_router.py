"""Tests for SemanticRouter component — requires rclpy."""

import pytest
from unittest.mock import MagicMock, PropertyMock, patch

from agents.config import SemanticRouterConfig, LLMConfig
from agents.ros import Topic, Route
from agents.components.llm import LLM
from agents.components.semantic_router import SemanticRouter, RouterMode
from agents.clients.model_base import ModelClient
from tests.conftest import mock_component_internals


@pytest.fixture
def routes():
    """Create sample routes."""
    return [
        Route(
            routes_to=Topic(name="nav", msg_type="String"),
            samples=["go to", "navigate to", "move to"],
        ),
        Route(
            routes_to=Topic(name="chat", msg_type="String"),
            samples=["hello", "how are you", "tell me a joke"],
        ),
    ]


class TestRouterConstruction:
    def test_vector_mode(self, rclpy_init, mock_db_client, routes):
        router = SemanticRouter(
            inputs=[Topic(name="in", msg_type="String")],
            routes=routes,
            db_client=mock_db_client,
            config=SemanticRouterConfig(router_name="test_router"),
            component_name="test_vector_router",
        )
        assert router.routing_mode == RouterMode.VECTOR

    def test_llm_mode_with_client(self, rclpy_init, routes):
        client = MagicMock(spec=ModelClient)
        type(client).supports_tool_calls = PropertyMock(return_value=True)
        type(client).inference_timeout = PropertyMock(return_value=30)
        client.inference.return_value = {"output": "test"}
        client.check_connection.return_value = None
        client.initialize.return_value = None
        client.deinitialize.return_value = None

        router = SemanticRouter(
            inputs=[Topic(name="in", msg_type="String")],
            routes=routes,
            model_client=client,
            component_name="test_llm_router",
        )
        assert router.routing_mode == RouterMode.LLM

    def test_llm_mode_with_local(self, rclpy_init, routes):
        router = SemanticRouter(
            inputs=[Topic(name="in", msg_type="String")],
            routes=routes,
            config=LLMConfig(enable_local_model=True),
            component_name="test_local_router",
        )
        assert router.routing_mode == RouterMode.LLM

    def test_no_client_no_db_no_local_raises(self, rclpy_init, routes):
        with pytest.raises(ValueError):
            SemanticRouter(
                inputs=[Topic(name="in", msg_type="String")],
                routes=routes,
                component_name="test_fail_router",
            )

    def test_agentic_local_mode_deploys_local_model(self, rclpy_init, routes):
        """Regression: agentic routing on the local LLM must deploy the model
        on configure (LLM.custom_on_configure only deploys for type(self) is
        LLM, so the router has to trigger its own deploy)."""
        router = SemanticRouter(
            inputs=[Topic(name="in", msg_type="String")],
            routes=routes,
            config=LLMConfig(enable_local_model=True),
            component_name="test_local_deploy_router",
        )
        mock_component_internals(router)
        router._deploy_local_model = MagicMock()
        with patch.object(LLM, "custom_on_configure"):
            router.custom_on_configure()
        router._deploy_local_model.assert_called_once()

    def test_agentic_client_mode_does_not_deploy_local_model(
        self, rclpy_init, routes, mock_model_client
    ):
        router = SemanticRouter(
            inputs=[Topic(name="in", msg_type="String")],
            routes=routes,
            model_client=mock_model_client,
            component_name="test_client_no_deploy_router",
        )
        mock_component_internals(router)
        router._deploy_local_model = MagicMock()
        with patch.object(LLM, "custom_on_configure"):
            router.custom_on_configure()
        router._deploy_local_model.assert_not_called()

    def test_no_tool_support_raises(self, rclpy_init, routes):
        client = MagicMock(spec=ModelClient)
        type(client).supports_tool_calls = PropertyMock(return_value=False)

        with pytest.raises(TypeError):
            SemanticRouter(
                inputs=[Topic(name="in", msg_type="String")],
                routes=routes,
                model_client=client,
                component_name="test_notool_router",
            )
