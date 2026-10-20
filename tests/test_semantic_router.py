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
        type(client).supports_decisions = PropertyMock(return_value=False)
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
        type(client).supports_decisions = PropertyMock(return_value=False)

        with pytest.raises(TypeError):
            SemanticRouter(
                inputs=[Topic(name="in", msg_type="String")],
                routes=routes,
                model_client=client,
                component_name="test_notool_router",
            )


def decision_answer(choice, confidence):
    return {
        "output": {
            "route": {
                "type": "choice",
                "choice": choice,
                "probabilities": {"nav": 0.5, "chat": 0.5},
                "confidence": confidence,
            }
        }
    }


@pytest.fixture
def decision_client():
    """A mock client serving a decision model"""
    client = MagicMock(spec=ModelClient)
    type(client).supports_decisions = PropertyMock(return_value=True)
    type(client).supports_tool_calls = PropertyMock(return_value=False)
    type(client).inference_timeout = PropertyMock(return_value=30)
    client.inference.return_value = decision_answer("nav", 0.9)
    return client


def decision_router(client, routes, default_route=None, config=None, name="router"):
    """A decision mode router, set up as on configure, with a mocked publisher
    per route and the payload of a trigger"""
    router = SemanticRouter(
        inputs=[Topic(name="in", msg_type="String")],
        routes=routes,
        default_route=default_route,
        model_client=client,
        config=config,
        component_name=name,
    )
    mock_component_internals(router)
    router.publishers_dict = {"nav": MagicMock(), "chat": MagicMock()}
    router._setup_decision_routes(router.routes_dict)
    router._current_payload = "Head over to the living room"
    return router


class TestDecisionMode:
    def test_a_decision_model_routes_in_decision_mode(
        self, rclpy_init, routes, decision_client, mock_db_client
    ):
        router = SemanticRouter(
            inputs=[Topic(name="in", msg_type="String")],
            routes=routes,
            model_client=decision_client,
            db_client=mock_db_client,
            component_name="test_decision_router",
        )
        assert router.routing_mode == RouterMode.DECISION
        assert router.model_client is decision_client and router.db_client is None
        assert router._internal_config.minimum_confidence == 0.3

    def test_the_routes_are_the_options_of_one_choice(
        self, rclpy_init, routes, decision_client
    ):
        router = decision_router(decision_client, routes)

        router._decision_mode_execution_step()

        sent = decision_client.inference.call_args.args[0]
        assert sent["state"] == "Head over to the living room"
        question = sent["questions"]["route"]
        assert question["type"] == "choice"
        assert question["criteria"] == {
            "nav": "Use this for intents like: 'go to', 'navigate to', 'move to'",
            "chat": "Use this for intents like: 'hello', 'how are you', 'tell me a joke'",
        }

    def test_the_payload_goes_to_the_chosen_route(
        self, rclpy_init, routes, decision_client
    ):
        router = decision_router(decision_client, routes)

        router._decision_mode_execution_step()

        router.publishers_dict["nav"].publish.assert_called_once_with(
            "Head over to the living room"
        )
        router.publishers_dict["chat"].publish.assert_not_called()

    def test_an_unsure_choice_goes_to_the_default_route(
        self, rclpy_init, routes, decision_client
    ):
        decision_client.inference.return_value = decision_answer("nav", 0.1)
        router = decision_router(decision_client, routes, default_route=routes[1])

        router._decision_mode_execution_step()

        router.publishers_dict["chat"].publish.assert_called_once()
        router.publishers_dict["nav"].publish.assert_not_called()

    def test_the_confidence_needed_is_set_in_the_config(
        self, rclpy_init, routes, decision_client
    ):
        decision_client.inference.return_value = decision_answer("nav", 0.1)
        config = SemanticRouterConfig(router_name="r", minimum_confidence=0.05)
        router = decision_router(
            decision_client, routes, default_route=routes[1], config=config
        )

        router._decision_mode_execution_step()

        router.publishers_dict["nav"].publish.assert_called_once()

    def test_without_a_default_route_the_choice_is_taken(
        self, rclpy_init, routes, decision_client
    ):
        decision_client.inference.return_value = decision_answer("nav", 0.1)
        router = decision_router(decision_client, routes)

        router._decision_mode_execution_step()

        router.publishers_dict["nav"].publish.assert_called_once()

    def test_a_failed_inference_uses_the_default_route_or_fails(
        self, rclpy_init, routes, decision_client
    ):
        decision_client.inference.return_value = None
        router = decision_router(decision_client, routes, default_route=routes[1])
        router._decision_mode_execution_step()
        router.publishers_dict["chat"].publish.assert_called_once()

        router = decision_router(decision_client, routes, name="router_no_default")
        router._decision_mode_execution_step()
        router.publishers_dict["nav"].publish.assert_not_called()
        router.health_status.set_fail_algorithm.assert_called()

    def test_warmup_asks_the_route_question(self, rclpy_init, routes, decision_client):
        config = SemanticRouterConfig(router_name="r", warmup=True)
        router = decision_router(decision_client, routes, config=config)
        with patch.object(LLM, "custom_on_configure"):
            router.custom_on_configure()
        sent = decision_client.inference.call_args.args[0]
        assert "route" in sent["questions"]
