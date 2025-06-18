from typing import Optional, List, Union
import json

from ..clients.db_base import DBClient
from ..config import SemanticRouterConfig
from ..publisher import Publisher
from ..ros import String, Topic, Route
from ..utils import validate_func_args
from .component_base import Component


class SemanticRouter(Component):
    """A component that routes semantic information from input topics to output topics based on pre-defined routes. The Semantic Router takes in a list of input topics, a list of routes, an optional default route, and a configuration object. It uses the database client to store and retrieve routing information.

    :param inputs:
        A list of input text topics that this component will subscribe to.
    :type inputs: list[Topic]
    :param routes:
        A list of pre-defined routes that publish incoming input to the routed output topics.
    :type routes: list[Route]
    :param default_route:
        An optional route that specifies the default behavior when no specific route matches up to a threshold. If not provided, the component will use the first route in the list.
    :type default_route: Optional[Route]
    :param config:
        The configuration object for this Semantic Router component.
    :type config: SemanticRouterConfig
    :param db_client:
        A database client that is used to store and retrieve routing information.
    :type db_client: DBClient
    :param component_name:
        The name of this Semantic Router component (default: "router_component").
    :type component_name: str
    :param kwargs:
        Additional keyword arguments.

    Example usage:
    ```python
    input_text = Topic(name="text0", msg_type="String")
    goto_route = Route(
        routes_to=goto,  # where goto is an input topic to another component
        samples=[
            "Go to the door",
            "Go to the kitchen",
            "Get me a glass",
            "Fetch a ball",
            "Go to hallway",
            "Go over there",
        ],
    )
    mllm_route = Route(
        routes_to=mllm_input,  # where mllm_input is an input topic to another component
        samples=[
            "Are we indoors or outdoors",
            "What do you see?",
            "Whats in front of you?",
            "Where are we",
            "Do you see any people?",
            "How many things are infront of you?",
            "Is this room occupied?",
        ],
    )
    config = SemanticRouterConfig(router_name="my_router")
    db_client = HTTPDBClient(db=ChromaDB(host='localhost', port=8080))
    semantic_router = SemanticRouter(
        inputs=[input_text],
        routes=[route1, route2],
        default_route=None,
        config=config,
        db_client=db_client
        component_name = "router"
    )
    ```
    """

    @validate_func_args
    def __init__(
        self,
        *,
        inputs: List[Topic],
        routes: List[Route],
        config: SemanticRouterConfig,
        db_client: DBClient,
        default_route: Optional[Route] = None,
        component_name: str,
        **kwargs,
    ):
        self.config: SemanticRouterConfig = config
        self.allowed_inputs = {"Required": [String]}
        self.allowed_outputs = {"Required": [String]}
        self.db_client = db_client

        super().__init__(
            inputs,
            None,
            self.config,
            inputs,
            component_name,
            **kwargs,
        )

        # create routes
        self._routes(routes)

        if default_route:
            if default_route.routes_to.name not in self.routes_dict:
                raise TypeError("default_route must be one of the specified routes")
            self.default_route = self.config._default_route = default_route

    def custom_on_configure(self):
        self.get_logger().debug(f"Current Status: {self.health_status.value}")

        # configure the rest
        super().custom_on_configure()

        # initialize db client
        self.db_client.check_connection()
        self.db_client.initialize()

        # initialize routes
        self._initialize_routes()

    def deactivate(self):
        # deactivate db client
        self.db_client.check_connection()
        self.db_client.deinitialize()

    def _initialize_routes(self):
        """Create routes by saving route samples in the database."""
        self.get_logger().info("Initializing all routes")
        for idx, (name, route) in enumerate(self.routes_dict.items()):
            route_to_add = {
                "collection_name": self.config.router_name,
                "distance_func": self.config.distance_func,
                "documents": route.samples,
                "metadatas": [{"route_name": name} for _ in range(len(route.samples))],
                "ids": [f"{name}.{i}" for i in range(len(route.samples))],
            }
            # reset collection on the addition of first route if it exists
            if idx == 0:
                route_to_add["reset_collection"] = True

            self.db_client.add(route_to_add)

    def _execution_step(self, **kwargs):
        """Execution step for Semantic Router component.
        :param args:
        :param kwargs:
        """
        trigger = kwargs.get("topic")
        if not trigger:
            return

        self.get_logger().debug(f"Received trigger on {trigger.name}")
        trigger_query = self.trig_callbacks[trigger.name].get_output()
        # get route
        db_input = {
            "collection_name": self.config.router_name,
            "query": trigger_query,
            "n_results": 1,
        }
        result = self.db_client.query(db_input)

        # TODO: Add treatment of multiple results by using an averaging function
        if result:
            distance = result["output"]["distances"][0][0]
            # if default route is specified and distance is less than min
            # threshold, redirect to default route
            route = (
                self.default_route.routes_to.name
                if self.default_route and distance > self.config.maximum_distance
                else result["output"]["metadatas"][0][0]["route_name"]
            )

            self.publishers_dict[route].publish(trigger_query)
        else:
            self.health_status.set_failure()

    def _routes(self, routes: List[Route]):
        """
        Set component Routes (topics)
        """
        self.routes_dict = {route.routes_to.name: route for route in routes}
        route_topics: List[Topic] = [route.routes_to for route in routes]  # type: ignore
        self.validate_topics(route_topics, self.allowed_outputs, "Outputs")
        self.publishers_dict = {
            route_topic.name: Publisher(route_topic) for route_topic in route_topics
        }

    def _update_cmd_args_list(self):
        """
        Update launch command arguments
        """
        super()._update_cmd_args_list()

        self.launch_cmd_args = [
            "--routes",
            self._get_routes_json(),
        ]

        self.launch_cmd_args = [
            "--db_client",
            self._get_db_client_json(),
        ]

    def _get_routes_json(self) -> Union[str, bytes, bytearray]:
        """
        Serialize component routes to json

        :return: Serialized inputs
        :rtype:  str | bytes | bytearray
        """
        if not hasattr(self, "routes_dict"):
            return "[]"
        return json.dumps([route.to_json() for route in self.routes_dict.values()])

    def _get_db_client_json(self) -> Union[str, bytes, bytearray]:
        """
        Serialize component routes to json

        :return: Serialized inputs
        :rtype:  str | bytes | bytearray
        """
        if not self.db_client:
            return ""
        return json.dumps(self.db_client.serialize())
