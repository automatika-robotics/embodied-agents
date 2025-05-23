import time
import asyncio
import base64
import queue
import threading
from enum import Enum
from typing import Any, Optional, Dict, Union

import websockets
import httpx

from .. import models
from ..models import Model, OllamaModel, TransformersLLM, TransformersMLLM
from ..utils import encode_arr_base64
from ..vectordbs import DB
from .db_base import DBClient
from .model_base import ModelClient

__all__ = ["HTTPModelClient", "HTTPDBClient", "RESPDBClient", "RESPModelClient"]


class Status(str, Enum):
    """Model Node Status."""

    LOADED = "LOADED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    INITIALIZATION_ERROR = "INITIALIZATION_ERROR"


class RoboMLError(Exception):
    """RoboMLError."""

    pass


class HTTPModelClient(ModelClient):
    """An HTTP client for interaction with ML models served on RoboML"""

    def __init__(
        self,
        model: Union[Model, Dict],
        host: str = "127.0.0.1",
        port: int = 8000,
        inference_timeout: int = 30,
        init_on_activation: bool = True,
        logging_level: str = "info",
        **kwargs,
    ):
        if isinstance(model, OllamaModel):
            raise TypeError(
                "An ollama model cannot be passed to a RoboML client. Please use the OllamaClient"
            )
        super().__init__(
            model=model,
            host=host,
            port=port,
            inference_timeout=inference_timeout,
            init_on_activation=init_on_activation,
            logging_level=logging_level,
            **kwargs,
        )
        self.url = f"http://{self.host}:{self.port}"

        # create httpx client
        self.client = httpx.Client(base_url=self.url, timeout=self.inference_timeout)
        self._check_connection()

    def _check_connection(self) -> None:
        """Check if the platfrom is being served on specified IP and port"""
        # Ping remote server to check connection
        self.logger.info("Checking connection with remote RoboML")
        try:
            self.client.get("/").raise_for_status()
        except Exception as e:
            self.__handle_exceptions(e)
            raise

    def _initialize(self) -> None:
        """
        Initialize the model on platform using the paramters provided in the model specification class
        """
        # Create a model node on RoboML
        self.logger.info("Creating model node on remote")
        model_class = getattr(models, self.model_type)
        if issubclass(model_class, TransformersLLM):
            model_type = TransformersLLM.__name__
        elif issubclass(model_class, TransformersMLLM):
            model_type = TransformersMLLM.__name__
        else:
            model_type = self.model_type
        start_params = {"node_name": self.model_name, "node_model": model_type}
        try:
            r = self.client.post("/add_node", params=start_params).raise_for_status()
            self.logger.debug(str(r.json()))
            self.logger.info(f"Initializing {self.model_name} on RoboML remote")
            # get initialization params and initiale model
            self.client.post(
                f"/{self.model_name}/initialize",
                params=self.model_init_params,
                timeout=self.init_timeout,
            ).raise_for_status()
        except Exception as e:
            self.__handle_exceptions(e)
            raise
        self.logger.info(f"{self.model_name} initialized on remote")

    def _inference(self, inference_input: Dict[str, Any]) -> Optional[Dict]:
        """Call inference on the model using data and inference parameters from the component"""
        # encode any byte or numpy array data
        if inference_input.get("query") and isinstance(inference_input["query"], bytes):
            inference_input["query"] = base64.b64encode(
                inference_input["query"]
            ).decode("utf-8")
        if images := inference_input.get("images"):
            inference_input["images"] = [encode_arr_base64(img) for img in images]

        # if stream is set to true, then return a generator
        if inference_input.get("stream"):

            def gen():
                with self.client.stream(
                    method="POST",
                    url=f"/{self.model_name}/inference",
                    json=inference_input,
                    timeout=self.inference_timeout,
                ) as r:
                    try:
                        r.raise_for_status()
                    except Exception as e:
                        self.__handle_exceptions(e)

                    for token in r.iter_text():
                        self.logger.debug(f"{token}")
                        yield token

            return {"output": gen()}

        try:
            # call inference method
            r = self.client.post(
                f"/{self.model_name}/inference",
                json=inference_input,
                timeout=self.inference_timeout,
            ).raise_for_status()
            result = r.json()
        except Exception as e:
            return self.__handle_exceptions(e)

        self.logger.debug(str(result))

        return result

    def _deinitialize(self) -> None:
        """Deinitialize the model on the platform"""

        self.logger.info(f"Deinitializing {self.model_name} model on RoboML remote")
        stop_params = {"node_name": self.model_name}
        try:
            self.client.post("/remove_node", params=stop_params).raise_for_status()
        except Exception as e:
            self.__handle_exceptions(e)
            if hasattr(self, "client") and self.client and not self.client.is_closed:
                self.logger.info("Closing HTTPX client.")
                self.client.close()

    def __handle_exceptions(self, excep: Exception) -> None:
        """__handle_exceptions.

        :param excep:
        :type excep: Exception
        :rtype: None
        """
        if isinstance(excep, httpx.RequestError):
            self.logger.error(
                f"{excep} RoboML server inaccessible. Might not be running. Make sure remote is correctly configured."
            )
        elif isinstance(excep, httpx.TimeoutException):
            self.logger.error(
                f"{excep}. Request to RoboML server timed out. Make sure the server is configured correctly."
            )
        elif isinstance(excep, httpx.HTTPStatusError):
            try:
                excep_json = excep.response.json()
                self.logger.error(
                    f"RoboML server returned an invalid status code. Error: {excep_json}"
                )
            except Exception:
                self.logger.error(
                    f"RoboML server returned an invalid status code. Error: {excep}"
                )
        else:
            self.logger.error(str(excep))


class WebSocketClient(HTTPModelClient):
    """An websocket client for interaction with ML models served on RoboML"""

    def __init__(
        self,
        model: Union[Model, Dict],
        host: str = "127.0.0.1",
        port: int = 8000,
        inference_timeout: int = 30,
        init_on_activation: bool = True,
        logging_level: str = "info",
        **kwargs,
    ):
        if isinstance(model, OllamaModel):
            raise TypeError(
                "An ollama model cannot be passed to a RoboML client. Please use the OllamaClient"
            )
        try:
            import msgpack
            import msgpack_numpy as m_pack

            # patch msgpack for numpy arrays
            m_pack.patch()
            self.packer = msgpack.packb
            self.unpacker = msgpack.unpackb
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "In order to use the WebSocketClient, you need msgpack packages installed. You can install it with 'pip install msgpack msgpack-numpy'"
            ) from e
        super().__init__(
            model=model,
            host=host,
            port=port,
            inference_timeout=inference_timeout,
            init_on_activation=init_on_activation,
            logging_level=logging_level,
            **kwargs,
        )
        # Add queues and events
        self.stop_event: Optional[threading.Event] = None
        self.request_queue: Optional[queue.Queue] = None
        self.response_queue: Optional[queue.Queue] = None
        self.websocket_endpoint = (
            f"ws://{self.host}:{self.port}/{self.model_name}/ws_inference"
        )

    def _inference(self) -> Optional[Dict]:
        """Run the event loop for websocket client function. This function is executed in a separate thread to not block the main component"""
        # Each thread needs its own asyncio event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.__websocket_client())
        finally:
            self.logger.info("Closing asyncio event loop.")
            # Gracefully cancel all pending asyncio tasks in this loop
            for task in asyncio.all_tasks(loop):
                task.cancel()
            # Run loop one last time to allow tasks to process cancellation
            loop.run_until_complete(
                asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True)
            )
            loop.close()
        self.logger.info("WebSocket client thread finished.")

    async def __websocket_client(self):
        if not (self.stop_event and self.request_queue and self.response_queue):
            self.logger.error("WebSocketClient is not configured.")
            return

        try:
            async with websockets.connect(self.websocket_endpoint) as websocket:
                # Create concurrent tasks for sending and receiving
                receiver_task = asyncio.create_task(
                    self.__receive_messages(
                        websocket, self.response_queue, self.stop_event
                    )
                )
                sender_task = asyncio.create_task(
                    self.__send_requests(websocket, self.request_queue, self.stop_event)
                )
                while not self.stop_event.is_set():
                    if receiver_task.done() or sender_task.done():
                        self.stop_event.set()  # Ensure full shutdown if a task finishes unexpectedly
                        break
                    await asyncio.sleep(
                        0.1
                    )  # Keep alive, check stop_event periodically

            # Tasks should ideally respond to stop_event, but cancellation is a fallback
            if not sender_task.done():
                sender_task.cancel()
            if not receiver_task.done():
                receiver_task.cancel()
            # Wait for tasks to complete cancellation/exit
            await asyncio.gather(sender_task, receiver_task, return_exceptions=True)

        except websockets.exceptions.InvalidURI:
            self.logger.error(f"Invalid WebSocket URI: {self.websocket_endpoint}")
        except (
            websockets.exceptions.WebSocketException
        ) as e:  # Covers connection errors like gaierror
            self.logger.error(
                f"Failed to connect to WebSocket server {self.websocket_endpoint}: {e}"
            )
        except Exception as e:
            self.logger.error(f"An unexpected error occurred in client_logic: {e}")
        finally:
            self.logger.info("WebSocket client logic finished.")
            self.stop_event.set()  # Ensure main thread knows if client dies

    async def __receive_messages(self, websocket, res_queue, stop_evt):
        """Coroutine to continuously receive messages from the WebSocket."""
        self.logger.debug("Receiver task started.")
        try:
            async for message in websocket:  # Continuously iterates as messages arrive
                if stop_evt.is_set():
                    self.logger.info("Receiver: Stop event detected, exiting.")
                    break
                if not isinstance(message, str):
                    message = self.unpacker(message)
                self.logger.debug(f"Receiver: Received from server: '{message}'")
                res_queue.put(message)
        except websockets.exceptions.ConnectionClosedOK:
            self.logger.info(
                "WebSocketClient Receiver: WebSocket connection closed gracefully by server."
            )
        except websockets.exceptions.ConnectionClosedError as e:
            self.logger.error(
                f"WebSocketClient Receiver: WebSocket connection closed with error: {e}"
            )
        except Exception as e:
            # Avoid logging error if we are stopping and the error is due to connection closure
            if not stop_evt.is_set() and not isinstance(e, asyncio.CancelledError):
                self.logger.error(f"Receiver: Unexpected error: {e}")
        finally:
            self.logger.debug("Receiver task finished.")
            stop_evt.set()  # Ensure other parts of the client know to stop

    async def __send_requests(self, websocket, req_queue, stop_evt):
        """Coroutine to send requests from the request_queue."""
        self.logger.debug("Sender task started.")
        try:
            while not stop_evt.is_set():
                try:
                    # Get request from main thread with a short timeout
                    inference_input = req_queue.get(block=True, timeout=0.1)
                    # encode any byte or numpy array data
                    if inference_input.get("query") and isinstance(
                        inference_input["query"], bytes
                    ):
                        inference_input["query"] = base64.b64encode(
                            inference_input["query"]
                        ).decode("utf-8")
                    if images := inference_input.get("images"):
                        inference_input["images"] = [
                            encode_arr_base64(img) for img in images
                        ]
                    if websocket.closed:
                        self.logger.warning("Sender: WebSocket is closed, cannot send.")
                        req_queue.put(
                            inference_input
                        )  # Put back for potential reprocessing or logging
                        stop_evt.set()
                        break
                    await websocket.send(self.packer(inference_input))
                    req_queue.task_done()  # Signal that this request item has been processed
                except queue.Empty:
                    # No request, loop again to check stop_event or new requests
                    if stop_evt.is_set():
                        self.logger.info("Sender: Stop event detected, exiting.")
                        break
                    continue
                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning(
                        "Sender: WebSocket connection closed. Cannot send."
                    )
                    stop_evt.set()  # Signal other parts to stop
                    break
                except Exception as e:
                    if not stop_evt.is_set() and not isinstance(
                        e, asyncio.CancelledError
                    ):
                        self.logger.error(f"Sender: Error sending message: {e}")
                    if stop_evt.is_set():
                        break
                    time.sleep(0.1)  # avoid busy loop on continuous errors
        finally:
            self.logger.debug("Sender task finished.")


class HTTPDBClient(DBClient):
    """An HTTP client for interaction with vector DBs served on RoboML"""

    def __init__(
        self,
        db: Union[DB, Dict],
        host: str = "127.0.0.1",
        port: int = 8000,
        response_timeout: int = 30,
        init_on_activation: bool = True,
        logging_level: str = "info",
        **kwargs,
    ):
        super().__init__(
            db=db,
            host=host,
            port=port,
            response_timeout=response_timeout,
            init_on_activation=init_on_activation,
            logging_level=logging_level,
            **kwargs,
        )
        self.url = f"http://{self.host}:{self.port}"
        self._check_connection()

    def _check_connection(self):
        """Check if the platfrom is being served on specified IP and port"""
        # Ping remote server to check connection
        self.logger.info("Checking connection with remote RoboML")
        try:
            # port specific to ollama
            httpx.get(f"{self.url}/").raise_for_status()
        except Exception as e:
            self.__handle_exceptions(e)
            raise

    def _initialize(self) -> None:
        """
        Initialize the vector DB on platform using the paramters provided in the DB specification class
        """
        # Create a DB node on RoboML
        self.logger.info("Creating db node on remote")
        start_params = {"node_name": self.db_name, "node_model": self.db_type}
        try:
            r = httpx.post(
                f"{self.url}/add_node", params=start_params, timeout=self.init_timeout
            ).raise_for_status()
            self.logger.debug(str(r.json()))
            self.logger.info(f"Initializing {self.db_name} on RoboML remote")
            # get initialization params and initiale db
            httpx.post(
                f"{self.url}/{self.db_name}/initialize",
                params=self.db_init_params,
                timeout=self.init_timeout,
            ).raise_for_status()
        except Exception as e:
            self.__handle_exceptions(e)
            raise
        self.logger.info(f"{self.db_name} initialized on remote")

    def _add(self, db_input: Dict[str, Any]) -> Optional[Dict]:
        """Add data.
        :param db_input:
        :type db_input: dict[str, Any]
        :rtype: dict | None
        """
        try:
            # add to DB
            r = httpx.post(
                f"{self.url}/{self.db_name}/add",
                json=db_input,
                timeout=self.response_timeout,
            ).raise_for_status()
            result = r.json()
        except Exception as e:
            return self.__handle_exceptions(e)

        self.logger.debug(str(result))

        return result

    def _conditional_add(self, db_input: Dict[str, Any]) -> Optional[Dict]:
        """Add data only if the ids dont exist. Otherwise update metadatas
        :param db_input:
        :type db_input: dict[str, Any]
        :rtype: dict | None
        """
        try:
            # add to DB
            r = httpx.post(
                f"{self.url}/{self.db_name}/conditional_add",
                json=db_input,
                timeout=self.response_timeout,
            ).raise_for_status()
            result = r.json()
        except Exception as e:
            return self.__handle_exceptions(e)

        self.logger.debug(str(result))

        return result

    def _metadata_query(self, db_input: Dict[str, Any]) -> Optional[Dict]:
        """Query based on given metadata.
        :param db_input:
        :type db_input: dict[str, Any]
        :rtype: dict | None
        """
        try:
            # query db
            r = httpx.post(
                f"{self.url}/{self.db_name}/metadata_query",
                json=db_input,
                timeout=self.response_timeout,
            ).raise_for_status()
            result = r.json()
        except Exception as e:
            return self.__handle_exceptions(e)

        self.logger.debug(str(result))

        return result

    def _query(self, db_input: Dict[str, Any]) -> Optional[Dict]:
        """Query using a query string.
        :param db_input:
        :type db_input: dict[str, Any]
        :rtype: dict | None
        """
        try:
            # query db
            r = httpx.post(
                f"{self.url}/{self.db_name}/query",
                json=db_input,
                timeout=self.response_timeout,
            ).raise_for_status()
            result = r.json()
        except Exception as e:
            return self.__handle_exceptions(e)

        self.logger.debug(str(result))

        return result

    def _deinitialize(self) -> None:
        """Deinitialize DB on the platform"""

        self.logger.error(f"Deinitializing {self.db_name} on RoboML remote")
        stop_params = {"node_name": self.db_name}
        try:
            httpx.post(f"{self.url}/remove_node", params=stop_params).raise_for_status()
        except Exception as e:
            self.__handle_exceptions(e)

    def __handle_exceptions(self, excep: Exception) -> None:
        """__handle_exceptions.

        :param excep:
        :type excep: Exception
        :rtype: None
        """
        if isinstance(excep, httpx.RequestError):
            self.logger.error(
                f"{excep} RoboML server inaccessible. Might not be running. Make sure remote is correctly configured."
            )
        elif isinstance(excep, httpx.HTTPStatusError):
            try:
                excep_json = excep.response.json()
                self.logger.error(
                    f"RoboML server returned an invalid status code. Error: {excep_json}"
                )
            except Exception:
                self.logger.error(
                    f"RoboML server returned an invalid status code. Error: {excep}"
                )
        else:
            self.logger.error(str(excep))


class RESPModelClient(ModelClient):
    """A Redis Serialization Protocol (RESP) based client for interaction with ML models served on RoboML"""

    def __init__(
        self,
        model: Union[Model, Dict],
        host: str = "127.0.0.1",
        port: int = 6379,
        inference_timeout: int = 30,
        init_on_activation: bool = True,
        logging_level: str = "info",
        **kwargs,
    ):
        if isinstance(model, OllamaModel):
            raise TypeError(
                "An ollama model cannot be passed to a RoboML client. Please use the OllamaClient"
            )
        super().__init__(
            model=model,
            host=host,
            port=port,
            inference_timeout=inference_timeout,
            init_on_activation=init_on_activation,
            logging_level=logging_level,
            **kwargs,
        )
        try:
            import msgpack
            import msgpack_numpy as m_pack

            # patch msgpack for numpy arrays
            m_pack.patch()
            from redis import Redis

            # TODO: handle timeout
            self.redis = Redis(self.host, port=self.port)
            self.packer = msgpack.packb
            self.unpacker = msgpack.unpackb

        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "In order to use the RESP clients, you need redis and msgpack packages installed. You can install it with 'pip install redis[hiredis] msgpack msgpack-numpy'"
            ) from e
        self._check_connection()

    def _check_connection(self) -> None:
        """Check if the platfrom is being served on specified IP and port"""
        # Ping remote server to check connection
        self.logger.info("Checking connection with remote RoboML")
        try:
            self.redis.execute_command(b"PING")
        except Exception as e:
            self.__handle_exceptions(e)
            raise

    def _initialize(self) -> None:
        """
        Initialize the model on platform using the paramters provided in the model specification class
        """
        # Create a model node on RoboML
        self.logger.info("Creating model node on remote")
        self.model_class = getattr(models, self.model_type)
        if issubclass(self.model_class, TransformersLLM):
            model_type = TransformersLLM.__name__
        elif issubclass(self.model_class, TransformersMLLM):
            model_type = TransformersMLLM.__name__
        else:
            model_type = self.model_type
        start_params = {"node_name": self.model_name, "node_model": model_type}
        try:
            start_params_b = self.packer(start_params)
            node_init_result = self.redis.execute_command("add_node", start_params_b)
            if node_init_result:
                self.logger.debug(str(self.unpacker(node_init_result)))
            self.logger.info(f"Initializing {self.model_name} on RoboML remote")
            # make initialization params
            model_dict = self.model_init_params
            # initialize model
            init_b = self.packer(model_dict)
            self.redis.execute_command(f"{self.model_name}.initialize", init_b)
        except Exception as e:
            self.__handle_exceptions(e)
            raise

        # check status for init completion after every second
        status = self.__check_model_status()
        if status == Status.READY:
            self.logger.info(f"{self.model_name} model initialized on remote")
        elif status == Status.INITIALIZING:
            raise Exception(f"{self.model_name} model initialization timed out.")
        elif status == Status.INITIALIZATION_ERROR:
            raise Exception(
                f"{self.model_name} model initialization failed. Check remote for logs."
            )
        else:
            raise Exception(
                f"Unexpected Error while initializing {self.model_name}: Check remote for logs."
            )

    def _inference(self, inference_input: Dict[str, Any]) -> Optional[Dict]:
        """Call inference on the model using data and inference parameters from the component"""
        try:
            data_b = self.packer(inference_input)
            # call inference method
            result_b = self.redis.execute_command(
                f"{self.model_name}.inference", data_b
            )
            result = self.unpacker(result_b)
        except Exception as e:
            return self.__handle_exceptions(e)

        self.logger.debug(str(result))

        return result

    def _deinitialize(self) -> None:
        """Deinitialize the model on the platform"""

        self.logger.error(f"Deinitializing {self.model_name} on RoboML remote")
        stop_params = {"node_name": self.model_name}
        try:
            stop_params_b = self.packer(stop_params)
            self.redis.execute_command("remove_node", stop_params_b)
        except Exception as e:
            self.__handle_exceptions(e)

    def __check_model_status(self) -> Optional[str]:
        """Check remote model node status.
        :rtype: str | None
        """
        try:
            status_b = self.redis.execute_command(f"{self.model_name}.get_status")
            status = self.unpacker(status_b)
        except Exception as e:
            return self.__handle_exceptions(e)

        return status

    def __handle_exceptions(self, excep: Exception) -> None:
        """__handle_exceptions.

        :param excep:
        :type excep: Exception
        :rtype: None
        """
        from redis.exceptions import ConnectionError, ModuleError

        if isinstance(excep, ConnectionError):
            self.logger.error(
                f"{excep} RoboML server inaccessible. Might not be running. Make sure remote is correctly configured."
            )
        elif isinstance(excep, ModuleError):
            self.logger.error(
                f"{self.model_type} is not a supported model type in RoboML library. Please use another model client or another model."
            )
            raise RoboMLError(
                f"{self.model_type} is not a supported model type in RoboML library. Please use another model client or another model."
            )
        else:
            self.logger.error(str(excep))


class RESPDBClient(DBClient):
    """A Redis Serialization Protocol (RESP) based client for interaction with vector DBs served on RoboML"""

    def __init__(
        self,
        db: Union[DB, Dict],
        host: str = "127.0.0.1",
        port: int = 6379,
        init_on_activation: bool = True,
        logging_level: str = "info",
        **kwargs,
    ):
        super().__init__(
            db=db,
            host=host,
            port=port,
            init_on_activation=init_on_activation,
            logging_level=logging_level,
            **kwargs,
        )
        try:
            import msgpack
            import msgpack_numpy as m_pack

            # patch msgpack for numpy arrays
            m_pack.patch()
            from redis import Redis

            # TODO: handle timeout
            self.redis = Redis(self.host, port=self.port)
            self.packer = msgpack.packb
            self.unpacker = msgpack.unpackb

        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "In order to use the RESP clients, you need redis and msgpack packages installed. You can install it with 'pip install redis[hiredis] msgpack msgpack-numpy'"
            ) from e
        self._check_connection()

    def _check_connection(self) -> None:
        """Check if the platfrom is being served on specified IP and port"""
        # Ping remote server to check connection
        self.logger.info("Checking connection with remote RoboML")
        try:
            self.redis.execute_command(b"PING")
        except Exception as e:
            self.__handle_exceptions(e)
            raise

    def _initialize(self) -> None:
        """
        Initialize the vector DB on platform using the paramters provided in the DB specification class
        """
        # Creating DB node on remote
        self.logger.info("Creating db node on remote")
        start_params = {"node_name": self.db_name, "node_model": self.db_type}

        try:
            start_params_b = self.packer(start_params)
            node_init_result = self.redis.execute_command("add_node", start_params_b)
            if node_init_result:
                self.logger.debug(str(self.unpacker(node_init_result)))
            self.logger.info(f"Initializing {self.db_name} on remote")
            init_b = self.packer(self.db_init_params)
            # initialize database
            self.redis.execute_command(f"{self.db_name}.initialize", init_b)
        except Exception as e:
            self.__handle_exceptions(e)
            raise

        # check status for init completion after every second
        status = self.__check_db_status()
        if status == Status.READY:
            self.logger.info(f"{self.db_name} db initialized on remote")
        elif status == Status.INITIALIZING:
            raise Exception(f"{self.db_name} db initialization timed out.")
        elif status == Status.INITIALIZATION_ERROR:
            raise Exception(
                f"{self.db_name} db initialization failed. Check remote for logs."
            )
        else:
            raise Exception(
                f"Unexpected Error while initializing {self.db_name}: Check remote for logs."
            )

    def _add(self, db_input: Dict[str, Any]) -> Optional[Dict]:
        """Add data.
        :param db_input:
        :type db_input: dict[str, Any]
        :rtype: dict | None
        """
        try:
            data_b = self.packer(db_input)
            # add to DB
            result_b = self.redis.execute_command(f"{self.db_name}.add", data_b)
            result = self.unpacker(result_b)
        except Exception as e:
            return self.__handle_exceptions(e)

        self.logger.debug(str(result))

        return result

    def _conditional_add(self, db_input: Dict[str, Any]) -> Optional[Dict]:
        """Add data only if the ids dont exist. Otherwise update metadatas
        :param db_input:
        :type db_input: dict[str, Any]
        :rtype: dict | None
        """
        try:
            data_b = self.packer(db_input)
            # add to DB
            result_b = self.redis.execute_command(
                f"{self.db_name}.conditional_add", data_b
            )
            result = self.unpacker(result_b)
        except Exception as e:
            return self.__handle_exceptions(e)

        self.logger.debug(str(result))

        return result

    def _metadata_query(self, db_input: Dict[str, Any]) -> Optional[Dict]:
        """Query based on given metadata.
        :param db_input:
        :type db_input: dict[str, Any]
        :rtype: dict | None
        """
        try:
            data_b = self.packer(db_input)
            # query db
            result_b = self.redis.execute_command(
                f"{self.db_name}.metadata_query", data_b
            )
            result = self.unpacker(result_b)
        except Exception as e:
            return self.__handle_exceptions(e)

        self.logger.debug(str(result))

        return result

    def _query(self, db_input: Dict[str, Any]) -> Optional[Dict]:
        """Query using a query string.
        :param db_input:
        :type db_input: dict[str, Any]
        :rtype: dict | None
        """
        try:
            data_b = self.packer(db_input)
            # query db
            result_b = self.redis.execute_command(f"{self.db_name}.query", data_b)
            result = self.unpacker(result_b)
        except Exception as e:
            return self.__handle_exceptions(e)

        self.logger.debug(str(result))

        return result

    def _deinitialize(self) -> None:
        """Deinitialize DB on the platform"""

        self.logger.error(f"Deinitializing {self.db_name} on remote")
        stop_params = {"node_name": self.db_name}
        try:
            stop_params_b = self.packer(stop_params)
            self.redis.execute_command("remove_node", stop_params_b)
        except Exception as e:
            self.__handle_exceptions(e)

    def __check_db_status(self) -> Optional[str]:
        """Check remote db node status.
        :rtype: str | None
        """
        try:
            status_b = self.redis.execute_command(f"{self.db_name}.get_status")
            status = self.unpacker(status_b)
        except Exception as e:
            return self.__handle_exceptions(e)

        return status

    def __handle_exceptions(self, excep: Exception) -> None:
        """__handle_exceptions.

        :param excep:
        :type excep: Exception
        :rtype: None
        """
        from redis.exceptions import ConnectionError, ModuleError

        if isinstance(excep, ConnectionError):
            self.logger.error(
                f"{excep} RoboML server inaccessible. Might not be running. Make sure remote is correctly configured."
            )
        elif isinstance(excep, ModuleError):
            self.logger.error(
                f"{self.db_type} is not a supported vectordb type in RoboML library. Please use another database client or another database."
            )
            raise RoboMLError(
                f"{self.db_type} is not a supported vectordb type in RoboML library. Please use another database client or another database."
            )
        else:
            self.logger.error(str(excep))
