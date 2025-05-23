import base64
import json
from enum import Enum
from typing import Any, Optional, Dict, Union

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
