"""Functions a planner writes at runtime which are compiled, checked against
the action and condition contracts, and described as tools to be kept in memory."""

import ast
import builtins
import dis
import inspect
import os
import re
import time
import typing
from typing import Any, Callable, Dict, List, Optional

from attrs import define, field
from ..ros import Action, Event

# What the planner can write
KINDS = ("action", "condition")

# JSON schema types of the parameter annotations a tool can describe
_JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# The keyword arguments an action factory passes on to the Action it builds,
# the way a plugin's factory does
_ACTION_POLICY = set(inspect.signature(Action.__init__).parameters) - {
    "self",
    "method",
    "args",
    "kwargs",
}

# How a condition factory is called by add_event
CONDITION_FACTORY_SIGNATURE = (
    "(check_rate: float = 1.0, on_change: bool = True, handle_once: bool = False)"
)


@define(kw_only=True)
class ScratchFunction:
    """A function the planner wrote

    :param name: Its name, which is the function's name in the source
    :param kind: 'action' or 'condition'
    :param source: The Python source, as compiled
    :param function: The function itself
    :param description: What it does, from its docstring
    :param schema: Its tool schema, for an action
    """

    name: str = field()
    kind: str = field()
    source: str = field()
    function: Callable = field()
    description: str = field(default="")
    schema: Optional[Dict] = field(default=None)

    @property
    def signature(self) -> str:
        """The call signature, as a tool listing shows it"""
        return str(inspect.signature(self.function))


class Scratchpad:
    """The functions a planner wrote, with what it takes to write one.

    :param run_action: Runs an action of the stack by reference, for the
        written functions to call: ``run_action('owner/name', **arguments)``
    :type run_action: Callable[..., Tuple[bool, str]]
    :param logger: The logger of the component holding the scratchpad
    :param log_once: The component's ``log_once(key, message, level)``, for
        what a written condition keeps doing wrong
    :type log_once: Callable[[str, str, str], None]
    """

    def __init__(self, run_action: Callable, logger: Any, log_once: Callable) -> None:
        self._functions: Dict[str, ScratchFunction] = {}
        self._run_action = run_action
        self.logger = logger
        self._log_once = log_once

    def __contains__(self, name: str) -> bool:
        """Whether a function of that name was written"""
        return name in self._functions

    def get(self, name: str) -> ScratchFunction:
        """A written function by name.

        :raises KeyError: If none has that name
        """
        if name not in self._functions:
            raise KeyError(
                f"No function named '{name}'. Written: "
                f"{', '.join(sorted(self._functions)) or 'none'}"
            )
        return self._functions[name]

    def list(self) -> List[ScratchFunction]:
        """Every written function, in the order written"""
        return list(self._functions.values())

    def remove(self, name: str) -> ScratchFunction:
        """Forget a written function.

        :raises KeyError: If none has that name
        """
        removed = self.get(name)
        del self._functions[name]
        return removed

    def write(self, name: str, kind: str, source: str) -> ScratchFunction:
        """Define a function from its source, after checking it keeps the
        contract of its kind. A function of the same name is replaced.

        :raises ValueError: Saying what is wrong with the source
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if not name.isidentifier():
            raise ValueError(f"'{name}' is not a valid function name")
        source = self._compiled(source)
        function = self._defined(name, source)
        parameters = self._parameters(function, kind)
        self._names_defined(function)
        description = (inspect.getdoc(function) or "").strip()
        if not description:
            raise ValueError("The function needs a docstring saying what it does")
        written = ScratchFunction(
            name=name,
            kind=kind,
            source=source,
            function=function,
            description=description.split("\n\n")[0].replace("\n", " "),
            schema=self._schema(name, description, parameters)
            if kind == "action"
            else None,
        )
        self._functions[name] = written
        self.logger.info(f"Written {kind} '{name}':\n{source}")
        return written

    # ---- Checks ---------------------------------------------------------

    @staticmethod
    def _compiled(source: str) -> str:
        """The source, once it parses. A source whose quotes were escaped a
        second time on the way through a tool call is unescaped, when that
        is what makes it parse.

        :raises ValueError: If it does not parse
        """
        try:
            ast.parse(source)
            return source
        except SyntaxError as first:
            if '\\"' not in source:
                raise ValueError(f"The source does not parse: {first}") from first
            repaired = source.replace('\\"', '"')
            try:
                ast.parse(repaired)
            except SyntaxError:
                raise ValueError(f"The source does not parse: {first}") from first
            return repaired

    def _defined(self, name: str, source: str) -> Callable:
        """The function the source defines under the name

        :raises ValueError: If running the source fails or defines no such
            function
        """
        namespace: Dict[str, Any] = {
            "__name__": f"scratch.{name}",
            "run_action": self._run_action,
            "os": os,
        }
        try:
            exec(compile(source, f"<scratch:{name}>", "exec"), namespace)
        except Exception as e:
            raise ValueError(
                f"Running the source failed: {type(e).__name__}: {e}"
            ) from e
        function = namespace.get(name)
        if not inspect.isfunction(function):
            raise ValueError(f"The source does not define a function named '{name}'")
        return function

    @staticmethod
    def _names_defined(function: Callable) -> None:
        """Check that every global name the function loads exists in its
        namespace or among the builtins.

        :raises ValueError: Naming the first undefined name
        """
        known = set(function.__globals__) | set(dir(builtins))

        def loaded(code) -> Optional[str]:
            """The first unknown global the code or its nested code loads"""
            for instruction in dis.get_instructions(code):
                if (
                    instruction.opname == "LOAD_GLOBAL"
                    and instruction.argval not in known
                ):
                    return instruction.argval
            for constant in code.co_consts:
                if inspect.iscode(constant) and (name := loaded(constant)):
                    return name
            return None

        if name := loaded(function.__code__):
            raise ValueError(
                f"The function uses '{name}', which is not defined. Import what it "
                "needs in the source, and run the robot's actions through "
                "run_action('<tool name>', **arguments)"
            )

    @staticmethod
    def _parameters(function: Callable, kind: str) -> List[inspect.Parameter]:
        """The function's parameters, once its signature fits its kind

        :raises ValueError: Naming what does not fit
        """
        try:
            signature = inspect.signature(function, eval_str=True)
        except Exception as e:
            raise ValueError(f"The annotations could not be read: {e}") from e
        parameters = list(signature.parameters.values())
        returns = signature.return_annotation
        if kind == "condition":
            # internal events dont take params and only return bool
            if parameters:
                raise ValueError("A condition takes no parameters")
            if returns is not bool:
                raise ValueError(
                    "A condition must declare that it returns bool. Define it as "
                    f"`def {function.__name__}() -> bool:`"
                )
            return []
        # actions only return tuples of (bool, str) and named typed parameters
        # that do not collide with _ACTION_POLICY
        if not (
            typing.get_origin(returns) in (tuple, typing.Tuple)
            and typing.get_args(returns) == (bool, str)
        ):
            raise ValueError(
                "An action must declare that it returns tuple[bool, str], whether "
                "it succeeded with a message. Define it as "
                f"`def {function.__name__}(<typed parameters>) -> tuple[bool, str]:`"
            )
        for parameter in parameters:
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                raise ValueError(
                    f"An action takes named parameters only, not *{parameter.name}"
                )
            if parameter.annotation is inspect.Parameter.empty:
                raise ValueError(
                    f"Parameter '{parameter.name}' needs a type annotation"
                )
            if parameter.name in _ACTION_POLICY:
                raise ValueError(
                    f"Parameter '{parameter.name}' is named like a setting of the "
                    "action running the function, which would take its value. "
                    f"Rename it; the taken names are {', '.join(sorted(_ACTION_POLICY))}"
                )
        return parameters

    # ---- Tool schema ----------------------------------------------------

    @staticmethod
    def _json_type(annotation: Any) -> Optional[str]:
        """The JSON schema type of an annotation, if it has one"""
        if typing.get_origin(annotation) is typing.Union:
            arguments = [a for a in typing.get_args(annotation) if a is not type(None)]
            annotation = arguments[0] if len(arguments) == 1 else annotation
        return _JSON_TYPES.get(typing.get_origin(annotation) or annotation)

    @staticmethod
    def _schema(name: str, docstring: str, parameters: List[inspect.Parameter]) -> Dict:
        """The tool schema of an action, from its signature and docstring. A
        docstring line of the form ``name: text`` describes a parameter."""
        described = {
            m.group(1): m.group(2).strip()
            for m in re.finditer(
                r"^\s*(\w+)\s*(?:\([^)]*\))?:\s*(.+)$", docstring, re.M
            )
        }
        properties: Dict[str, Dict] = {}
        required = []
        for parameter in parameters:
            prop: Dict[str, Any] = {}
            if json_type := Scratchpad._json_type(parameter.annotation):
                prop["type"] = json_type
            if parameter.name in described:
                prop["description"] = described[parameter.name]
            properties[parameter.name] = prop
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter.name)
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": docstring.split("\n\n")[0].replace("\n", " "),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
            "phase": "execution",
        }

    # ---- Factories, as the action registry runs them --------------------

    def action_factory(self, name: str) -> Callable[..., Action]:
        """A factory building the Action of a written function. The function's
        own arguments and monitoring policy come as keyword arguments."""
        written = self.get(name)

        def factory(**arguments) -> Action:
            """The Action of the written function with these arguments"""
            policy = {
                key: arguments.pop(key)
                for key in list(arguments)
                if key in _ACTION_POLICY
            }
            policy.setdefault("name", name)
            policy.setdefault("description", written.description)
            return Action(method=written.function, kwargs=arguments, **policy)

        factory.__name__ = name
        factory.__doc__ = written.description
        return factory

    def condition_factory(self, name: str) -> Callable[..., Event]:
        """A factory building the Event polled on a written condition. The
        condition counts as met only when the function returned True; anything
        else, or an exception, is logged once and counts as not met."""
        written = self.get(name)

        def factory(
            check_rate: float = 1.0, on_change: bool = True, handle_once: bool = False
        ) -> Event:
            """The Event polling the written condition at this rate"""
            period = 1.0 / check_rate

            def condition() -> bool:
                """The written condition, held to returning exactly True"""
                started = time.perf_counter()
                try:
                    result = written.function()
                except Exception as e:
                    self._log_once(
                        f"{name}:raised",
                        f"Condition '{name}' raised {type(e).__name__}: {e}. Counted as not met",
                        "error",
                    )
                    return False
                finally:
                    took = time.perf_counter() - started
                    if took > period:
                        self._log_once(
                            f"{name}:slow",
                            f"Condition '{name}' took {took:.2f}s, longer than its "
                            f"polling period of {period:.2f}s. It holds up its own "
                            "polling while it runs",
                        )
                if result is not True and result is not False:
                    self._log_once(
                        f"{name}:not_bool",
                        f"Condition '{name}' returned {result!r}, not a bool. Counted as not met",
                        "error",
                    )
                    return False
                return result

            condition.__name__ = name
            return Event(
                condition,
                check_rate=check_rate,
                on_change=on_change,
                handle_once=handle_once,
            )

        factory.__name__ = name
        factory.__doc__ = written.description
        return factory
