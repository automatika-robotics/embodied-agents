import base64
import inspect
import uuid
from functools import wraps
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import (
    List,
    Dict,
    Optional,
    Union,
    get_args,
    get_origin,
    _GenericAlias,
)
from collections.abc import Iterable

import cv2
import httpx
import numpy as np
from attrs import Attribute
from jinja2 import Environment, FileSystemLoader
from jinja2.environment import Template
from .pluralize import pluralize


def draw_detection_bounding_boxes(
    img: np.ndarray, bounding_boxes: List, labels: List, handle_bbox2d_msg: bool = True
) -> np.ndarray:
    """Draw bounding boxes and labels"""

    for i, bbox in enumerate(bounding_boxes):
        # Bounding box expected format: (x1, y1, x2, y2)
        if handle_bbox2d_msg:
            bbox = [
                bbox.top_left_x,
                bbox.top_left_y,
                bbox.bottom_right_x,
                bbox.bottom_right_y,
            ]  # reading from BBox2D msg
        x1, y1, x2, y2 = map(int, bbox)

        # Choose color based on label if available
        if labels and i < len(labels):
            label_text = str(labels[i])
            color_seed = abs(hash(label_text)) % 0xFFFFFF
            color = (
                color_seed & 255,
                (color_seed >> 8) & 255,
                (color_seed >> 16) & 255,
            )
        else:
            label_text = ""
            color = (0, 255, 0)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label text background and text
        if label_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img,
                label_text,
                (x1 + 2, y1 - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
    return img


def draw_points_2d(img: np.ndarray, points: np.ndarray, radius: int = 3) -> np.ndarray:
    """Draw 2D points on an image."""
    if points is None or len(points) == 0:
        return img

    # Ensure points is a np array of shape (N, 2)
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points of shape (N, 2), got {points.shape}")

    for x, y in points:
        cv2.circle(img, (int(x), int(y)), radius, (255, 0, 0), -1)  # red filled circle

    return img


def create_detection_context(obj_list: Optional[List]) -> str:
    """
    Creates a context prompt based on detections.
    :param      detections:  The detections
    :type       detections:  str
    :returns:   Context string
    :rtype:     str
    """
    if not obj_list:
        return ""
    context_list = []
    for obj_class in set(obj_list):
        obj_count = obj_list.count(obj_class)
        if obj_count > 1:
            context_list.append(f"{str(obj_count)} {pluralize(obj_class)}")
        else:
            context_list.append(f"{str(obj_count)} {obj_class}")

    if len(obj_list) > 1:
        return f"{', '.join(context_list)}"
    return f"{context_list[0]}"


def get_prompt_template(template: Union[str, Path]) -> Template:
    """Method to read prompt jinja prompt templates
    :param template:
    :type template: str | Path
    :rtype: None
    """
    # check if prompt is a filename
    try:
        path_exists = Path(template).exists()
    except OSError:
        path_exists = False
    if path_exists:
        try:
            env = Environment(
                loader=FileSystemLoader(Path(template).parent), autoescape=True
            )
            return env.get_template(Path(template).name)
        except Exception as e:
            raise Exception(
                f"Exception occured while reading template from file: {e}"
            ) from e
    else:
        # read from string
        try:
            env = Environment()
            return env.from_string(format(template))
        except Exception as e:
            raise Exception(
                f"Exception occured while reading template from string: {e}"
            ) from e


def validate_kwargs_from_default(_, attribute: Attribute, value: Dict):
    """Validate kwargs
    :param attribute:
    :type attribute: Attribute
    :param value:
    :type value: dict
    """

    def _create_default_type_str():
        """_create_default_type_str."""
        return "\n".join([f"{k}: {type(v)}" for k, v in attribute.default.items()])

    try:
        value_keys = value.keys()
    except AttributeError as e:
        raise AttributeError(f"{attribute.name} needs to be set with a dict") from e

    for key in value_keys:
        if key not in attribute.default:
            raise AttributeError(
                f"{attribute.name} can be one of the following:\n{_create_default_type_str()}"
            )
    # add missings to final value of attribute
    for k, v in attribute.default.items():
        if k not in value_keys:
            value[k] = v


def _check_type_from_signature(value, fn_param: inspect.Parameter) -> None:
    """Check parameter value type based on parameter signature.
    :param value:
    :param fn_param:
    :type fn_param: inspect.Parameter
    :rtype: None
    """
    # Handles only one layer of Union
    if get_origin(fn_param.annotation) is Union:
        _annotated_types = get_args(fn_param.annotation)
    else:
        _annotated_types = [fn_param.annotation]

    # Handles only the origin of GenericAlias (dict, list)
    _annotated_types = [
        get_origin(t) if isinstance(t, _GenericAlias) else t for t in _annotated_types
    ]

    type_check = any(isinstance(value, t) for t in _annotated_types)
    if not type_check:
        raise TypeError(
            f"Invalid type encountered for {fn_param.name}. Should be of type(s) {fn_param.annotation}. Passed value might be of type {type(value)}"
        )


def _check_type_from_default(value, fn_param: inspect.Parameter) -> None:
    """Check parameter value type based on default value.
    :param value:
    :param fn_param:
    :type fn_param: inspect.Parameter
    :rtype: None
    """
    # for parameters with default value
    _default = fn_param.default
    # Flag to skip parameters with None as default value
    is_none = _default is None or value is None

    if not is_none and not isinstance(value, type(_default)):
        raise TypeError(
            f"Invalid type encountered for {fn_param.name}. Expected type {type(_default)}"
        )


def validate_func_args(func):
    """Decorator for validating function parameters based on function signature
    :param func:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """wrapper.
        :param args:
        :param kwargs:
        """
        args_count = len(args)
        fn_params = inspect.signature(func).parameters

        # for parameters with annotation, preference is given to checking by annotation
        for arg, param in zip(args, list(fn_params)[:args_count]):
            if fn_params[param].annotation is not fn_params[param].empty:
                _check_type_from_signature(arg, fn_params[param])
            elif fn_params[param].default is not fn_params[param].empty:
                _check_type_from_default(arg, fn_params[param])

        for kwarg, value in kwargs.items():
            param = fn_params.get(kwarg)
            if not param:
                continue
            if fn_params[kwarg].annotation is not fn_params[kwarg].empty:
                _check_type_from_signature(value, fn_params[kwarg])
            elif fn_params[kwarg].default is not fn_params[kwarg].empty:
                _check_type_from_default(value, fn_params[kwarg])

        # Call the function after validation
        result = func(*args, **kwargs)
        return result

    return wrapper


def encode_img_base64(img: np.ndarray) -> str:
    """Encode a numpy array to a base64 str.
    :param img:
    :type img: np.ndarray
    :rtype: str
    """
    encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    _, buffer = cv2.imencode(".png", img, encode_params)
    return base64.b64encode(buffer).decode("utf-8")


class VADStatus(Enum):
    """VAD Status for start and end of detected speech"""

    START = 0
    ONGOING = 1
    END = 2


class WakeWordStatus(Enum):
    """WakeWord Status for start and end of detected wake word"""

    START = 0
    ONGOING = 1
    END = 2


def load_model(model_name: str, model_path: str) -> str:
    """Model download utility function"""
    from tqdm import tqdm
    from platformdirs import user_cache_dir

    cachedir = user_cache_dir("ros_agents")
    model_full_path = Path(cachedir) / Path("models") / Path(f"{model_name}.onnx")

    # create cache dir
    model_full_path.parent.mkdir(parents=True, exist_ok=True)

    # return if a file path is provided
    if Path(model_path).exists():
        return str(model_path)

    # check for cached model
    elif model_full_path.is_file():
        return str(model_full_path)

    else:
        # assume model path is a url open stream
        with httpx.stream("GET", model_path, timeout=20, follow_redirects=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            progress_bar = tqdm(
                total=total_size, unit="iB", unit_scale=True, desc=f"{model_name}"
            )
            # delete the file if an exception occurs while downloadin
            try:
                with open(model_full_path, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1024):
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            except Exception:
                import logging

                logging.error(
                    f"Error occured while downloading model {model_name} from given url. Try restarting your components."
                )
                if model_full_path.exists():
                    model_full_path.unlink()
                raise

    progress_bar.close()
    return str(model_full_path)


def flatten(xs):
    """Generator to flatten lists of arbitrary types"""
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


class PDFReader:
    """Load pdf using pdfreader. Used for testing PDF RAG"""

    @validate_func_args
    def __init__(
        self,
        pdf_file: Union[str, Path],
        password: Optional[str] = None,
    ) -> None:
        """__init__.
        :param pdf_file:
        :type pdf_file: Union[str, Path]
        :param password:
        :type password: Optional[str]
        :rtype: None
        """
        try:
            from pypdf import PdfReader
            from PIL import Image
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "In order to use the PDFReader, you need pypdf and pillow packages installed. You can install them with 'pip install pypdf pillow'"
            ) from e
        if not Path(pdf_file).is_file():
            raise TypeError(f"{pdf_file} is not a valid file")
        try:
            self.reader = PdfReader(pdf_file, password=password)
            self.image_reader = Image
        except Exception as e:
            raise TypeError(f"{pdf_file} is not a valid PDF") from e

    def extract(self, extract_images: bool = False):
        """Extract text from PDF documents.
        :param extract_images
        :type extract_images: bool
        """
        metadatas = []
        documents = []
        images = []
        ids = []

        for page_num, page in enumerate(self.reader.pages):
            # add pdf metadata as reader metadata and add pagenumber to it
            metadata = self.reader.metadata if self.reader.metadata else {}
            metadata["page"] = page_num
            metadatas.append(metadata)
            # get content
            content = page.get_contents()
            documents.append(content)
            # create a unique ID
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, content)))
            # get images if asked
            if extract_images:
                page_images = [
                    self.image_reader.open(BytesIO(img.data)) for img in page.images
                ]
                images += page_images

        return ids, metadatas, documents


_LANGUAGE_CODES = [
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
    "yue",
]
