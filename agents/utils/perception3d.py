"""Turn 2D detections into metric 3D boxes given a depth image registered to the
frame the detections were made in and the camera's intrinsics.

The geometry itself is done by kompass-core's depth detector, which takes the
median of the depth pixels inside a box, keeps the pixels within a median
absolute deviation of it, and derives the object's extents from those. It is an
optional dependency.

Being a median, the reported distance is that of whatever fills most of the
box, so detections should be tight around their objects.

Boxes come back in the frame the camera's pose was given in, or in the
camera's own optical frame when no pose is given.
"""

# TODO: See if any of these utilities need to be upstreamed

from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from attrs import define, field

_KOMPASS_MIN_VERSION = (0, 8, 4)
_KOMPASS_INSTALL_HINT = (
    "'kompass-core' >= 0.8.4 is required to lift 2D detections into 3D. "
    "Install it with: pip install 'kompass-core>=0.8.4'"
)

# Depth encodings carrying integer millimeters rather than float meters
_MILLIMETER_ENCODINGS = {"16uc1", "mono16"}

# kompass-core works in millimeters
_METERS_TO_MM = 1000.0


def ensure_kompass_core() -> None:
    """Raise an actionable error when kompass_core is missing or too old."""
    if find_spec("kompass_core") is None:
        raise ModuleNotFoundError(_KOMPASS_INSTALL_HINT)
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = tuple(int(part) for part in version("kompass-core").split(".")[:3])
    except (PackageNotFoundError, ValueError):
        return  # a source checkout with no metadata: trust it
    if installed < _KOMPASS_MIN_VERSION:
        raise ImportError(
            f"{_KOMPASS_INSTALL_HINT} (found {'.'.join(map(str, installed))})"
        )


def resolve_lift_camera(
    inputs: Sequence[Any],
    depth: Optional[Any],
    camera_info: Optional[Any],
    *,
    frame: str,
    component: str,
) -> str:
    """Validate a component's 3D lifting contract and name its lift camera.

    A component asked for Detections3D output needs a frame to report boxes
    in, and depth registered to the picture stream the detections are made
    on: either an RGBD input, or a plain depth topic with the calibration of
    the stream it was measured on. Raises TypeError describing whichever
    part of the contract is missing.

    :param inputs: The component's input topics
    :param depth: Topic given as the ``depth`` keyword, if any
    :param camera_info: Topic given as the ``camera_info`` keyword, if any
    :param frame: The configured ``detections_frame``
    :param component: Component name for the error messages
    :return: Name of the picture topic the lift applies to
    """
    from ..ros import CameraInfo, CompressedImage, Image, RGBD

    # Check if detection frame has been set. 3D Boxes are axis aligned in it.
    if not frame:
        raise TypeError(
            f"{component} was given a Detections3D output, which needs a frame "
            "to report boxes in. Set `detections_frame` on the config to the "
            "frame the consumer works in, such as a robotic arm's planning "
            "frame."
        )

    pictures = [
        t
        for t in inputs
        if issubclass(t.msg_type, (Image, RGBD)) and (not depth or t.name != depth.name)
    ]
    if camera_info and not issubclass(camera_info.msg_type, CameraInfo):
        raise TypeError(
            f"{component} camera_info topic must be of type CameraInfo, got "
            f"{camera_info.msg_type.__name__}."
        )

    # Prefer RGBD over plain image. Return first RGBD if multiple.
    # Otherwise the first Image topic is taken at the end.
    rgbd = [t for t in pictures if issubclass(t.msg_type, RGBD)]
    if rgbd:
        return rgbd[0].name

    if not depth:
        raise TypeError(
            f"{component} was given a Detections3D output, which requires "
            "depth to place detections in space. Either give it an RGBD "
            "input, which carries depth registered to its picture, or pass "
            "the camera's registered depth topic as `depth=Topic(...)` along "
            f"with its `camera_info=Topic(...)`. Inputs given: "
            f"{[t.name for t in inputs]}"
        )
    if issubclass(depth.msg_type, CompressedImage) or not issubclass(
        depth.msg_type, Image
    ):
        raise TypeError(
            f"{component} depth topic must be an uncompressed Image, got "
            f"{depth.msg_type.__name__}."
        )
    if not camera_info:
        raise TypeError(
            f"{component} was given a depth topic but no camera_info topic. "
            "Depth pixels cannot be turned into distances without the "
            "calibration of the stream they were measured on: pass it as "
            "`camera_info=Topic(...)`."
        )
    # Take first image topic by default
    return pictures[0].name


@define
class Box3D:
    """An object detected in metric space.

    :param index: Index of the 2D detection this box was lifted from, so the
        label and score of that detection can be carried over. Boxes without
        usable depth are dropped, so this is not the position in the output
    :param center: Center of the box in meters
    :param size: Full extents of the box in meters
    :param validity: Fraction of the depth pixels inside the 2D box that were
        usable, in [0, 1]
    """

    index: int = field()
    center: Tuple[float, float, float] = field()
    size: Tuple[float, float, float] = field()
    validity: float = field(default=0.0)


def prepare_depth(
    depth: np.ndarray,
    encoding: Optional[str] = None,
    scale: Optional[float] = None,
) -> np.ndarray:
    """Put a depth image in the layout kompass-core reads.

    Depth arrives either as float meters or as integer millimeters depending
    on the camera, and invalid pixels are marked as 0, NaN or infinity
    depending on the driver. All of them become a plain 0, which is what
    kompass-core treats as no reading.

    :param depth: Depth image as (H, W) or (H, W, 1)
    :param encoding: ROS encoding of the depth image, used to tell integer
        millimeters from float meters when the dtype alone is ambiguous
    :param scale: Multiplier from the image's units to millimeters,
        overriding what the encoding and dtype imply
    :returns: Depth in millimeters as a column major uint16 array
    """
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be 2D, got shape {depth.shape}")

    if scale is None:
        if encoding and encoding.lower() in _MILLIMETER_ENCODINGS:
            scale = 1.0
        elif np.issubdtype(depth.dtype, np.floating):
            scale = _METERS_TO_MM
        else:
            scale = 1.0

    if np.issubdtype(depth.dtype, np.floating):
        # NaN for no return, infinity for out of range: both mean no reading
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    depth = np.clip(depth * scale, 0, np.iinfo(np.uint16).max)
    return np.require(depth, dtype=np.uint16, requirements=["F"])


def depth_validity(
    depth_mm: np.ndarray,
    box: Sequence[float],
    depth_range: Tuple[float, float] = (0.1, 5.0),
) -> float:
    """Fraction of a 2D box's depth pixels that carry a usable reading.

    A box built from a handful of pixels is geometrically meaningless, so
    consumers use this to decide whether to trust the 3D box built from it.

    :param depth_mm: Depth image in millimeters
    :param box: 2D box as (x1, y1, x2, y2) in pixels
    :param depth_range: Usable range of the sensor in meters
    :returns: Fraction in [0, 1], 0 when the box is empty or off the image
    """
    height, width = depth_mm.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    x1, x2 = max(0, min(x1, x2)), min(width, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(height, max(y1, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    patch = depth_mm[y1:y2, x1:x2]
    low, high = (limit * _METERS_TO_MM for limit in depth_range)
    usable = np.count_nonzero((patch >= low) & (patch <= high))
    return float(usable) / float(patch.size)


def make_detector(
    intrinsics: Any,
    translation: Optional[Sequence[float]] = None,
    rotation: Optional[Sequence[float]] = None,
    depth_range: Tuple[float, float] = (0.1, 5.0),
) -> Any:
    """Build a depth detector for one camera.

    The pose is that of the camera's **optical** frame, which is the one an
    image names in its header and the one TF can resolve. Boxes come back in
    the frame that pose is given in, or in the camera's own optical frame when
    no pose is given.

    :param intrinsics: Camera intrinsics (ros_sugar.io.CameraIntrinsics)
    :param translation: Camera position in the target frame, identity if unset
    :param rotation: Camera orientation in the target frame as (x, y, z, w),
        identity if unset
    :param depth_range: Usable range of the sensor in meters
    :returns: kompass_core DepthDetector
    """
    ensure_kompass_core()
    from kompass_core.vision import CameraFrameConvention, DepthDetector

    # The pose is that of the OPTICAL frame, what a TF lookup against an
    # image's frame_id gives; the convention is passed explicitly below
    rotation = rotation if rotation is not None else (0.0, 0.0, 0.0, 1.0)

    return DepthDetector(
        np.asarray(depth_range, dtype=np.float32),
        np.asarray(
            translation if translation is not None else (0.0, 0.0, 0.0),
            dtype=np.float32,
        ),
        np.asarray(rotation, dtype=np.float32),
        np.asarray([intrinsics.fx, intrinsics.fy], dtype=np.float32),
        np.asarray([intrinsics.cx, intrinsics.cy], dtype=np.float32),
        1e-3,  # the depth images handed over are in millimeters
        convention=CameraFrameConvention.OPTICAL,
    )


def boxes_from_detections(
    detector: Any,
    depth_mm: np.ndarray,
    boxes_2d: Sequence[Sequence[float]],
    image_size: Optional[Tuple[int, int]] = None,
    depth_range: Tuple[float, float] = (0.1, 5.0),
    camera_position: Optional[Sequence[float]] = None,
) -> List[Box3D]:
    """Lift 2D detections into metric boxes.

    Detections whose pixels carry no usable depth cannot be placed in space
    and are left out, so each returned box records which detection it came
    from rather than relying on position.

    :param detector: Detector from :func:`make_detector`
    :param depth_mm: Depth image from :func:`prepare_depth`
    :param boxes_2d: 2D boxes as (x1, y1, x2, y2) in pixels
    :param image_size: Size of the image the boxes were found in as
        (width, height), taken from the depth image if unset
    :param depth_range: Usable range of the sensor in meters
    :param camera_position: Position of the camera in the frame the boxes come
        back in, which must be gravity aligned (z up). When given, each box's
        center is pushed away from the camera along the view ray by half the
        box's smallest extent, horizontally. The push
        assumes the hidden half mirrors the visible one. Exact for spheres and
        face-on boxes. Vertical extant is left alone as its observed in most
        cases.
    :returns: The boxes that could be placed, in the frame the detector's
        camera pose was given in
    """
    ensure_kompass_core()
    from kompass_cpp.types import Bbox2D

    if not len(boxes_2d):
        return []

    height, width = depth_mm.shape[:2]
    if image_size is None:
        image_size = (width, height)

    inputs = []
    for index, box in enumerate(boxes_2d):
        x1, y1, x2, y2 = (float(value) for value in box)
        corner = np.array([min(x1, x2), min(y1, y2)], dtype=np.int32)
        extent = np.array([abs(x2 - x1), abs(y2 - y1)], dtype=np.int32)
        # The detector carries the label through untouched
        detection = Bbox2D(
            top_left_corner=corner,
            size=extent,
            timestamp=0.0,
            label=str(index),
        )
        detection.set_img_size(np.array(image_size, dtype=np.int32))
        inputs.append(detection)

    detected = detector.compute_3d_detections(depth_mm, inputs, 0.0, 0.0, 0.0, 0.0)
    if not detected:
        return []

    camera = (
        None
        if camera_position is None
        else np.asarray(camera_position, dtype=np.float64)
    )
    boxes = []
    for box in detected:
        try:
            index = int(box.label)
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(boxes_2d):
            continue
        center = np.asarray(box.center, dtype=np.float64)
        size = np.asarray(box.size, dtype=np.float64)
        if camera is not None:
            # push center within the object (assumptions in docstring)
            ray = center - camera
            norm = float(np.linalg.norm(ray))
            if norm > 1e-6:
                push = float(size.min()) / 2.0
                center[0] += ray[0] / norm * push
                center[1] += ray[1] / norm * push
        boxes.append(
            Box3D(
                index=index,
                center=(float(center[0]), float(center[1]), float(center[2])),
                size=(float(size[0]), float(size[1]), float(size[2])),
                validity=depth_validity(depth_mm, boxes_2d[index], depth_range),
            )
        )
    return boxes


def detections_to_message_fields(
    boxes: Sequence[Box3D],
    labels: Optional[Sequence[str]] = None,
    scores: Optional[Sequence[float]] = None,
    boxes_2d: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, List]:
    """Line metric boxes up with the 2D detections they came from.

    :param boxes: Boxes from :func:`boxes_from_detections`
    :param labels: Labels of the 2D detections
    :param scores: Scores of the 2D detections
    :param boxes_2d: The 2D boxes themselves, to carry into the message
    :returns: Keyword arguments for the Detections3D converter
    """
    fields: Dict[str, List] = {
        "output": [(box.center, box.size) for box in boxes],
        "labels": [],
        "scores": [],
        "depth_validity": [box.validity for box in boxes],
        "boxes_2d": [],
    }
    for box in boxes:
        if labels is not None and box.index < len(labels):
            fields["labels"].append(labels[box.index])
        if scores is not None and box.index < len(scores):
            fields["scores"].append(float(scores[box.index]))
        if boxes_2d is not None and box.index < len(boxes_2d):
            fields["boxes_2d"].append(boxes_2d[box.index])
    return fields
