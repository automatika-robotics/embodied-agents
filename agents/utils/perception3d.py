"""Turn 2D detections into metric 3D boxes given a depth image registered to the
frame the detections were made in and the camera's intrinsics or a point cloud from
a LiDAR or a stereo camera.

The geometry itself is done by kompass-core's depth detector, which takes the
median of the depth pixels or points inside a box, keeps them within a median
absolute deviation of it, and derives the object's extents from those. It is an
optional dependency.

Being a median, the reported distance is that of whatever fills most of the
box, so detections should be tight around their objects. Boxes come back in the
frame the camera's pose was given in, or in the camera's own optical frame when no
pose is given.
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
    in, and depth for the picture stream the detections are made on. It can be RGBD
    input, or a depth Image registered to the pictures or a PointCloud2, each
    with the camera's calibration.

    :param inputs: The component's input topics
    :param depth: Topic given as the ``depth`` keyword, if any
    :param camera_info: Topic given as the ``camera_info`` keyword, if any
    :param frame: The configured ``detections_frame``
    :param component: Component name for the error messages
    :return: Name of the picture topic the lift applies to
    """
    from ..ros import CameraInfo, CompressedImage, Image, PointCloud2, RGBD

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
            "`depth=Topic(...)`, a depth Image registered to the pictures or "
            "a PointCloud2, along with the camera's `camera_info=Topic(...)`. "
            f"Inputs given: {[t.name for t in inputs]}"
        )
    if issubclass(depth.msg_type, CompressedImage) or not issubclass(
        depth.msg_type, (Image, PointCloud2)
    ):
        raise TypeError(
            f"{component} depth topic must be an uncompressed Image or a "
            f"PointCloud2, got {depth.msg_type.__name__}."
        )
    if not camera_info:
        raise TypeError(
            f"{component} was given a depth topic but no camera_info topic. "
            "Depth cannot be tied to the picture's pixels without the "
            "calibration of the camera that took it: pass it as "
            "`camera_info=Topic(...)`."
        )
    if not pictures:
        raise TypeError(
            f"{component} has no picture topic to detect on: `inputs` needs an "
            "Image other than the depth topic. Inputs given: "
            f"{[t.name for t in inputs]}"
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
    :param validity: Depth readings the box rests on per pixel of its 2D box,
        clamped to [0, 1]. For a depth image that is the fraction of usable
        pixels; for a point cloud it counts the points that landed in the
        box, which is small for a sparse LiDAR
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
    :returns: Depth in millimeters as a C-contiguous uint16 array, the
        layout kompass-core reads without a copy
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
    return np.require(depth, dtype=np.uint16, requirements=["C"])


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


def set_cloud_sensor(
    detector: Any,
    translation: Optional[Sequence[float]] = None,
    rotation: Optional[Sequence[float]] = None,
    field_datatype: Optional[int] = None,
) -> None:
    """Tell a detector where its point clouds are measured from.

    The pose is that of the cloud's frame in the same frame the detector's
    camera pose was given in.

    :param detector: Detector from :func:`make_detector`
    :param translation: Position of the cloud's frame, identity if unset
    :param rotation: Orientation of the cloud's frame as (x, y, z, w),
        identity if unset
    :param field_datatype: sensor_msgs/PointField datatype of the x, y, z
        fields, FLOAT32 if unset
    """
    from kompass_cpp.types import PointFieldType, SensorConfig

    detector.set_point_cloud_sensor(
        SensorConfig(
            position=np.asarray(
                translation if translation is not None else (0.0, 0.0, 0.0),
                dtype=np.float32,
            ),
            rotation=np.asarray(
                rotation if rotation is not None else (0.0, 0.0, 0.0, 1.0),
                dtype=np.float32,
            ),
            cloud_field_type=PointFieldType.FLOAT32
            if field_datatype is None
            else PointFieldType.from_int(field_datatype),
        )
    )


def _pixel_boxes(
    boxes_2d: Sequence[Sequence[float]], image_size: Tuple[int, int]
) -> List[Any]:
    """The 2D boxes as detector inputs."""
    from kompass_cpp.types import Bbox2D

    inputs = []
    for box in boxes_2d:
        x1, y1, x2, y2 = (float(value) for value in box)
        corner = np.array([min(x1, x2), min(y1, y2)], dtype=np.int32)
        extent = np.array([abs(x2 - x1), abs(y2 - y1)], dtype=np.int32)
        detection = Bbox2D(top_left_corner=corner, size=extent, timestamp=0.0)
        detection.set_img_size(np.array(image_size, dtype=np.int32))
        inputs.append(detection)
    return inputs


def boxes_from_detections(
    detector: Any,
    depth: Any,
    boxes_2d: Sequence[Sequence[float]],
    image_size: Optional[Tuple[int, int]] = None,
    camera_position: Optional[Sequence[float]] = None,
) -> List[Box3D]:
    """Lift 2D detections into metric boxes.

    Detections whose pixels carry no usable depth cannot be placed in space
    and are left out, so each returned box records which detection it came
    from (the detector's ``source_index``).

    :param detector: Detector from `make_detector`
    :param depth: Depth image from `prepare_depth`, or a point cloud as
        a ros_sugar.io.PointCloudData, from a sensor the detector was told
        about with `set_cloud_sensor`
    :param boxes_2d: 2D boxes as (x1, y1, x2, y2) in pixels
    :param image_size: Size of the image the boxes were found in as
        (width, height), taken from the depth image if unset. Required with
        a point cloud, which has no pixel grid of its own
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
    from ..ros import PointCloudData

    if not len(boxes_2d):
        return []

    cloud = isinstance(depth, PointCloudData)
    if image_size is None:
        if cloud:
            raise TypeError(
                "image_size is required to lift detections from a point cloud"
            )
        height, width = depth.shape[:2]
        image_size = (width, height)

    inputs = _pixel_boxes(boxes_2d, image_size)
    if cloud:
        detected = detector.compute_3d_detections(
            **depth.buffer_layout(),
            input=inputs,
            robot_x=0.0,
            robot_y=0.0,
            robot_yaw=0.0,
            robot_speed=0.0,
        )
    else:
        detected = detector.compute_3d_detections(depth, inputs, 0.0, 0.0, 0.0, 0.0)
    if not detected:
        return []

    camera = (
        None
        if camera_position is None
        else np.asarray(camera_position, dtype=np.float64)
    )
    boxes = []
    for box in detected:
        index = box.source_index
        x1, y1, x2, y2 = boxes_2d[index]
        area = abs(x2 - x1) * abs(y2 - y1)
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
                # the detector reads its limits inclusively, hence the clamp
                validity=min(1.0, box.sample_count / area) if area else 0.0,
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
