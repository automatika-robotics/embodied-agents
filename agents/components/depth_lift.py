"""Shared machinery for lifting 2D detections into metric 3D boxes.

Components that publish Detections3D pair an rgb image with the depth registered to
it, the calibration of that stream, and the transform into the configured
`detections_frame`. This mixin holds that pairing and validation machinery.
"""

from typing import Any, Optional

from ..utils import get_frame_id, get_stamp_secs


class DepthLiftMixin:
    """Depth, calibration and detector handling for a 3D-lifting component."""

    def _init_lift_state(self) -> None:
        """The detector for the 3D lifted camera and its parsed calibration,
        each rebuilt only when the camera reports new values, and one-time
        warnings."""
        self._detector = None
        self._detector_key = None
        self._intrinsics = None
        self._intrinsics_key = None
        self._depth_encoding = None
        self._depth_encoding_key = None
        self._lift_msg = None
        self._lift_depth = None
        self._warned: set = set()

    def _depth_snapshot(self) -> Optional[Any]:
        """Depth as it stands the instant its picture is taken.

        Read alongside the picture rather than at publish time.
        Not needed for RGBD msgs
        """
        if self.depth_topic and (callback := self.callbacks.get(self.depth_topic.name)):
            return callback.msg
        return None

    def _depth_for(self) -> Optional[Any]:
        """The depth image registered to the lifted picture, or None"""
        # For RGBD
        if (depth := getattr(self._lift_msg, "depth", None)) is not None and depth.data:
            return depth

        if (depth := self._lift_depth) is None:
            self._warn_once(
                "no_depth",
                f"Nothing has been received on depth topic "
                f"'{self.depth_topic.name if self.depth_topic else '<unknown>'}', so no detection "
                "can be placed in space.",
                error=True,
            )
            return None

        # For Image. Captured at the same instant, check for age nonetheless
        age = abs(get_stamp_secs(self._lift_msg) - get_stamp_secs(depth))
        if age > self.config.max_depth_age:
            self._warn_once(
                "depth_age",
                f"Depth on '{self.depth_topic.name}' is {age:.2f}s away from the "
                f"picture it would be paired with, more than max_depth_age "
                f"({self.config.max_depth_age}s), so these detections are not "
                "being published.",
            )
            return None
        return depth

    def _camera_intrinsics(self) -> Optional[Any]:
        """Calibration of the stream the depth was measured on.
        A camera_info topic wins when one was given.
        """

        from ..ros import read_camera_info

        if self.camera_info_topic and (
            info := self.callbacks.get(self.camera_info_topic.name)
        ):
            if (parsed := info.get_output()) is not None:
                return parsed

        info = getattr(self._lift_msg, "depth_camera_info", None)
        if info is None:
            return None
        # width, height, K and P are full-frame calibration values that do not
        # change when the driver applies binning or a region of interest
        roi = getattr(info, "roi", None)
        key = (
            info.header.frame_id,
            info.width,
            info.height,
            tuple(info.k),
            tuple(info.p),
            getattr(info, "binning_x", 0),
            getattr(info, "binning_y", 0),
            (roi.x_offset, roi.y_offset, roi.width, roi.height) if roi else None,
        )
        # Update key if it changed
        if key != self._intrinsics_key:
            self._intrinsics = read_camera_info(info)
            self._intrinsics_key = key
        return self._intrinsics

    def _depth_detector(self):
        """The 3D detector for the lifted camera, and the depth to run it on.

        :returns: (detector, depth in millimeters), both None when this tick
            cannot be lifted
        """
        from ..callbacks import image_pre_processing, process_encoding

        from ..utils.perception3d import make_detector, prepare_depth

        depth_msg = self._depth_for()
        if depth_msg is None:
            return None, None

        intrinsics = self._camera_intrinsics()
        if intrinsics is None:
            self._warn_once(
                "no_intrinsics",
                "No camera calibration has been received, so depth cannot be "
                "turned into distances. Pass the camera's `camera_info` topic, "
                "or use an RGBD input, which carries its own.",
                error=True,
            )
            return None, None

        # The intrinsics need to describe the depth image
        if (intrinsics.width, intrinsics.height) != (depth_msg.width, depth_msg.height):
            self._warn_once(
                "intrinsics_resolution",
                f"The camera reports intrinsics for {intrinsics.width}x"
                f"{intrinsics.height} images but publishes depth at "
                f"{depth_msg.width}x{depth_msg.height}. Detections cannot be "
                "placed in metric space until the two agree.",
                error=True,
            )
            return None, None

        # Check if the streams encoding changed (unlikely)
        if depth_msg.encoding != self._depth_encoding_key:
            self._depth_encoding = process_encoding(depth_msg.encoding)
            self._depth_encoding_key = depth_msg.encoding

        depth_mm = prepare_depth(
            image_pre_processing(depth_msg, *self._depth_encoding),
            encoding=depth_msg.encoding,
            scale=self.config.depth_scale,
        )

        # Registered depth shares the color image's pixel grid preferrably
        color_frame = get_frame_id(getattr(self._lift_msg, "rgb", self._lift_msg))
        depth_frame = get_frame_id(depth_msg)
        if color_frame and depth_frame and color_frame != depth_frame:
            self._warn_once(
                "frame_mismatch",
                f"The depth stream reports frame '{depth_frame}' while the "
                f"pictures are in '{color_frame}'. Depth registered to the "
                "color image lives in the color camera's frame, so that is "
                "the one used. If this depth stream is not the registered "
                "one, every 3D box will be misplaced.",
            )
        camera_frame = color_frame or depth_frame
        target = self.config.detections_frame
        translation = rotation = None
        if target != camera_frame:
            listener = self.get_transform_listener(
                camera_frame, target, self.config.static_camera_tf
            )
            if not listener.got_transform:
                self._warn_once(
                    "camera_transform",
                    f"The transform from camera frame '{camera_frame}' to "
                    f"'{target}' has not been resolved yet, so detections are "
                    "not being published in 3D.",
                )
                return None, None
            translation, rotation = listener.translation, listener.rotation

        # Rebuilding is cheap, so the detector is kept only until something it
        # was built from moves
        key = (
            camera_frame,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            None if translation is None else tuple(translation),
            None if rotation is None else tuple(rotation),
        )
        if key != self._detector_key:
            self._detector = make_detector(
                intrinsics,
                translation=translation,
                rotation=rotation,
                depth_range=(self.config.min_depth, self.config.max_depth),
            )
            self._detector_key = key
        return self._detector, depth_mm

    # TODO: Upstream to sugarcoat component
    def _warn_once(self, key: str, message: str, error: bool = False) -> None:
        """Report a lasting misconfiguration the first time it is noticed"""
        if key in self._warned:
            return
        self._warned.add(key)
        if error:
            self.get_logger().error(message)
        else:
            self.get_logger().warning(message)
