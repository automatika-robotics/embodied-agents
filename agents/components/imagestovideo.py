"""Deprecated module kept for backwards compatibility."""

import warnings

from .motion_detection import MotionDetector

__all__ = ["VideoMessageMaker"]


class VideoMessageMaker(MotionDetector):
    """Deprecated alias of :class:`~agents.components.motion_detection.MotionDetector`.

    .. deprecated::
        VideoMessageMaker is deprecated and will be removed in a future
        release. Use the :class:`~agents.components.motion_detection.MotionDetector`
        component instead, which generalizes it to motion detection from
        images and point clouds with a Bool motion state output.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "VideoMessageMaker is deprecated; use MotionDetector instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
