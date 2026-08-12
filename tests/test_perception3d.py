"""Tests for lifting 2D detections into metric 3D boxes — no ROS node needed."""

import numpy as np
import pytest

from agents.utils.perception3d import (
    Box3D,
    depth_validity,
    detections_to_message_fields,
    ensure_kompass_core,
    prepare_depth,
)

# A synthetic camera and a scene: a background wall with a nearer patch on it
FX = FY = 500.0
WIDTH, HEIGHT = 200, 120
CX, CY = WIDTH / 2, HEIGHT / 2
PATCH = (100, 40, 140, 60)  # x1, y1, x2, y2 — 40x20 px
PATCH_DEPTH_M = 0.5
BACKGROUND_DEPTH_M = 2.0


def scene(patch_depth=PATCH_DEPTH_M, background=BACKGROUND_DEPTH_M):
    """Depth image in meters with a nearer patch on a far background."""
    depth = np.full((HEIGHT, WIDTH), background, dtype=np.float32)
    x1, y1, x2, y2 = PATCH
    depth[y1:y2, x1:x2] = patch_depth
    return depth


class TestPrepareDepth:
    def test_float_meters_become_millimeters(self):
        depth = prepare_depth(np.full((4, 4), 1.5, dtype=np.float32))
        assert depth.dtype == np.uint16
        assert depth[0, 0] == 1500

    def test_integer_millimeters_are_kept(self):
        depth = prepare_depth(np.full((4, 4), 1500, dtype=np.uint16), encoding="16UC1")
        assert depth[0, 0] == 1500

    def test_encoding_wins_over_dtype(self):
        """A float image already in millimeters must not be scaled again."""
        depth = prepare_depth(np.full((4, 4), 1500.0, dtype=np.float32), encoding="mono16")
        assert depth[0, 0] == 1500

    def test_explicit_scale_overrides_everything(self):
        depth = prepare_depth(np.full((4, 4), 15, dtype=np.uint16), scale=100.0)
        assert depth[0, 0] == 1500

    def test_invalid_pixels_become_no_reading(self):
        raw = np.array([[np.nan, np.inf], [-np.inf, 1.0]], dtype=np.float32)
        depth = prepare_depth(raw)
        assert depth[0, 0] == 0 and depth[0, 1] == 0 and depth[1, 0] == 0
        assert depth[1, 1] == 1000

    def test_layout_is_column_major(self):
        """Row major gives the same answer but is copied element by element."""
        depth = prepare_depth(np.ascontiguousarray(scene()))
        assert depth.flags["F_CONTIGUOUS"]

    def test_single_channel_image_is_accepted(self):
        depth = prepare_depth(np.full((4, 4, 1), 1.0, dtype=np.float32))
        assert depth.shape == (4, 4)

    def test_non_2d_image_raises(self):
        with pytest.raises(ValueError, match="2D"):
            prepare_depth(np.zeros((4, 4, 3), dtype=np.float32))

    def test_values_beyond_the_range_are_capped(self):
        depth = prepare_depth(np.full((2, 2), 100.0, dtype=np.float32))
        assert depth[0, 0] == np.iinfo(np.uint16).max


class TestDepthValidity:
    def test_fully_covered_box(self):
        assert depth_validity(prepare_depth(scene()), PATCH) == 1.0

    def test_box_with_no_usable_depth(self):
        depth = prepare_depth(np.zeros((HEIGHT, WIDTH), dtype=np.float32))
        assert depth_validity(depth, PATCH) == 0.0

    def test_out_of_range_depth_does_not_count(self):
        depth = prepare_depth(scene(patch_depth=9.0, background=9.0))
        assert depth_validity(depth, PATCH, depth_range=(0.1, 5.0)) == 0.0

    def test_partially_valid_box(self):
        depth = prepare_depth(scene())
        # half the box sits on background at 2 m, still in range
        assert depth_validity(depth, (100, 40, 140, 60), (0.1, 1.0)) == 1.0
        assert depth_validity(depth, (100, 40, 140, 80), (0.1, 1.0)) == pytest.approx(0.5)

    def test_box_off_the_image(self):
        depth = prepare_depth(scene())
        assert depth_validity(depth, (500, 500, 600, 600)) == 0.0

    def test_empty_box(self):
        assert depth_validity(prepare_depth(scene()), (10, 10, 10, 10)) == 0.0


class TestMessageFields:
    def test_labels_and_scores_follow_the_boxes(self):
        """Boxes without usable depth are dropped, so metadata must be taken
        by the detection index rather than by position."""
        boxes = [
            Box3D(index=0, center=(0.5, 0.0, 0.0), size=(0.1, 0.1, 0.1), validity=0.9),
            Box3D(index=2, center=(1.0, 0.0, 0.0), size=(0.2, 0.2, 0.2), validity=0.4),
        ]
        fields = detections_to_message_fields(
            boxes,
            labels=["orange", "cup", "bowl"],
            scores=[0.9, 0.8, 0.7],
            boxes_2d=[(0, 0, 1, 1), (1, 1, 2, 2), (2, 2, 3, 3)],
        )
        assert fields["labels"] == ["orange", "bowl"]
        assert fields["scores"] == [0.9, 0.7]
        assert fields["depth_validity"] == [0.9, 0.4]
        assert fields["boxes_2d"] == [(0, 0, 1, 1), (2, 2, 3, 3)]
        assert fields["output"][0] == ((0.5, 0.0, 0.0), (0.1, 0.1, 0.1))

    def test_without_metadata(self):
        fields = detections_to_message_fields(
            [Box3D(index=0, center=(0.5, 0.0, 0.0), size=(0.1, 0.1, 0.1))]
        )
        assert fields["labels"] == [] and fields["scores"] == []
        assert len(fields["output"]) == 1

    def test_no_boxes(self):
        assert detections_to_message_fields([])["output"] == []


class TestKompassCoreGuard:
    def test_missing_dependency_is_actionable(self, monkeypatch):
        monkeypatch.setattr("agents.utils.perception3d.find_spec", lambda name: None)
        with pytest.raises(ModuleNotFoundError, match="kompass-core"):
            ensure_kompass_core()


class TestLifting:
    """The geometry itself, against a synthetic camera and scene."""

    @pytest.fixture(autouse=True)
    def _needs_kompass_core(self):
        pytest.importorskip("kompass_core.vision")

    @staticmethod
    def _intrinsics():
        from ros_sugar.io import CameraIntrinsics

        return CameraIntrinsics(
            fx=FX, fy=FY, cx=CX, cy=CY, width=WIDTH, height=HEIGHT,
            frame_id="camera_optical",
        )

    def _lift(self, boxes_2d, depth=None, **kwargs):
        from agents.utils.perception3d import boxes_from_detections, make_detector

        detector = make_detector(self._intrinsics(), **kwargs)
        depth_mm = prepare_depth(scene() if depth is None else depth)
        return boxes_from_detections(detector, depth_mm, boxes_2d)

    def test_patch_is_placed_at_its_distance_and_size(self):
        boxes = self._lift([PATCH])
        assert len(boxes) == 1
        box = boxes[0]
        # with no pose given, boxes come back in the camera's own optical
        # frame: x right, y down, z forward
        assert box.center[2] == pytest.approx(PATCH_DEPTH_M, abs=0.02)
        # 40 px wide at 0.5 m with a 500 px focal length is 4 cm
        assert box.size[0] == pytest.approx(40 * PATCH_DEPTH_M / FX, abs=0.01)
        assert box.size[1] == pytest.approx(20 * PATCH_DEPTH_M / FY, abs=0.01)
        assert box.validity == 1.0

    def test_boxes_are_placed_along_the_optical_axes(self):
        """An object right of and below the principal point has to come back
        with a positive x and a positive y, or every box handed to a planner
        is mirrored."""
        right_and_low = (130, 80, 150, 95)
        box = self._lift([right_and_low])[0]
        assert box.center[0] > 0 and box.center[1] > 0

    def test_a_pose_places_boxes_in_the_frame_it_is_given_in(self):
        """The detector works in forward-left-up axes internally. Handing it
        the camera's own optical-to-body rotation must therefore land the box
        in those axes, with the distance on x."""
        optical_to_body = (-0.5, 0.5, -0.5, 0.5)
        box = self._lift([PATCH], rotation=optical_to_body)[0]
        assert box.center[0] == pytest.approx(PATCH_DEPTH_M, abs=0.02)

    def test_depth_is_that_of_whatever_fills_the_box(self):
        """The distance is a median, so a box padded well beyond its object
        reports the wall behind it. Detections must be tight."""
        tight = (PATCH[0] - 2, PATCH[1] - 2, PATCH[2] + 2, PATCH[3] + 2)
        assert self._lift([tight])[0].center[2] == pytest.approx(
            PATCH_DEPTH_M, abs=0.05
        )

        loose = (PATCH[0] - 10, PATCH[1] - 10, PATCH[2] + 10, PATCH[3] + 10)
        assert self._lift([loose])[0].center[2] == pytest.approx(
            BACKGROUND_DEPTH_M, abs=0.05
        )

    def test_translation_moves_the_box(self):
        here = self._lift([PATCH])[0]
        moved = self._lift([PATCH], translation=(1.0, 0.0, 0.0))[0]
        assert moved.center[0] - here.center[0] == pytest.approx(1.0, abs=0.01)

    def test_boxes_without_depth_are_dropped_and_indices_survive(self):
        """kompass drops boxes it cannot place, so the surviving boxes must
        still say which detection they came from."""
        depth = scene()
        # a region with no readings, larger than the box below because the
        # detector reads its pixel limits inclusively
        depth[0:40, 0:40] = 0.0
        boxes = self._lift([(0, 0, 20, 20), PATCH], depth=depth)
        assert [box.index for box in boxes] == [1]

    def test_no_detections(self):
        assert self._lift([]) == []

    def test_lifted_boxes_build_a_message(self):
        from agents.ros import Detections3D

        boxes = self._lift([PATCH])
        fields = detections_to_message_fields(
            boxes, labels=["orange"], scores=[0.9], boxes_2d=[PATCH]
        )
        message = Detections3D.convert(**fields)
        assert message.labels == ["orange"]
        assert list(message.depth_validity) == [1.0]
        assert message.boxes[0].center.position.z == pytest.approx(
            PATCH_DEPTH_M, abs=0.02
        )
        # the 2D boxes have to become messages, not stay as pixel tuples
        assert message.boxes_2d[0].top_left_x == float(PATCH[0])
