# Detected objects in metric space.
#
# Produced by lifting 2D detections with a registered depth image and the
# camera intrinsics.

# The frame every box in this message is given in. Unlike 2D detections, which
# stay in the pixels of the camera that saw them and so cannot share a frame,
# metric boxes from several cameras can be transformed into one common frame.
std_msgs/Header header

# Detector confidence per object, parallel to `boxes`.
float64[] scores
# Object labels, parallel to `boxes`.
string[] labels
# The objects.
Bbox3D[] boxes

# Fraction of the depth pixels inside each object's 2D box that were valid
# and used to build its 3D box, in [0, 1] and parallel to `boxes`. A box built
# from a handful of pixels is geometrically meaningless, so consumers should
# check this before trusting one.
float64[] depth_validity

# The 2D detections these boxes were lifted from, parallel to `boxes`, and the
# frame of the image they were found in. Empty when the boxes did not come
# from an image.
Bbox2D[] boxes_2d
string source_frame
