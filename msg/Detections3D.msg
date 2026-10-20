# Detected objects in metric space.
#
# Produced by lifting 2D detections into metric space with the camera
# intrinsics and depth (either a depth image registered to the picture, or a
# point cloud projected onto it).

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

# How much depth each object's 3D box rests on, in [0, 1]: the valid depth
# samples used to build it divided by the pixel area of its 2D box, capped at 1.
# From a depth image a sample is a pixel, so this is the fraction of the box with
# valid depth. From a point cloud a sample is a point projecting into the box, so
# a sparse cloud such as a LiDAR reads low even for a well placed box. A box
# built from a handful of samples is geometrically meaningless, so consumers
# should check this before trusting one.
float64[] depth_validity

# The 2D detections these boxes were lifted from, parallel to `boxes`, and the
# frame of the image they were found in. Empty when the boxes did not come
# from an image.
Bbox2D[] boxes_2d
string source_frame
