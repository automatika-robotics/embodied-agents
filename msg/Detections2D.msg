# Detected objects in image space.

# Frame of the camera the detections were made in.
std_msgs/Header header

# Detector confidence per object, parallel to `boxes`.
float64[] scores
# Object labels, parallel to `boxes`.
string[] labels
# The objects, in pixels.
Bbox2D[]  boxes

# The frame the objects were found in. Either an image or compressed image
sensor_msgs/Image image
sensor_msgs/CompressedImage compressed_image

# Registered depth for the same frame, when the camera provides it. This is
# what lets a consumer lift these detections into metric space.
sensor_msgs/Image depth
