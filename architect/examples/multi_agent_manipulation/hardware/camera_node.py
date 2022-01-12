"""A ROS node for getting turtlebot and box positions from an overhead camera."""

import apriltag
import cv2
import rospy

import numpy as np


class CameraNode(object):
    """A ROS node for interfacing with an overhead camera"""

    def __init__(self, rate: float = 20):
        """Initialize a CameraNode

        args:
            rate: rate in Hz at which state estimates will be published. Extracting
                positions for two turtlebots and one box takes ~6 ms, so the rate should
                be less than 100 Hz
        """
        super(CameraNode, self).__init__()
        self.rate = rospy.Rate(rate)

        # Create a ROS node
        rospy.init_node("CameraNode", anonymous=True)

        # Initialize the camera
        self.camera = cv2.VideoCapture(0)

        # Make sure to release the camera on shutdown
        rospy.on_shutdown(self.on_shutdown)

    def on_shutdown(self) -> None:
        """Release the camera on shutdown"""
        self.camera.release()

    def detect_apriltags(self, img: np.ndarray):
        """Detect all apriltags in the given grayscale image.

        args:
            img: grayscale image
        returns:
            a list of apriltag DetectionBase objects indicating the
            detected apriltags
        """
        # Set up the APRIL tag detector
        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)

        # Detect APRIL tags in the image
        results = detector.detect(img)

    def turtlebot_box_poses_from_image(self, img: np.ndarray):
        """Get the turtlebot and box poses from an image

        args:
            img: grayscale image
        returns:
            Three np.arrays of (x, y, theta) for the position and orientation of
            the box and two turtlebots. (x, y) are in meters relative to an origin
            tag, while theta is in radians relative to the x-axis of the origin tag.

            Could return None if a failure occurs.
        """
        # define ID numbers for the four tags we expect to see.
        origin_tag_id = 0
        box_tag_id = 1
        turtle1_tag_id = 2
        turtle2_tag_id = 3

        # define the size of the origin tag to provide a scale bar for the image
        origin_tag_side_length_m = 0.1

        # Get the apriltags from the image
        detections = self.detect_apriltags(img)
        origin_tag = None
        box_tag = None
        turtle1_tag = None
        turtle2_tag = None
        for d in detections:
            if d.tag_id == origin_tag_id:
                origin_tag = d
            elif d.tag_id == box_tag_id:
                box_tag = d
            elif d.tag_id == turtle1_tag_id:
                turtle1_tag = d
            elif d.tag_id == turtle2_tag_id:
                turtle2_tag = d

        # Fail if not all tags were detected
        if (
            origin_tag is None
            or box_tag is None
            or turtle1_tag is None
            or turtle2_tag is None
        ):
            return None

        # Use the size of the origin tag to set the scale for the image
        (c0, c1, _, _) = origin_tag.corners
        origin_tag_side_length_px = np.sqrt((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2)
        scale_m_per_px = origin_tag_side_length_m / origin_tag_side_length_px

        # Get the center of the origin in px
        origin_px = np.array(origin_tag.center)

        # Get the box and turtle positions relative to the origin (in camera frame)
        p_OBox_Camera = np.array(box_tag.center) - origin_px
        p_OTurtle1_Camera = np.array(turtle1_tag.center) - origin_px
        p_OTurtle2_Camera = np.array(turtle2_tag.center) - origin_px

        # Convert to meters
        p_OBox_Camera *= scale_m_per_px
        p_OTurtle1_Camera *= scale_m_per_px
        p_OTurtle2_Camera *= scale_m_per_px

        # Get orientation of the origin

        # Convert from camera frame to origin frame through a 2D rotation

        # Assemble results list and return
