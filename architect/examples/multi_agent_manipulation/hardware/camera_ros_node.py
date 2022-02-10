"""A ROS node for getting turtlebot and box positions from an overhead camera."""
from typing import List, Optional, Tuple

import apriltag
import cv2
from cv_bridge import CvBridge
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from architect.components.geometry.transforms_2d import rotation_matrix_2d


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

        # Create a ROS node
        rospy.init_node("CameraNode", anonymous=True)
        self.rate = rospy.Rate(rate)

        # Create topics for turtlebot and box poses
        self.box_pose_publisher = rospy.Publisher(
            "box/pose", PoseStamped, queue_size=10
        )
        self.turtle1_pose_publisher = rospy.Publisher(
            "turtle1/pose", PoseStamped, queue_size=10
        )
        self.turtle2_pose_publisher = rospy.Publisher(
            "turtle2/pose", PoseStamped, queue_size=10
        )
        self.pose_publishers = [
            self.box_pose_publisher,
            self.turtle1_pose_publisher,
            self.turtle2_pose_publisher,
        ]
        # Also publish images
        self.image_publisher = rospy.Publisher("overhead_image", Image, queue_size=10)
        self.cv_bridge = CvBridge()

        # Subscribe to the desired box pose so we can draw it
        self.desired_box_pose = None
        rospy.Subscriber(
            "desired_box_pose", PoseStamped, self.desired_box_pose_callback
        )

        # Track message IDs
        self.message_id = 0

        # Initialize the camera and set the resolution to 1920x1080
        # and FPS to the set rate
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.camera.set(cv2.CAP_PROP_FPS, 30.0)

        channel = self.camera.get(cv2.CAP_PROP_CHANNEL)
        w = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.camera.get(cv2.CAP_PROP_FPS)
        rospy.loginfo(f"Camera channel {channel}, resolution {w}x{h}, fps {fps}.")

        # Make sure to release the camera on shutdown
        rospy.on_shutdown(self.on_shutdown)

    def on_shutdown(self) -> None:
        """Release the camera on shutdown"""
        self.camera.release()

    def desired_box_pose_callback(self, msg):
        """Process desired box pose messages"""
        # Convert from quaternion to orientation angle
        quaternion = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        euler_rotation = euler_from_quaternion(quaternion)

        # Store the xy position and orientation in the pose vector
        self.desired_box_pose = np.array(
            [msg.pose.position.x, msg.pose.position.y, euler_rotation[2]]
        )

    def step(self) -> None:
        """Run the detector once and publish the detected poses"""
        # Capture the image
        ret, img = self.camera.read()

        # Pass if no image captured
        if not ret:
            return

        # Resize the image and convert to grayscale
        scale_factor = 1000.0 / img.shape[1]  # new image should have width 1000 pixels
        new_dimensions = (1000, int(img.shape[0] * scale_factor))
        img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get poses from image
        poses, annotated_img = self.turtlebot_box_poses_from_image(grayscale_img, img)

        # Publish the image
        image_message = self.cv_bridge.cv2_to_imgmsg(annotated_img, encoding="bgr8")
        self.image_publisher.publish(image_message)

        # Format poses into messages to publish
        for pose, pose_publisher in zip(poses, self.pose_publishers):
            # Skip if no pose detected
            if pose is None:
                continue

            pose_msg = PoseStamped()

            # Fill in the header
            pose_msg.header.seq = self.message_id
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "origin_tag"

            # Fill in the pose
            pose_msg.pose.position.x = pose[0]
            pose_msg.pose.position.y = pose[1]
            pose_msg.pose.position.z = 0.0

            quaternion_pose = quaternion_from_euler(0.0, 0.0, pose[2])
            pose_msg.pose.orientation.x = quaternion_pose[0]
            pose_msg.pose.orientation.y = quaternion_pose[1]
            pose_msg.pose.orientation.z = quaternion_pose[2]
            pose_msg.pose.orientation.w = quaternion_pose[3]

            # publish
            pose_publisher.publish(pose_msg)

            self.message_id += 1

        # Sleep
        self.rate.sleep()

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

        return results

    def turtlebot_box_poses_from_image(
        self, grayscale_img: np.ndarray, color_img: np.ndarray
    ) -> Tuple[List[Optional[np.ndarray]], np.ndarray]:
        """Get the turtlebot and box poses from an image

        args:
            grayscale_img: grayscale image
            color_img: color image (for annotation)
        returns:
            - List of np.arrays of (x, y, theta) for the position and orientation of
                the box and two turtlebots. (x, y) are in meters relative to an origin
                tag, while theta is in radians relative to the x-axis of the origin tag.
                Each element could be none if there is a detection issue.
            - An annotated image of any detections.
        """
        # Create a copy of the color image and annotate it
        annotated_img = color_img.copy()

        # define ID numbers for the four tags we expect to see.
        origin_tag_id = 0
        box_tag_id = 1
        turtle1_tag_id = 2
        turtle2_tag_id = 3

        # define the size of the origin tag to provide a scale bar for the image
        origin_tag_side_length_m = 0.1

        # Get the apriltags from the image
        detections = self.detect_apriltags(grayscale_img)
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

        # Fail if the origin was not detected
        if origin_tag is None:
            return [None, None, None], annotated_img

        # Use the size of the origin tag to set the scale for the image
        (co0, co1, _, _) = origin_tag.corners
        origin_tag_side_length_px = np.sqrt(
            (co0[0] - co1[0]) ** 2 + (co0[1] - co1[1]) ** 2
        )
        scale_m_per_px = origin_tag_side_length_m / origin_tag_side_length_px

        # Get the center of the origin in px
        origin_px = np.array(origin_tag.center)

        # Get orientation of the origin
        # co0 is the upper left corner, co1 is the upper right, so co0 -> co1 points
        # along the x axis of the origin
        p_Co0Co1_Camera_px = np.array(co1) - np.array(co0)
        ydiff = p_Co0Co1_Camera_px[1]
        xdiff = p_Co0Co1_Camera_px[0]
        theta_CameraO = np.arctan2(ydiff, xdiff)

        # Make a 2D rotation to convert from camera to origin frames
        R_CameraO = rotation_matrix_2d(theta_CameraO)
        R_OCamera = R_CameraO.T  # type: ignore

        # Annotate the origin
        cv2.circle(annotated_img, origin_px.astype(int), 10, (255, 0, 0), -1)
        x_axis = origin_px + R_CameraO @ np.array([50.0, 0.0])
        y_axis = origin_px + R_CameraO @ np.array([0.0, -50.0])
        cv2.line(
            annotated_img,
            origin_px.astype(int),
            (int(x_axis[0]), int(x_axis[1])),
            (0, 0, 255),
            10,
        )
        cv2.line(
            annotated_img,
            origin_px.astype(int),
            (int(y_axis[0]), int(y_axis[1])),
            (0, 255, 0),
            10,
        )
        cv2.putText(
            annotated_img,
            "Origin",
            origin_px.astype(int),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Process the rest of the points
        tags = [box_tag, turtle1_tag, turtle2_tag]
        tag_names = ["Box", "Turtle1", "Turtle2"]
        poses: List[Optional[np.ndarray]] = [None, None, None]
        for idx, (tag, tag_name) in enumerate(zip(tags, tag_names)):
            # Skip if no detection
            if tag is None:
                continue

            # Get tag position in pixels relative to origin in camera frame
            p_CameraTag_px = np.array(tag.center)
            p_OTag_Camera_px = p_CameraTag_px - origin_px

            # Convert to meters
            p_OTag_Camera_m = p_OTag_Camera_px * scale_m_per_px

            # Convert to origin frame
            p_OTag_m = R_OCamera @ p_OTag_Camera_m

            # Store position
            pose = np.zeros(3)
            pose[:2] = p_OTag_m

            # Get tag orientation in camera frame
            (ctag0, ctag1, _, _) = tag.corners
            p_Ctag0Ctag1_Camera_px = np.array(ctag1) - np.array(ctag0)
            ydiff = p_Ctag0Ctag1_Camera_px[1]
            xdiff = p_Ctag0Ctag1_Camera_px[0]
            theta_CameraTag = np.arctan2(ydiff, xdiff)

            # Convert to origin frame
            theta_OTag = theta_CameraTag - theta_CameraO

            # Save orientation
            pose[2] = theta_OTag
            poses[idx] = pose

            # Annotate the tag
            R_CameraTag = rotation_matrix_2d(theta_CameraTag)
            cv2.circle(annotated_img, p_CameraTag_px.astype(int), 10, (255, 0, 0), -1)
            x_axis = p_CameraTag_px + R_CameraTag @ np.array([50.0, 0.0])
            y_axis = p_CameraTag_px + R_CameraTag @ np.array([0.0, -50.0])
            cv2.line(
                annotated_img,
                p_CameraTag_px.astype(int),
                (int(x_axis[0]), int(x_axis[1])),
                (0, 0, 255),
                10,
            )
            cv2.line(
                annotated_img,
                p_CameraTag_px.astype(int),
                (int(y_axis[0]), int(y_axis[1])),
                (0, 255, 0),
                10,
            )
            cv2.putText(
                annotated_img,
                tag_name,
                p_CameraTag_px.astype(int),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        # Draw the desired box pose
        if self.desired_box_pose is not None:
            p_OBox = self.desired_box_pose[:2]
            p_OBox_Camera = R_CameraO @ p_OBox
            p_OBox_Camera_px = p_OBox_Camera / scale_m_per_px
            box_center_px = origin_px
            box_center_px[0] += p_OBox_Camera_px[0]
            box_center_px[1] -= p_OBox_Camera_px[1]
            box_center_px = np.round(box_center_px).astype(np.int32)

            cv2.circle(
                annotated_img,
                (int(box_center_px[0]), int(box_center_px[1])),
                10,
                (255, 0, 0),
                -1,
            )
            R_CameraBox = rotation_matrix_2d(self.desired_box_pose[2] + theta_CameraO)
            x_axis = box_center_px + R_CameraBox @ np.array([50.0, 0.0])
            y_axis = box_center_px + R_CameraBox @ np.array([0.0, -50.0])
            cv2.line(
                annotated_img,
                box_center_px.astype(int),
                (int(x_axis[0]), int(x_axis[1])),
                (0, 0, 255),
                10,
            )
            cv2.line(
                annotated_img,
                box_center_px.astype(int),
                (int(y_axis[0]), int(y_axis[1])),
                (0, 255, 0),
                10,
            )
            cv2.putText(
                annotated_img,
                "Desired Box Pose",
                box_center_px.astype(int),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        # Flip y axis and orientation to account for image axes
        for i in range(len(poses)):
            pose_i = poses[i]
            if pose_i is not None:
                pose_i[1:] *= -1.0
                poses[i] = pose_i

        return poses, annotated_img


def main():
    # Initialize a camera node and run it
    camera_node = CameraNode()

    while not rospy.is_shutdown():
        camera_node.step()


# main function; executes the run_turtlebot function until we hit control + C
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
