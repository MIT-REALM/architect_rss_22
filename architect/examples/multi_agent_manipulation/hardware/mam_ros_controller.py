"""A ROS node for testing the AGV navigation task in hardware."""
from typing import Optional, Tuple

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

from architect.components.geometry.transforms_2d import rotation_matrix_2d

from architect.examples.multi_agent_manipulation.mam_plotting import (
    make_turtle_patches,
    make_box_patches,
)


class SingleTurtleROSController(object):
    """ROS Controller for a single turtlebot"""

    def __init__(self, name: str, high_level_control_gains: jnp.ndarray):
        """Initialize a controller for a single turtlebot

        args:
            name: name of turtlebot (e.g. "turtle1")
            high_level_control_gains: (2,) array of linear feedback gains for the
                tracking control law.
        """
        super(SingleTurtleROSController, self).__init__()
        self.name = name

        # Extract control gains
        self.k_v, self.k_w = high_level_control_gains

        # create a publisher to send velocity commands to turtlebot
        self.command_publisher = rospy.Publisher(
            f"{name}/cmd_vel", Twist, queue_size=10
        )

        # create a subscriber to get turtlebot pose
        self.pose = jnp.zeros(3)
        rospy.Subscriber(f"{name}/pose", PoseStamped, self.pose_callback)

    def pose_callback(self, pose_msg):
        """Extract the pose from the message"""
        # Convert from quaternion to orientation angle
        quaternion = (
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w,
        )
        euler_rotation = euler_from_quaternion(quaternion)

        # Store the xy position and orientation in the pose vector
        self.pose = jnp.array(
            [pose_msg.pose.position.x, pose_msg.pose.position.y, euler_rotation[2]]
        )

    def execute_command(self, control_command: jnp.ndarray):
        """Execute the given control command by sending it to the robot.

        args:
            control_command: (2,) array of linear and angular speed commands
        """
        # Create a new Twist message to store the command
        command = Twist()

        # Load the linear and angular velocities into the command
        # NOTE: flip the sign of the linear command so we drive backwards, which
        # is needed to push the box
        command.linear.x = -control_command[0]
        command.angular.z = control_command[1]

        self.command_publisher.publish(command)

    def spline_position(
        self,
        spline_start_pt: jnp.ndarray,
        spline_control_pt: jnp.ndarray,
        spline_end_pt: jnp.ndarray,
        spline_t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate the spline with the given start, end, and control points

        args:
            spline_start_pt: the starting point of the spline
            spline_control_pt: the control point of the spline
            spline_end_pt: the end point of the spline.
            spline_t: the time at which to evaluate the curve. Must satisfy 0 <= t <= 1
        """
        spline_position = (
            (1 - spline_t) ** 2 * spline_start_pt
            + 2 * (1 - spline_t) * spline_t * spline_control_pt
            + spline_t ** 2 * spline_end_pt
        )

        return spline_position

    def track_spline(
        self,
        spline_start_pt: jnp.ndarray,
        spline_control_pt: jnp.ndarray,
        spline_end_pt: jnp.ndarray,
        spline_t: jnp.ndarray,
    ):
        """
        Evaluate a tracking controller that steers the turtlebot to follow the indicated
        position and velocity. Execute that command on the turtlebot.

        args:
            spline_start_pt: the starting point of the spline
            spline_control_pt: the control point of the spline
            spline_end_pt: the end point of the spline.
            spline_t: the time at which to evaluate the curve. Must satisfy 0 <= t <= 1
        """
        # Evaluate the desired position along the spline
        tracking_position = self.spline_position(
            spline_start_pt,
            spline_control_pt,
            spline_end_pt,
            spline_t,
        )

        # Compute the along-track and cross-track error (which are a defined relative
        # to the current track of the turtlebot, not the current track of the reference
        # path)
        turtle_position = self.pose[:2]
        position_error = turtle_position - tracking_position
        turtle_theta = self.pose[2]
        turtle_tangent = jnp.array([jnp.cos(turtle_theta), jnp.sin(turtle_theta)])
        R = rotation_matrix_2d(jnp.array(jnp.pi / 2.0))
        turtle_normal = R @ turtle_tangent
        # Along-track is positive if the turtlebot is ahead
        along_track_error = turtle_tangent.dot(position_error)
        # Cross-track is positive if the turtlebot is to the left
        cross_track_error = turtle_normal.dot(position_error)

        # v is set based on along-track error
        v = -self.k_v * along_track_error

        # w is set based on the cross-track error
        w = -self.k_w * cross_track_error

        command = jnp.stack((v, w))
        self.execute_command(command)

    def seek_pose(
        self,
        goal_pose: jnp.ndarray,
    ) -> bool:
        """
        Move the turtlebot in a straight line to the goal pose (for one control step)

        args:
            goal_pose: goal (x, y, theta)

        returns: true if the turtlebot is near the goal
        """
        # If we're not pointing towards the goal pose, turn to point towards it
        theta_OTurtle = self.pose[2]
        p_OTurtle = self.pose[:2]
        p_OG = goal_pose[:2]
        p_TurtleG_O = p_OG - p_OTurtle
        theta_GTurtle_O = jax.lax.atan2(p_TurtleG_O[1], p_TurtleG_O[0])
        error_pointing_towards_goal = theta_OTurtle - theta_GTurtle_O
        # wrap the angle error into [-pi, pi]
        error_pointing_towards_goal = jax.lax.atan2(
            jnp.sin(error_pointing_towards_goal), jnp.cos(error_pointing_towards_goal)
        )
        distance_from_goal = jnp.linalg.norm(p_TurtleG_O)

        # print("----------------------------------")
        # print(f"theta_OTurtle: {theta_OTurtle}")
        # print(f"p_OTurtle: {p_OTurtle}")
        # print(f"p_OG: {p_OG}")
        # print(f"p_TurtleG_O: {p_TurtleG_O}")
        # print(f"theta_GTurtle_O: {theta_GTurtle_O}")
        # print(f"error_pointing_towards_goal: {error_pointing_towards_goal}")
        # print(f"distance_from_goal: {distance_from_goal}")

        if jnp.abs(error_pointing_towards_goal) > 0.1 and distance_from_goal > 0.1:
            v = 0.0
            w = -1.0 * (theta_OTurtle - theta_GTurtle_O)

            command = jnp.stack((v, w))
            self.execute_command(command)
            return False

        # If we are pointing towards the goal but are too far away, then move towards
        # the goal
        if distance_from_goal > 0.05:
            v = 1.0 * distance_from_goal
            w = -1.0 * (theta_OTurtle - theta_GTurtle_O)

            command = jnp.stack((v, w))
            self.execute_command(command)
            return False

        # If we are close to the goal, turn until we meet the goal pose
        theta_GTurtle = theta_OTurtle - goal_pose[2]
        if jnp.abs(theta_GTurtle) > 0.05:
            v = 0.0
            w = -1.0 * theta_GTurtle

            command = jnp.stack((v, w))
            self.execute_command(command)
            return False

        # Otherwise, do nothing (we're at the goal!)
        v = 0.0
        w = 0.0
        command = jnp.stack((v, w))
        self.execute_command(command)
        return True

    def reverse(
        self,
    ):
        """
        Reverse the turtlebot
        """
        v = -1.0
        w = 0.0
        command = jnp.stack((v, w))
        self.execute_command(command)
        return True


class BoxROSInterface(object):
    """ROS interface for getting the pose of the box"""

    def __init__(self):
        super(BoxROSInterface, self).__init__()

        # create a subscriber to get the box pose
        self.pose = jnp.zeros(3)
        rospy.Subscriber("box/pose", PoseStamped, self.pose_callback)

    def pose_callback(self, pose_msg):
        """Extract the pose from the message"""
        # Convert from quaternion to orientation angle
        quaternion = (
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w,
        )
        euler_rotation = euler_from_quaternion(quaternion)

        # Store the xy position and orientation in the pose vector
        self.pose = jnp.array(
            [pose_msg.pose.position.x, pose_msg.pose.position.y, euler_rotation[2]]
        )


class MAMROSController(object):
    """Multi-agent Manipulation ROS Controller"""

    # State machine constants
    IDLE = 0
    SEEKING_START = 1
    AT_START = 2
    PUSHING = 3
    AT_END = 4
    BACKING_OFF = 5

    def __init__(
        self, design_params: jnp.ndarray, layer_widths: Tuple[int], control_period=0.01
    ):
        """Initialize the ROS controller for the multi-agent box pushing system

        args:
            design_params: an array containing the weights and biases of a controller
                 neural network along with the tracking controller gains.
            layer_widths: number of units in each layer. First element should be
                          15. Last element should be 6
            control_period: the duration of each time step.
        """
        super(MAMROSController, self).__init__()
        self.control_period = control_period

        # Unpack design parameters
        high_level_control_gains = design_params[:2]
        self.controller_params = []
        n_layers = len(layer_widths)
        n_turtles = 2
        assert layer_widths[0] == n_turtles * 3 + 3
        assert layer_widths[-1] == n_turtles * 2
        start_index = 2
        for i in range(1, n_layers):
            input_width = layer_widths[i - 1]
            output_width = layer_widths[i]

            num_weight_values = input_width * output_width
            weight = design_params[start_index : start_index + num_weight_values]
            weight = weight.reshape(output_width, input_width)
            start_index += num_weight_values

            num_bias_values = output_width
            bias = design_params[start_index : start_index + num_bias_values]
            start_index += num_bias_values

            self.controller_params.append((weight, bias))

        self.box_size = 0.61
        self.chassis_radius = 0.08

        # ------------------------------
        # ROS Initialization
        # ------------------------------

        # Create a ROS node (credit to Dylan Goff for this template)
        rospy.init_node("MAMROSController", anonymous=True)

        # set update rate; i.e. how often we send commands (Hz)
        self.rate = rospy.Rate(1 / self.control_period)

        # Create interfaces for both turtlebots and the box
        self.box = BoxROSInterface()
        self.turtle1 = SingleTurtleROSController("turtle1", high_level_control_gains)
        self.turtle2 = SingleTurtleROSController("turtle2", high_level_control_gains)

        # Subscribe to the annotated image
        self.cv_bridge = CvBridge()
        self.annotated_image = None
        rospy.Subscriber("overhead_image", Image, self.annotated_image_callback)

        # Publish an image with the plan
        self.plan_publisher = rospy.Publisher("plan_image", Image, queue_size=10)

        # Set up somewhere to store the plan
        self.desired_box_pose = None
        self.spline_pts: Optional[jnp.ndarray] = None
        self.push_duration = 7.0
        self.push_start_time = 0.0
        # Publish the desired box pose
        self.desired_box_pose_pub = rospy.Publisher(
            "desired_box_pose", PoseStamped, queue_size=10
        )

        # Set up state machine
        self.state = MAMROSController.IDLE

    def annotated_image_callback(self, msg):
        """Process image messages"""
        self.annotated_image = self.cv_bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough"
        )

    def step(self) -> None:
        """Execute the plan"""
        # Behavior depends on current state
        if self.state == MAMROSController.IDLE:
            # If we are idle, no need to do anything
            pass
        elif self.state == MAMROSController.SEEKING_START:
            # If we are seeking the start pose, we need to get the current box pose,
            # compute the desired starting positions in the origin frame, and seek
            # those positions using the turtlebots

            # Get box pose and compute turtle positions relative to origin
            R_OBox = rotation_matrix_2d(self.box.pose[2])
            p_BoxStartpts = jnp.array(
                [
                    [-(self.box_size / 2 + self.chassis_radius + 0.1), 0.0],
                    [0.0, -(self.box_size / 2 + self.chassis_radius + 0.1)],
                ]
            ).T
            p_OStartpts = self.box.pose[:2].reshape(2, 1) + R_OBox @ p_BoxStartpts

            # Seek those positions (add an angle goal)
            turtle1_goal = jnp.concatenate(
                (p_OStartpts[:, 0], jnp.array([self.box.pose[2]]))
            )
            turtle2_goal = jnp.concatenate(
                (p_OStartpts[:, 1], jnp.array([self.box.pose[2] + jnp.pi / 2.0]))
            )
            turtle1_at_start = self.turtle1.seek_pose(turtle1_goal)
            turtle2_at_start = self.turtle2.seek_pose(turtle2_goal)

            # Transition to the next state when both turtles are there
            if turtle1_at_start and turtle2_at_start:
                rospy.loginfo("Both turtles at start")
                self.state = MAMROSController.AT_START

        elif self.state == MAMROSController.AT_START:
            # If we are at the start, and a desired box pose has been set, and the
            # box is not yet at that desired pose, then we can plan (don't move)

            # Publish desired box pose if available
            if self.desired_box_pose is not None:
                pose_msg = PoseStamped()

                # Fill in the header
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "origin_tag"

                # Fill in the pose
                pose_msg.pose.position.x = self.desired_box_pose[0]
                pose_msg.pose.position.y = self.desired_box_pose[1]
                pose_msg.pose.position.z = 0.0

                quaternion_pose = quaternion_from_euler(
                    0.0, 0.0, self.desired_box_pose[2]
                )
                pose_msg.pose.orientation.x = quaternion_pose[0]
                pose_msg.pose.orientation.y = quaternion_pose[1]
                pose_msg.pose.orientation.z = quaternion_pose[2]
                pose_msg.pose.orientation.w = quaternion_pose[3]
                self.desired_box_pose_pub.publish(pose_msg)

            distance_to_desired = (
                0.0
                if self.desired_box_pose is None
                else jnp.linalg.norm(self.box.pose - self.desired_box_pose)
            )
            if self.desired_box_pose is not None and distance_to_desired > 0.1:
                # Convert the desired box pose from the origin frame into the current
                # box frame
                p_ODesired = self.desired_box_pose[:2]
                p_OBox = self.box.pose[:2]
                p_BoxDesired_O = p_ODesired - p_OBox
                R_OBox = rotation_matrix_2d(self.box.pose[2])
                R_BoxO = R_OBox.T  # type: ignore
                p_BoxDesired = R_BoxO @ p_BoxDesired_O.reshape(2, 1)

                desired_box_pose_relative = jnp.concatenate(
                    (
                        p_BoxDesired.reshape(2),
                        jnp.array([self.desired_box_pose[2] - self.box.pose[2]]),
                    )
                )

                # If this desired pose is not in the first quadrant, then raise an error
                if (
                    desired_box_pose_relative[0] < 0.0
                    or desired_box_pose_relative[1] < 0.0
                ):
                    rospy.logerror(
                        (
                            f"Desired box pose {self.desired_box_pose} is"
                            f" {desired_box_pose_relative}"
                            " relative to current box pose, which is not allowed "
                            "(must be first quadrant)."
                        )
                    )

                # This will save the plan in self.spline_pts
                rospy.loginfo("Planning")
                self.plan(desired_box_pose_relative)

                # Move to the next state
                self.state = MAMROSController.PUSHING
                self.push_start_time = rospy.get_time()

        elif self.state == MAMROSController.PUSHING:
            # Push until the box is close to the goal position,
            # or until some maximum time has elapsed
            distance_to_desired = jnp.linalg.norm(self.box.pose - self.desired_box_pose)
            pushing_time_elapsed = rospy.get_time() - self.push_start_time

            if (
                distance_to_desired > 0.05
                and pushing_time_elapsed <= self.push_duration
            ):
                # Normalize time
                t = pushing_time_elapsed / self.push_duration

                # Execute the plan on each turtlebot
                if self.spline_pts is None:
                    raise ValueError("self.spline_pts not set!")

                self.turtle1.track_spline(
                    self.spline_pts[0, 0, :],
                    self.spline_pts[0, 1, :],
                    self.spline_pts[0, 2, :],
                    t,
                )
                self.turtle2.track_spline(
                    self.spline_pts[1, 0, :],
                    self.spline_pts[1, 1, :],
                    self.spline_pts[1, 2, :],
                    t,
                )
            else:
                # If either of those conditions are met, then move to the next mode
                rospy.loginfo(
                    (
                        f"Pushing done! Desired box pose: {self.desired_box_pose}, "
                        f"box pose: {self.box.pose}"
                    )
                )
                self.state = MAMROSController.IDLE

        elif self.state == MAMROSController.BACKING_OFF:
            # CURRENTLY NOT USED. May be needed when we try to chain multiple pushes
            # In this state, just tell both robots to seek to a point some distance away
            # Get box pose and compute turtle positions relative to origin
            R_OBox = rotation_matrix_2d(self.box.pose[2])
            p_BoxClearancepts = jnp.array(
                [
                    [-(self.box_size / 2 + self.chassis_radius + 0.3), 0.0],
                    [0.0, -(self.box_size / 2 + self.chassis_radius + 0.3)],
                ]
            ).T
            p_OClearancepts = (
                self.box.pose[:2].reshape(2, 1) + R_OBox @ p_BoxClearancepts
            )

            # Seek those positions (add an angle goal)
            turtle1_goal = jnp.concatenate((p_OClearancepts[:, 0], jnp.array([0.0])))
            turtle2_goal = jnp.concatenate(
                (p_OClearancepts[:, 1], jnp.array([jnp.pi / 2.0]))
            )
            turtle1_done = self.turtle1.seek_pose(turtle1_goal)
            turtle2_done = self.turtle2.seek_pose(turtle2_goal)

            # Transition to the next state when both turtles are there
            if turtle1_done and turtle2_done:
                rospy.loginfo("Done!")
                self.state = MAMROSController.IDLE

        # Sleep and return
        self.rate.sleep()
        return

    def plan(self, desired_box_pose: jnp.ndarray) -> jnp.ndarray:
        """
        Plan spline control points for pushing to some final box location by evaluating
        a neural network. Uses tanh activations everywhere except the final layer.

        Saves the plan.

        args:
            desired_box_pose: (3,) array of desired x, y, and theta for the box,
                expressed in the current box frame.
        returns
            an (n_turtles, 3, 2) array of (x, y) spline points for each turtlebot.
            Structured so that [:, 0, :] are the start points, [:, 1, :] are the control
            points, and [:, 2, :] are the end points
        """
        # Get the poses of the turtlebots in the box frame.
        # Start with poses in origin frame
        p_OTurtle1 = self.turtle1.pose[:2]
        p_OTurtle2 = self.turtle2.pose[:2]
        p_OBox = self.box.pose[:2]

        # Express relative to box
        p_BoxTurtle1_O = p_OTurtle1 - p_OBox
        p_BoxTurtle2_O = p_OTurtle2 - p_OBox

        # Get 2D rotation from the origin to the box
        theta_OBox = self.box.pose[2]
        R_OBox = rotation_matrix_2d(theta_OBox)
        R_BoxO = R_OBox.T  # type: ignore

        # Get turtlebot positions in box frame
        p_BoxTurtle1 = R_BoxO @ p_BoxTurtle1_O
        theta_BoxTurtle1 = self.turtle1.pose[2] - theta_OBox
        p_BoxTurtle2 = R_BoxO @ p_BoxTurtle2_O
        theta_BoxTurtle2 = self.turtle2.pose[2] - theta_OBox

        # Combine the turtlebot poses into a big array
        turtle_poses = jnp.array(
            [
                [p_BoxTurtle1[0], p_BoxTurtle1[1], theta_BoxTurtle1],
                [p_BoxTurtle2[0], p_BoxTurtle2[1], theta_BoxTurtle2],
            ]
        )

        # Concatenate all inputs into a single feature array
        activations = jnp.concatenate((turtle_poses.reshape(-1), desired_box_pose))

        # Loop through all layers except the final one
        for weight, bias in self.controller_params[:-1]:
            activations = jnp.dot(weight, activations)
            activations = jnp.tanh(activations) + bias

        # The last layer is just linear
        weight, bias = self.controller_params[-1]
        output = jnp.dot(weight, activations) + bias
        control_pt_residual = output.reshape(2, 2)

        # Set spline start points as the current positions of the turtlebots
        spline_pts = jnp.zeros((2, 3, 2))
        spline_pts = spline_pts.at[0, 0, :].set(p_OTurtle1)
        spline_pts = spline_pts.at[1, 0, :].set(p_OTurtle2)

        # Set end points as offset from the desired box position
        R_BoxinitialBoxfinal = rotation_matrix_2d(desired_box_pose[2])
        p_BfinalEndpts = jnp.array(
            [
                [-(self.box_size / 2 + self.chassis_radius), 0.0],
                [0.0, -(self.box_size / 2 + self.chassis_radius)],
            ]
        ).T
        p_BoxinitialEndpts = (
            desired_box_pose[:2].reshape(2, 1) + R_BoxinitialBoxfinal @ p_BfinalEndpts
        )
        # Convert end points to global frame
        spline_pts = spline_pts.at[:, 2, :].set(
            (p_OBox.reshape(2, 1) + R_OBox @ p_BoxinitialEndpts).T
        )
        # Convert end points to global frame
        spline_pts = spline_pts.at[:, 2, :].set(p_OBox + R_BoxO @ p_BoxinitialEndpts)

        # Control points are set based on a learned difference from the center of the
        # line connecting the start and end points
        control_pts = (spline_pts[:, 0, :] + spline_pts[:, 2, :]) / 2.0
        control_pts += control_pt_residual
        spline_pts = spline_pts.at[:, 1, :].set(control_pts)

        # Save the plan
        self.spline_pts = spline_pts

        # Publish the plan image
        desired_box_pose_global = jnp.zeros(3)
        desired_box_pose_global = desired_box_pose_global.at[:2].set(
            p_OBox + R_OBox @ desired_box_pose[:2]
        )
        desired_box_pose_global = desired_box_pose_global.at[2].set(
            desired_box_pose[2] + theta_OBox
        )

        self.publish_plan_image(spline_pts, desired_box_pose_global)

        return spline_pts

    def publish_plan_image(
        self, spline_pts: jnp.ndarray, desired_box_pose: jnp.ndarray
    ):
        """Overlay the spline points on the annotated image and publish it

        args:
            spline_pts: (n_turtles, 3, 2) array of (x, y) spline points for each
                turtlebot. Structured so that [:, 0, :] are the start points, [:, 1, :]
                are the control points, and [:, 2, :] are the end points
            desired_box_pose: global frame (x, y, theta) desired box pose
        """
        if self.annotated_image is None:
            return

        # Create an axis on which to plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Plot the starting and desired poses of the box
        make_box_patches(desired_box_pose, 1.0, self.box_size, plt.gca(), hatch=True)
        make_box_patches(self.box.pose, 1.0, self.box_size, plt.gca(), hatch=False)

        # Plot the starting pose of the turtlebots
        make_turtle_patches(self.turtle1.pose, 1.0, self.chassis_radius, ax)
        make_turtle_patches(self.turtle2.pose, 1.0, self.chassis_radius, ax)

        # Plot the splines
        spline1_fn = lambda t: self.turtle1.spline_position(
            spline_pts[0, 0, :], spline_pts[0, 1, :], spline_pts[0, 2, :], t
        )
        spline2_fn = lambda t: self.turtle2.spline_position(
            spline_pts[1, 0, :], spline_pts[1, 1, :], spline_pts[1, 2, :], t
        )

        t = jnp.linspace(0, 1.0, 100)
        spline1 = jax.vmap(spline1_fn, in_axes=0)(t)
        spline2 = jax.vmap(spline2_fn, in_axes=0)(t)

        ax.plot(spline1[:, 0], spline1[:, 1], "k--", linewidth=3)
        ax.plot(spline2[:, 0], spline2[:, 1], "k--", linewidth=3)
        ax.plot(spline_pts[0, :, 0], spline_pts[0, :, 1], "ko:", markersize=10)
        ax.plot(spline_pts[1, :, 0], spline_pts[1, :, 1], "ko:", markersize=10)

        # Format the plot
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$z$")
        ax.set_aspect("equal")

        # Render the plot to numpy array
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = (int(dim) for dim in fig.canvas.get_width_height())
        plan_image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
            width, height, 3
        )

        # Publish the image
        image_message = self.cv_bridge.cv2_to_imgmsg(plan_image, encoding="rgb8")
        self.plan_publisher.publish(image_message)


def main():
    # Initialize a camera node and run it
    logfile = (
        "logs/multi_agent_manipulation/real_turtle_dimensions/"
        "design_optimization_512_samples_0p5x0p5xpi_4_target_"
        "9x32x4_network_spline_1e-1_variance_weight_solution.csv"
    )
    design_param_values = jnp.array(
        np.loadtxt(
            logfile,
            delimiter=",",
        )
    )
    layer_widths = (2 * 3 + 3, 32, 2 * 2)
    mam_controller_node = MAMROSController(
        design_param_values, layer_widths, control_period=0.01
    )

    sleep_time = 0.5
    for t in range(int(sleep_time / mam_controller_node.control_period)):
        mam_controller_node.step()

    mam_controller_node.state = MAMROSController.SEEKING_START
    mam_controller_node.desired_box_pose = jnp.array([1.3, 1.0, 0.0])

    run_time = 25.0
    for t in range(int(run_time / mam_controller_node.control_period)):
        mam_controller_node.step()


# main function; executes the run_turtlebot function until we hit control + C
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
