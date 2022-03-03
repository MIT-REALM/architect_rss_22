"""A ROS node for testing the turtlebot rendezvous task in hardware."""
import os
from copy import copy

import rospy
from geometry_msgs.msg import Twist
import tf
from tf.transformations import euler_from_quaternion

import jax.numpy as jnp
import numpy as np
import pandas as pd

from architect.components.geometry.transforms_2d import rotation_matrix_2d


class TBSTLROSController(object):
    """Turtlebot rendezvous ROS controller"""

    def __init__(
        self, design_params: jnp.ndarray, control_period=0.1, step_period=0.02
    ):
        super(TBSTLROSController, self).__init__()
        self.control_period = control_period
        self.step_period = step_period

        self.planned_trajectory = design_params.reshape(-1, 2)

        # ------------------------------
        # ROS Initialization
        # ------------------------------

        # Create a ROS node (credit to Dylan Goff for this template)
        rospy.init_node("TBSTLROSController", anonymous=True)

        # set update rate; i.e. how often we send commands (Hz)
        self.rate = rospy.Rate(1 / self.step_period)

        # create transform listener to get turtlebot transform from odometry
        self.listener = tf.TransformListener()
        self.odom_frame = "turtle1/odom"

        # create a publisher node to send velocity commands to turtlebot
        self.command_publisher = rospy.Publisher(
            "turtle1/cmd_vel", Twist, queue_size=10
        )

        # Make sure to stop the turtlebot using a callback when this node gets shut down
        rospy.on_shutdown(self.on_shutdown)

        # Find the coordinate conversion from the turtlebot to the ground truth frame
        # First try one frame name, and if that doesn't work try another one.
        self.base_frame = "turtle1/base_footprint"
        try:
            self.listener.waitForTransform(
                self.odom_frame, self.base_frame, rospy.Time(), rospy.Duration(1.0)
            )
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            self.base_frame = "turtle1/base_link"

        try:
            self.listener.waitForTransform(
                self.odom_frame, self.base_frame, rospy.Time(), rospy.Duration(1.0)
            )
        except (tf.Exception, tf.ConnectivityException, tf.LookupException) as e:
            rospy.loginfo(
                "No transform found between odom and either base_link or base_footprint"
            )
            rospy.signal_shutdown(f"tf Exception:\n{e}")

        # ------------------------------
        # Navigation Initialization
        # ------------------------------

        self.state_est_mean = jnp.array([-1.5, -1.5, jnp.pi / 2])

        # Get the starting odometry measurement and find an offset that puts it at
        # the intended starting location
        (trans, rot) = self.listener.lookupTransform(
            self.odom_frame, self.base_frame, rospy.Time(0)
        )
        rotation = euler_from_quaternion(rot)
        self.R_MapWorld = rotation_matrix_2d(rotation[2])
        self.R_WorldMap = self.R_MapWorld.T  # type: ignore
        p_MapStart = jnp.array([trans[0], trans[1]]).reshape(2, 1)
        p_WorldStart_W = self.state_est_mean[:2].reshape(2, 1)
        p_StartWorld_W = -p_WorldStart_W
        self.p_MapWorld = p_MapStart + self.R_MapWorld @ p_StartWorld_W
        self.theta_MapWorld = rotation[2]

        # Create a dataframe for logging the results of running this experiment
        self.log_df = pd.DataFrame()

        # Get the start time of the trajectory
        self.start_time = rospy.get_time()

    def on_shutdown(self) -> None:
        """Stop the turtlebot and save the log on shutdown"""
        self.command_publisher.publish(Twist())
        rospy.sleep(1.0)

        save_dir = "logs/turtle_stl/all_constraints/hw"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/tbstl_ros_controller_log.csv"
        self.log_df.to_csv(filename)

    def step(self) -> None:
        """
        Evaluate one step of the controller
        """
        # Get the current state estimate from odometry
        self.get_odometry_estimate()

        # Get the current control input
        current_time = rospy.get_time()
        timestep = int((current_time - self.start_time) // self.control_period)
        if timestep > self.planned_trajectory.shape[0]:
            control_command = jnp.zeros(2)
        else:
            control_command = self.planned_trajectory[timestep]

        # Log the state estimate, control command, and navigation function value
        # Use the tidy data format
        base_log_packet = {"time": rospy.get_time()}
        log_packet = copy(base_log_packet)
        log_packet["measurement"] = "state_est_mean"
        log_packet["value"] = self.state_est_mean.tolist()
        self.log_df = self.log_df.append(log_packet, ignore_index=True)
        log_packet["measurement"] = "control_command"
        log_packet["value"] = control_command.tolist()
        self.log_df = self.log_df.append(log_packet, ignore_index=True)

        # Execute the control command and hold for the desired frequency
        self.execute_command(control_command)
        self.rate.sleep()

    def get_odometry_estimate(self):
        """Get the estimated location from odometry"""
        # Fake the beacon measurements by getting the state from odometry
        (trans, rot) = self.listener.lookupTransform(
            self.odom_frame, self.base_frame, rospy.Time(0)
        )
        rotation = euler_from_quaternion(rot)
        theta_MapTurtle = rotation[2]

        # Spatial algebra!
        p_MapTurtle = jnp.array([trans[0], trans[1]]).reshape(2, 1)
        p_WorldTurtle_Map = p_MapTurtle - self.p_MapWorld
        p_WorldTurtle = self.R_WorldMap @ p_WorldTurtle_Map
        theta_WorldTurtle = theta_MapTurtle - self.theta_MapWorld

        self.state_est_mean = jnp.array(
            [p_WorldTurtle[0, 0], p_WorldTurtle[1, 0], theta_WorldTurtle]
        )

    def execute_command(self, control_command: jnp.ndarray):
        """Execute the given control command by sending it to the robot.

        args:
            control_command: (2,) array of linear and angular speed commands
        """
        # Create a new Twist message to store the command
        command = Twist()

        # Load the linear and angular velocities into the command
        command.linear.x = control_command[0]
        command.angular.z = control_command[1]

        self.command_publisher.publish(command)


def main():
    # Initialize a controller and run it
    design_params = jnp.array(
        np.loadtxt(
            "logs/turtle_stl/all_constraints/solutions/counterexample_guided_0.csv",
            delimiter=",",
        )
    )
    dt = 0.1
    controller = TBSTLROSController(design_params, dt)
    print("Controller initialized")

    while not rospy.is_shutdown():
        controller.step()


# main function; executes the run_turtlebot function until we hit control + C
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
