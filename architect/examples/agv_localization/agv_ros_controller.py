"""A ROS node for testing the AGV navigation task in hardware."""
from copy import copy
from typing import Optional, Tuple

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, LaserScan
import tf
from tf.transformations import euler_from_quaternion

import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import scipy.spatial

from architect.components.dynamics.dubins import (
    dubins_next_state,
    dubins_linearized_dynamics,
)
from architect.components.estimation.ekf import dt_ekf_predict_covariance, dt_ekf_update
from architect.components.geometry.transforms_2d import rotation_matrix_2d
from architect.components.sensing.range_beacons import (
    beacon_range_measurements,
)
from architect.examples.agv_localization.agv_simulator import (
    navigate,
    navigation_function,
    get_observations,
    get_observations_jacobian,
)


class AGVROSController(object):
    """docstring for AGVROSController"""

    def __init__(self, design_params: jnp.ndarray, control_period=0.5):
        super(AGVROSController, self).__init__()
        self.control_period = control_period

        # Set a random seed for repeatability
        self.prng_key = jax.random.PRNGKey(0)

        # Unpack design parameters
        self.control_gains = design_params[:2]
        self.beacon_locations = design_params[2:]
        n_beacons = self.beacon_locations.shape[0] // 2
        self.beacon_locations = self.beacon_locations.reshape(n_beacons, 2)

        # ------------------------------
        # ROS Initialization
        # ------------------------------

        # Create a ROS node (credit to Dylan Goff for this template)
        rospy.init_node("AGVROSController", anonymous=True)

        # set update rate; i.e. how often we send commands (Hz)
        self.rate = rospy.Rate(1 / self.control_period)

        # create transform listener to get turtlebot transform from odometry
        self.listener = tf.TransformListener()
        self.odom_frame = "/odom"

        # create a publisher node to send velocity commands to turtlebot
        self.command_publisher = rospy.Publisher("cmd_vel", Twist, queue_size=10)

        # Subscribe to the IMU topic to get the heading angle
        self.heading_angle = None
        self.heading_offset = None
        rospy.Subscriber("/imu", Imu, self.imu_callback)

        # Subscribe to the lidar scan to get the xy position
        # (from which we fake beacons)
        self.ranges = None
        self.angles = None
        self.p_WT = None
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)

        # Make sure to stop the turtlebot using a callback when this node gets shut down
        rospy.on_shutdown(self.on_shutdown)

        # Find the coordinate conversion from the turtlebot to the ground truth frame
        # First try one frame name, and if that doesn't work try another one.
        self.base_frame = "base_footprint"
        try:
            self.listener.waitForTransform(
                self.odom_frame, self.base_frame, rospy.Time(), rospy.Duration(1.0)
            )
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            self.base_frame = "base_link"

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

        # Get the starting odometry measurement and find an offset that puts it at
        # the intended starting location
        (trans, rot) = self.listener.lookupTransform(
            self.odom_frame, self.base_frame, rospy.Time(0)
        )
        rotation = euler_from_quaternion(rot)
        initial_odometry_state = jnp.array([trans[0], trans[1], rotation[2]])
        initial_state_mean = jnp.array([-2.2, 0.5, 0.0])
        initial_state_covariance = 0.001 * jnp.eye(3)
        self.state_offset = initial_odometry_state - initial_state_mean

        # Initialize the state estimate and error covariance for the EKF
        self.state_est_mean = initial_state_mean
        self.state_est_cov = initial_state_covariance

        # Save the covariance matrices for actuation and observations noise
        # (these are used to update the EKF as we go)
        self.observation_noise_covariance = jnp.diag(jnp.array([0.1, 0.01, 0.01]))
        self.actuation_noise_covariance = self.control_period ** 2 * jnp.diag(
            jnp.array([0.001, 0.001, 0.01])
        )

        # Create a dataframe for logging the results of running this experiment
        self.log_df = pd.DataFrame()

    def on_shutdown(self) -> None:
        """Stop the turtlebot and save the log on shutdown"""
        self.command_publisher.publish(Twist())
        rospy.sleep(1.0)

        self.log_df.to_csv(f"agv_controller_log.csv")

        # Also plot the data to see what happened

    def step(self) -> None:
        """
        Evaluate one step of the AGV localization, navigation, and control system.
        """
        # Get the current observations
        observations = self.get_observations()
        # # Update the state estimate using the EKF
        # if observations is not None:
        #     self.state_est_mean, self.state_est_cov = self.ekf_update(observations)

        # Get the control command based on the current state estimate
        # (this also returns the navigation function value at this point)
        control_command, V = self.compute_control_command()

        # Log the state estimate, control command, and navigation function value
        # Use the tidy data format
        base_log_packet = {"time": rospy.get_time()}
        log_packet = copy(base_log_packet)
        log_packet["measurement"] = "state_est_mean"
        log_packet["value"] = self.state_est_mean.tolist()
        self.log_df = self.log_df.append(log_packet, ignore_index=True)
        log_packet["measurement"] = "state_est_cov"
        log_packet["value"] = self.state_est_cov.tolist()
        self.log_df = self.log_df.append(log_packet, ignore_index=True)
        log_packet["measurement"] = "control_command"
        log_packet["value"] = control_command.tolist()
        self.log_df = self.log_df.append(log_packet, ignore_index=True)
        log_packet["measurement"] = "V"
        log_packet["value"] = V
        self.log_df = self.log_df.append(log_packet, ignore_index=True)

        # # Predict the next state
        # self.state_est_mean, self.state_est_cov = self.ekf_predict(control_command)

        # Execute the control command and hold for the desired frequency
        self.execute_command(control_command)
        self.rate.sleep()

    def imu_callback(self, data):
        # Extract the orientation and convert it to a heading angle
        orientation_quat = data.orientation
        orientation_quat = [
            orientation_quat.x,
            orientation_quat.y,
            orientation_quat.z,
            orientation_quat.w,
        ]
        orientation_euler = euler_from_quaternion(orientation_quat)

        # If we have not saved an offset, save this reading as the offset
        if self.heading_offset is None:
            self.heading_offset = orientation_euler[2]

        # Save the heading angle accounting for the offset
        self.heading_angle = orientation_euler[2] - self.heading_offset

    def scan_callback(self, data):
        # We don't actually have beacons installed, so we have to fake them by
        # using Lidar to find our location in a rectangular arena and use that
        # state to compute what the beacon measurement would be (with noise)

        # Start by getting the lidar measurements
        angles = jnp.arange(
            data.angle_min, data.angle_max + data.angle_increment, data.angle_increment
        )
        ranges = jnp.array(data.ranges)

        # Get valid points
        valid = ranges > 0.0
        angles = angles[valid]
        ranges = ranges[valid]

        self.angles = angles
        self.ranges = ranges

        # Convert to cartesian
        x = self.ranges * jnp.cos(self.angles)
        y = self.ranges * jnp.sin(self.angles)

        # Rotate to world frame using the estimated angle
        try:
            R_WT = rotation_matrix_2d(self.state_est_mean[2])
            R_TW = R_WT.T
        except AttributeError:
            return  # happens if called before state_est_mean is set

        p_TurtleLidar = jnp.vstack((x, y))
        p_WLidar = R_TW @ p_TurtleLidar
        p_WLidar = p_WLidar.T

        # Get the upper right corner of the box relative to the turtle
        p_TCorner_W = jnp.array([p_WLidar[:, 0].max(), p_WLidar[:, 1].max()])

        # Hand-measured from box origin to corner
        p_WCorner = jnp.array([0.38, 0.83])

        self.p_WT = p_WCorner - p_TCorner_W

    def get_observations(self) -> Optional[jnp.ndarray]:
        """Get the observations from beacons. We don't actually have radio beacons
        in the lab, so simulate them

        returns:
            (self.beacon_locations.shape[0] + 1,) vector of ranges to beacons, or None
            if no data is available.
        """
        # Fake the beacon measurements by getting the state from odometry
        (trans, rot) = self.listener.lookupTransform(
            self.odom_frame, self.base_frame, rospy.Time(0)
        )
        rotation = euler_from_quaternion(rot)
        odometry_state = jnp.array([trans[0], trans[1], rotation[2]])
        odometry_state_corrected = odometry_state - self.state_offset
        self.state_est_mean = odometry_state_corrected
        print(self.state_est_mean)

        ranges = beacon_range_measurements(
            odometry_state_corrected[:2],
            self.beacon_locations,
            jnp.zeros(self.beacon_locations.shape[0]),
        )

        heading = jnp.array([rotation[2]])

        return jnp.concatenate((heading, ranges))

    def compute_control_command(self) -> Tuple[jnp.ndarray, float]:
        """Compute the control command and navigation function value given the current
        state estimate (stored as self.state_est_mean)

        returns:
            (2,) array of forward and angular speed commands
            float of navigation function value
        """
        control_command = navigate(self.state_est_mean, self.control_gains)
        V = navigation_function(self.state_est_mean[:2]).item()
        return control_command, V

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

    def ekf_update(self, observations: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update the state estimate given the observations.

        args:
            observations: (self.beacon_locations.shape[0] + 1,) vector of ranges to
                beacons
        returns:
            new state estimate mean
            new state estimate covariance
        """
        # Get the expected observations and observations jacobian at the expected state
        expected_observations = get_observations(
            self.state_est_mean,
            self.beacon_locations,
            jnp.zeros(self.beacon_locations.shape[0] + 1),
        )
        observations_jacobian = get_observations_jacobian(
            self.state_est_mean, self.beacon_locations
        )

        # Update
        return dt_ekf_update(
            self.state_est_mean,
            self.state_est_cov,
            observations,
            expected_observations,
            observations_jacobian,
            self.observation_noise_covariance,
        )

    def ekf_predict(
        self, control_command: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predict the next state given the control commands.

        args:
            control_command: (2,) array of linear and angular speed commands
        """
        # Update the state estimate and covariance with the dynamics
        new_state_est_mean = dubins_next_state(
            self.state_est_mean,
            control_command,
            jnp.zeros(self.state_est_mean.shape),
            self.control_period,
        )
        dynamics_jac = dubins_linearized_dynamics(
            self.state_est_mean, control_command, self.control_period
        )
        new_state_est_cov = dt_ekf_predict_covariance(
            self.state_est_cov,
            dynamics_jac,
            self.actuation_noise_covariance,
            self.control_period,
        )

        return new_state_est_mean, new_state_est_cov


def main():
    # Initialize a controller and run it
    design_params = jnp.array([2.535058, 0.09306894, -1.6945883, -1.0, 0.0, -0.8280163])
    dt = 0.5
    controller = AGVROSController(design_params, dt)

    while not rospy.is_shutdown():
        controller.step()


# main function; executes the run_turtlebot function until we hit control + C
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
