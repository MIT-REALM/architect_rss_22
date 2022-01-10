"""A ROS node for testing the AGV navigation task in hardware."""
from copy import copy
from typing import Optional, Tuple

import rospy
from geometry_msgs.msg import Twist
import tf
from tf.transformations import euler_from_quaternion

import jax
import jax.numpy as jnp
import pandas as pd

from architect.components.dynamics.dubins import (
    dubins_next_state,
    dubins_linearized_dynamics,
)
from architect.components.estimation.ekf import dt_ekf_predict_covariance, dt_ekf_update
from .agv_simulator import (
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

        # When the turtlebot wakes up, it's odometry thinks that it is at the origin.
        # We can fake starting at some random position by adding a random offset
        initial_state_mean = jnp.array([-2.2, 0.5, 0.0])
        initial_state_covariance = 0.001 * jnp.eye(3)
        self.prng_key, subkey = jax.random.split(self.prng_key)
        self.state_offset = jax.random.multivariate_normal(
            subkey,
            initial_state_mean,
            initial_state_covariance,
        )

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

        self.log_df.to_csv(f"agv_controller_log_{round(rospy.get_time(), 2)}.csv")

    def step(self) -> None:
        """
        Evaluate one step of the AGV localization, navigation, and control system.
        """
        # Get the current observations
        observations = self.get_observations()
        # Update the state estimate using the EKF
        if observations is not None:
            self.state_est_mean, self.state_est_cov = self.ekf_update(observations)

        # Get the control command based on the current state estimate
        # (this also returns the navigation function value at this point)
        control_command, V = self.compute_control_command()

        # Log the state estimate, control command, and navigation function value
        # Use the tidy data format
        base_log_packet = {"time": rospy.get_time()}
        log_packet = copy(base_log_packet)
        log_packet["measurement"] = "state_est_mean"
        log_packet["value"] = self.state_est_mean
        self.log_df = self.log_df.append(log_packet, ignore_index=True)
        log_packet["measurement"] = "state_est_cov"
        log_packet["value"] = self.state_est_cov
        self.log_df = self.log_df.append(log_packet, ignore_index=True)
        log_packet["measurement"] = "control_command"
        log_packet["value"] = control_command
        self.log_df = self.log_df.append(log_packet, ignore_index=True)
        log_packet["measurement"] = "V"
        log_packet["value"] = V
        self.log_df = self.log_df.append(log_packet, ignore_index=True)

        # Predict the next state
        self.state_est_mean, self.state_est_cov = self.ekf_predict(control_command)

        # Execute the control command and hold for the desired frequency
        self.execute_command(control_command)
        self.rate.sleep()

    def get_observations(self) -> Optional[jnp.ndarray]:
        """Get the observations from beacons. We don't actually have radio beacons
        in the lab, so simulate them

        returns:
            (self.beacon_locations.shape[0] + 1,) vector of ranges to beacons, or None
            if no data is available.
        """
        # Get the location of the robot as estimated by the latest available odometry
        try:
            (trans, rot) = self.listener.lookupTransform(
                self.odom_frame, self.base_frame, rospy.Time(0)
            )
            rotation = euler_from_quaternion(rot)
        except (tf.Exception, tf.ConnectivityException, tf.LookupException) as e:
            rospy.loginfo(f"tf Exception:\n{e}")
            return None

        state = jnp.array([trans[0], trans[1], rotation[2]])

        return get_observations(
            state, self.beacon_locations, jnp.zeros(self.beacon_locations.shape[0] + 1)
        )

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
