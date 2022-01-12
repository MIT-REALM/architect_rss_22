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


class SingleTurtleROSController(object):
    """ROS Controller for a single turtlebot"""
    def __init__(self, name: str):
        super(SingleTurtleROSController, self).__init__()
        self.name = name

        # create transform listener to get turtlebot transform from odometry
        self.listener = tf.TransformListener()
        self.odom_frame = f"/{name}/odom"

        # create a publisher node to send velocity commands to turtlebot
        self.command_publisher = rospy.Publisher(f"/{name}/cmd_vel", Twist, queue_size=10)

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
        

class MAMROSController(object):
    """Multi-agent Manipulation ROS Controller"""

    def __init__(self, design_params: jnp.ndarray, control_period=0.1):
        super(MAMROSController, self).__init__()
        self.control_period = control_period

        # Unpack design parameters
        high_level_control_gains = design_params[:2]
        controller_params = []
        n_layers = len(layer_widths)
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

            controller_params.append((weight, bias))

        # ------------------------------
        # ROS Initialization
        # ------------------------------

        # Create a ROS node (credit to Dylan Goff for this template)
        rospy.init_node("MAMROSController", anonymous=True)

        # set update rate; i.e. how often we send commands (Hz)
        self.rate = rospy.Rate(1 / self.control_period)
