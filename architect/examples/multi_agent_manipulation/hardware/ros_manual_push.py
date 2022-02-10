import rospy
from geometry_msgs.msg import Twist


if __name__ == "__main__":
    # Create a ROS node (credit to Dylan Goff for this template)
    rospy.init_node("MAMROSController", anonymous=True)

    # set update rate; i.e. how often we send commands (Hz)
    control_period = 0.1
    rate = rospy.Rate(1 / control_period)

    # Turtle command publishers
    command_publisher_1 = rospy.Publisher("turtle1/cmd_vel", Twist, queue_size=10)
    command_publisher_2 = rospy.Publisher("turtle2/cmd_vel", Twist, queue_size=10)

    # Sleep for 2 seconds to connect everything
    for i in range(int(2 / control_period)):
        rate.sleep()

    # Push for 2 seconds
    cmd = Twist()
    cmd.linear.x = -0.1
    for i in range(int(4 / control_period)):
        command_publisher_1.publish(cmd)
        command_publisher_2.publish(cmd)
        rate.sleep()
