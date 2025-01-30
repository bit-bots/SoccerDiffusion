from typing import Optional

import rclpy
import torch.nn.functional as F  # noqa
from bitbots_msgs.msg import JointCommand
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ddlitlab2024 import DEFAULT_RESAMPLE_RATE_HZ


class TrajectoryPlayer(Node):
    def __init__(self, node_name, context):
        super().__init__(node_name, context=context)
        # Activate sim time
        self.get_logger().info("Activate sim time")
        self.set_parameters(
            [rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)],
        )

        # Subscribers
        self.trajectory_subscriber = self.create_subscription(
            JointTrajectory,
            "traj",
            self.trajectory_callback,
            10,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.joint_command_publisher = self.create_publisher(JointCommand, "DynamixelController/command", 10)

        # Current trajectory
        self.current_trajectory = None

        self.create_timer(1 / DEFAULT_RESAMPLE_RATE_HZ, self.timer_callback)

    def trajectory_callback(self, msg: JointTrajectory):
        self.current_trajectory = msg

    def timer_callback(self):
        if self.current_trajectory is None:
            self.get_logger().info("No trajectory available")
            return

        # Get the current time
        current_time = self.get_clock().now()

        # Find the current point based on the current time, the time the trajectory created and the time of the points
        current_point: Optional[JointTrajectoryPoint] = None
        for i, point in enumerate(self.current_trajectory.points):
            time_of_point = Time.from_msg(self.current_trajectory.header.stamp) + Duration.from_msg(
                point.time_from_start
            )
            if time_of_point < current_time:
                current_point = point
                a = i

        print(
            f"Using point {a} at time {time_of_point} as current point of the "
            f"trajectory starting at {Time.from_msg(self.current_trajectory.header.stamp)}"
        )

        if current_point is None:
            self.get_logger().info("Trajectory is in the future")
            return

        # Publish the current point
        self.joint_command_publisher.publish(
            JointCommand(
                joint_names=self.current_trajectory.joint_names,
                velocities=current_point.velocities,
                accelerations=current_point.accelerations,
                max_currents=current_point.effort,
                positions=current_point.positions,
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlayer("traj_player", None)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
