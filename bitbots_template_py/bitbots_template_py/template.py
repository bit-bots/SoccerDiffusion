#!/usr/bin/env python3

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.timer import Timer

from bitbots_template_py.bitbots_template_py_parameters import bitbots_template_py


class TemplateNode(Node):
    """
    Template Node class that print some stuff in a timer.
    """

    def __init__(self) -> None:
        node_name = "bitbots_template_py_node"
        super().__init__(node_name)
        self.get_logger().info(f"Starting {node_name}...")
        self.param_listener = bitbots_template_py.ParamListener(self)
        self.params = self.param_listener.get_params()
        self.timer: Timer = self.create_timer(1, self.log_foo_param)

    def log_foo_param(self) -> None:
        self.params = self.param_listener.get_params()
        self.get_logger().info(f"Parameter 'foo' was: '{self.params.foo}'.")


def main(args=None):
    rclpy.init(args=args)
    node = TemplateNode()
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    ex.spin()
    node.destroy_node()
    rclpy.shutdown()
