#include <bitbots_template_cpp/template.hpp>

namespace bitbots_template_cpp {

Template::Template() : Node("bitbots_template_cpp_node"), param_listener_(get_node_parameters_interface()) {
  RCLCPP_INFO(this->get_logger(), "Starting bitbots_template_cpp_node...");

  // get parameters for generated code
  config_ = param_listener_.get_params();
}

void Template::log_foo_param() {
  // Update parameters
  config_ = param_listener_.get_params();

  // Log value of parameter foo
  RCLCPP_INFO(this->get_logger(), "Parameter 'foo' was: '%s'.", config_.foo.c_str());
}
}  // namespace bitbots_template_cpp

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<bitbots_template_cpp::Template>();

  rclcpp::Duration timer_duration = rclcpp::Duration::from_seconds(1.0);
  rclcpp::TimerBase::SharedPtr timer =
      rclcpp::create_timer(node, node->get_clock(), timer_duration, [node]() -> void { node->log_foo_param(); });
  rclcpp::experimental::executors::EventsExecutor exec;
  exec.add_node(node);

  exec.spin();
  rclcpp::shutdown();
}
