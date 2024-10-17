#include <bitbots_template_cpp_parameters.hpp>
#include <rclcpp/experimental/executors/events_executor/events_executor.hpp>
#include <rclcpp/rclcpp.hpp>

namespace bitbots_template_cpp {

class Template : public rclcpp::Node {
 public:
  Template();
  void log_foo_param();

 private:
  // Declare parameter listener and struct from the generate_parameter_library
  bitbots_template_cpp::ParamListener param_listener_;
  // Data structure to hold all parameters, which is build from the schema in the 'parameters.yaml'
  bitbots_template_cpp::Params config_;
};

}  // namespace bitbots_template_cpp
