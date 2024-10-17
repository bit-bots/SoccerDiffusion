cmake_minimum_required(VERSION 3.5)
project(bitbots_template_cpp)

# Add support for C++17
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif ()

find_package(ament_cmake REQUIRED)
find_package(bitbots_docs REQUIRED)
find_package(generate_parameter_library REQUIRED)
find_package(rclcpp REQUIRED)

generate_parameter_library(
  bitbots_template_cpp_parameters
  config/bitbots_template_cpp_parameters.yaml
)

include_directories(include)

add_compile_options(-Wall -Werror -Wno-unused)

add_executable(template src/template.cpp)

target_link_libraries(
  template
  rclcpp::rclcpp
  bitbots_template_cpp_parameters
)

## Specify libraries to link a library or executable target against
ament_target_dependencies(
  template
  ament_cmake
  bitbots_docs
  generate_parameter_library
  rclcpp
)

enable_bitbots_docs()

install(TARGETS template
        DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
        DESTINATION share/${PROJECT_NAME})

ament_package()
