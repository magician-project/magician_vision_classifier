cmake_minimum_required(VERSION 3.5)
project(magician_vision_classifier)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
#find_package(service_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection.msg"
  "msg/DetectionM.msg"
  "srv/SetInt64.srv"
  "srv/SetFloat64.srv"
  DEPENDENCIES std_msgs geometry_msgs #service_msgs
)


ament_export_dependencies(rosidl_default_runtime)
ament_package()

