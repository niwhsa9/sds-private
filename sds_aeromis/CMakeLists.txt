cmake_minimum_required(VERSION 3.8)
project(sds_aeromis)
enable_language(CUDA)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
set(CMAKE_CUDA_ARCHITECTURES native) 
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../sds_core ${CMAKE_BINARY_DIR}/sds_core)
find_package(ament_cmake REQUIRED)
find_package(aeromis_core REQUIRED)
find_package(rclcpp REQUIRED)
find_package(aeromis_msgs REQUIRED)
#find_package(aeromis_interfaces REQUIRED)
#find_package(aeromis_estimators REQUIRED)
find_package(sds_core CONFIG REQUIRED)

add_library(sds_aeromis STATIC src/aerial_vehicle.cu)
target_include_directories(sds_aeromis
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
        ${S}
)
ament_target_dependencies(sds_aeromis
  rclcpp
  aeromis_core
  aeromis_msgs
)

set_target_properties(sds_aeromis PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  #CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(sds_aeromis sds_core::sds_core)
install(
  TARGETS sds_aeromis
  EXPORT export_sds_aeromis
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_targets(export_sds_aeromis HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp
  aeromis_core
  aeromis_msgs
)

#set_target_properties(sds_core PROPERTIES
#  CUDA_SEPARABLE_COMPILATION ON
#)


add_executable(uav_demo example/uav_perch_demo.cu src/aerial_vehicle.cu)
set_target_properties(uav_demo PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(uav_demo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${AEROMIS_INCLUDE_DIRS}
  )
ament_target_dependencies(uav_demo
  rclcpp
  aeromis_core
  aeromis_msgs

  #aeromis_interfaces
  #aeromis_estimators
)
target_link_libraries(uav_demo sds_core::sds_core )
  target_compile_options(uav_demo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

install(TARGETS uav_demo DESTINATION lib/${PROJECT_NAME})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  add_executable(aerial_vehicle_integrator_test test/aerial_vehicle_integrator_test.cu)
  target_link_libraries(aerial_vehicle_integrator_test sds_aeromis)
  set_target_properties(aerial_vehicle_integrator_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_test(NAME aerial_vehicle_integrator_test COMMAND aerial_vehicle_integrator_test)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
