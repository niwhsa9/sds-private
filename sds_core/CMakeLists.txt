cmake_minimum_required(VERSION 3.7)
project(sds_core LANGUAGES CXX CUDA)
#cmake_policy(SET CMP0057 NEW)
enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 20)


find_package(Eigen3 REQUIRED)
#set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

add_library(sds_core STATIC src/sds_core.cu)
#add_library(sds_core INTERFACE)
target_include_directories(sds_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(sds_core PUBLIC Eigen3::Eigen)
#target_link_libraries(sds_core INTERFACE cublas)

set_target_properties(sds_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

include(CMakePackageConfigHelpers)

install(
    TARGETS sds_core
    EXPORT sds_coreTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(
    DIRECTORY include/
    DESTINATION include
)

install(
    EXPORT sds_coreTargets
    FILE sds_coreTargets.cmake
    NAMESPACE sds_core::
    DESTINATION lib/cmake/sds_core
)

configure_package_config_file(
    cmake/sds_coreConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sds_coreConfig.cmake
    INSTALL_DESTINATION lib/cmake/sds_core
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/sds_coreConfigVersion.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/sds_coreConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/sds_coreConfigVersion.cmake
    DESTINATION lib/cmake/sds_core
)



add_executable(dubin_cem example/dubin_cem.cu)
#target_include_directories(dubin_cem PRIVATE include)
#target_include_directories(dubin_cem SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(dubin_cem sds_core)

target_compile_options(dubin_cem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)


#set(PYBIND11_FINDPYTHON ON)
#find_package(pybind11 CONFIG REQUIRED)
#find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
#pybind11_add_module(vortex_pybinds src/pybinds.cu src/vortex_model.cu src/flat_plate_glider.cu src/tvlqr.cu src/mppi.cu)
#target_link_libraries(vortex_pybinds PRIVATE cublas)
#target_include_directories(vortex_pybinds SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})
#target_include_directories(vortex_pybinds PRIVATE include)
#set(Stubgen_Executable "/usr/bin/stubgen")
#add_custom_command(TARGET vortex_pybinds POST_BUILD
#    COMMAND ${Stubgen_Executable} -m vortex_pybinds -o .
#    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#    COMMENT "Use stubgen to create .pyi for statement completion")
#
#set(PYI_FILE "${CMAKE_CURRENT_BINARY_DIR}/vortex_pybinds.pyi")


#install(TARGETS vortex_pybinds 
#        LIBRARY DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)
#execute_process(
#  COMMAND ${Python3_EXECUTABLE} -c
#  "import site; print(site.getsitepackages()[0])"
#  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
#  OUTPUT_STRIP_TRAILING_WHITESPACE
#)
#install(TARGETS vortex_pybinds
#        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})

#if(EXISTS "${PYI_FILE}")
#    #install(FILES "${PYI_FILE}"
#    #    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
#    #)
#    #install(FILES "${PYI_FILE}"
#    #    DESTINATION ${PYTHON_SITE_PACKAGES}
#    #)
#else()
#    message(WARNING "PYI file ${PYI_FILE} not found, skipping installation")
#endif()
#
include(CTest)
if(BUILD_TESTING)
  add_executable(infinite_horizon_lqr_test test/infinite_horizon_lqr.cu)
  target_link_libraries(infinite_horizon_lqr_test sds_core)
  add_test(NAME infinite_horizon_lqr COMMAND infinite_horizon_lqr_test)
endif()
