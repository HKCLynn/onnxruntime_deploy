cmake_minimum_required(VERSION 3.10)

project(test)

set(ONNXRUNTIME_ROOT_PATH /usr/local)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session/
                             ${ONNXRUNTIME_ROOT_PATH}/core/providers/cuda/)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so)

find_package(OpenCV REQUIRED)

add_executable(test main.cpp)
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

target_link_libraries(test PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})


