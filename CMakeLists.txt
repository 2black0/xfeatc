cmake_minimum_required(VERSION 3.26)
project(xfeatc)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependencies
find_package(OpenCV REQUIRED)

# --- ONNX Runtime section ---
# Expect ONNXRUNTIME_ROOT to be provided via CMake command line, e.g.:
#   cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
if (NOT DEFINED ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT is not set. Please pass -DONNXRUNTIME_ROOT=/path/to/onnxruntime when running CMake.")
endif ()

# Header include path
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT}/include)

# Library path depends on OS
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "Configuring for Windows")
    set(ONNXRUNTIME_LIBS ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "Configuring for Linux")
    set(ONNXRUNTIME_LIBS ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
else ()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}. Please extend CMakeLists.txt for your OS.")
endif ()

# Include directories for headers
include_directories(
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Collect libraries to link against
set(LIBS
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBS}
)

# Collect all source files
file(GLOB_RECURSE SRC
    src/*.cc
    src/*.cpp
    src/*.h
    src/*.hpp
)

# --- Executable targets ---
add_executable(DetectDemo DetectDemo.cc ${SRC})
target_link_libraries(DetectDemo ${LIBS})

add_executable(MatchDemo MatchDemo.cc ${SRC})
target_link_libraries(MatchDemo ${LIBS})

add_executable(FlowDemo FlowDemo.cc ${SRC})
target_link_libraries(FlowDemo ${LIBS})

add_executable(WebcamMatchDemo WebcamMatchDemo.cc ${SRC})
target_link_libraries(WebcamMatchDemo ${LIBS})