cmake_minimum_required(VERSION 3.19)
project(dlib_cpp)
find_package(dlib REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(dlib_cpp main.cpp)
target_link_libraries(dlib_cpp dlib::dlib ${OpenCV_LIBS})
