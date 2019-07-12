cmake_minimum_required(VERSION 3.0)
project(FilterTest)

find_package(OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS} )

set(SOURCE main.cpp filter.cpp chi2inv.cpp)

add_executable(filter-test ${SOURCE})
target_compile_options(filter-test PUBLIC --std=c++14 -g)
target_link_libraries(filter-test ${OpenCV_LIBS} )
