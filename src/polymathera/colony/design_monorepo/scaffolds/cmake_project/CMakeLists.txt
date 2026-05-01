cmake_minimum_required(VERSION 3.18)
project(${name_snake} VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(${name_snake}
    src/${name_snake}.cpp
)
target_include_directories(${name_snake} PUBLIC include)

include(CTest)
if (BUILD_TESTING)
    add_executable(${name_snake}_smoke tests/test_smoke.cpp)
    target_link_libraries(${name_snake}_smoke PRIVATE ${name_snake})
    add_test(NAME ${name_snake}_smoke COMMAND ${name_snake}_smoke)
endif()
