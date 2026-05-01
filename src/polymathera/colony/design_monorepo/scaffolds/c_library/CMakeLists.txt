cmake_minimum_required(VERSION 3.18)
project(${name_snake} VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(${name_snake}
    src/${name_snake}.c
)
target_include_directories(${name_snake}
    PUBLIC include
)

include(CTest)
if (BUILD_TESTING)
    add_executable(${name_snake}_smoke tests/test_smoke.c)
    target_link_libraries(${name_snake}_smoke PRIVATE ${name_snake})
    add_test(NAME ${name_snake}_smoke COMMAND ${name_snake}_smoke)
endif()
