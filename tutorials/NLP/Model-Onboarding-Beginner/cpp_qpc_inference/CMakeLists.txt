# ==============================================================================
#
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 Qualcomm Technologies, Inc.
#
# All rights reserved.
# Confidential & Proprietary - Qualcomm Technologies, Inc.
#
# @@-COPYRIGHT-END-@@
#
# ==============================================================================

project(simple-bert-inference-example)
cmake_minimum_required (VERSION 3.17.2)
set(CMAKE_CXX_STANDARD 17)

add_executable(simple-bert-inference-example main.cpp)

target_include_directories(simple-bert-inference-example
    PRIVATE
        "/opt/qti-aic/dev/inc"
)


set_target_properties(simple-bert-inference-example
    PROPERTIES
    LINK_FLAGS "-Wl,--no-as-needed"
)

target_compile_options(simple-bert-inference-example
    PRIVATE
        -fstack-protector-all
        -Werror
        -Wall
        -Wextra
        -Wno-sign-compare
        -Wno-unused-parameter
        -Wno-missing-field-initializers
)

target_link_libraries(simple-bert-inference-example
    PRIVATE
        pthread
        dl
)