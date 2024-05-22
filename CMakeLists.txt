cmake_minimum_required(VERSION 3.28)
project(c_test C)

set(CMAKE_C_STANDARD 11)

add_executable(c_test main.c
        Sources/nn.c
        Headers/nn.h
)
