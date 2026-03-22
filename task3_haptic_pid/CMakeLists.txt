cmake_minimum_required(VERSION 3.16)

project(haptic_pid_demo
  VERSION 1.0
  DESCRIPTION "Task 3 — Haptic Dial PID controller (SocialHW Robotics Assessment)"
  LANGUAGES CXX
)

# ── Language standard ────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)   # -std=c++17, not -std=gnu++17

# ── Default to Release if no build type specified ────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(haptic_pid_demo
  src/haptic_pid.cpp   # PID class implementation
  src/main.cpp         # Demo harness + tuning report
)

# ── Compile options ──────────────────────────────────────────────────────────
target_compile_options(haptic_pid_demo PRIVATE
  -Wall -Wextra -Wpedantic
  $<$<CONFIG:Release>:-O2>
  $<$<CONFIG:Debug>:-O0 -g>
)

# ── Link libraries ───────────────────────────────────────────────────────────
# libm is required for M_PI on some Linux toolchains; harmless elsewhere.
target_link_libraries(haptic_pid_demo PRIVATE m)

# ── Header include path ──────────────────────────────────────────────────────
target_include_directories(haptic_pid_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── Install rule (optional) ──────────────────────────────────────────────────
install(TARGETS haptic_pid_demo RUNTIME DESTINATION bin)
