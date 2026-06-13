// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// Temporary because nanobind really didn't want to cooperate when I tried to add common_cpp as nanobind module to use in the ray tracer
// (Yes, I updated CMakeLists and bindings, it just really likes to complain about module not found for some reason)

#pragma once
#include <atomic>

extern std::atomic<bool> stop_request;
void signalHandler(int signal);