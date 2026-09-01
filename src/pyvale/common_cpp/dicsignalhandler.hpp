// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once
#include <atomic>
#include <pybind11/pybind11.h>

extern std::atomic<bool> stop_request;
void signalHandler(int signal);
void raise_on_interrupt();
