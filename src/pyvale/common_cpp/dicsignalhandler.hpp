// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once
#include <atomic>

extern std::atomic<bool> stop_request;
void signalHandler(int signal);
