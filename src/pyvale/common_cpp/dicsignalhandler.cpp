// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#include "./dicsignalhandler.hpp"
#include <csignal>

std::atomic<bool> stop_request = false;

void signalHandler(int signal) {
    if (signal == SIGINT) {
        stop_request = true;
    }
}
