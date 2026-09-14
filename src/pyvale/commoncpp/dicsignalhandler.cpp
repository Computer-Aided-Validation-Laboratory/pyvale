// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#include "./dicsignalhandler.hpp"
#include <csignal>
#include <Python.h>

std::atomic<bool> stop_request = false;

void signalHandler(int signal) {
    if (signal == SIGINT) {
        stop_request = true;
    }
}

void raise_on_interrupt() {
    if (stop_request) {
        stop_request = false;  // Reset for next operation
        PyErr_SetString(PyExc_KeyboardInterrupt, "");
        throw pybind11::error_already_set();
    }
}
