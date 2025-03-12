// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICJACOBIAN_H
#define DICJACOBIAN_H

// STD library Header files

// GNU Scientific Library Header files
#include <gsl/gsl_blas.h>

// Program Header files





namespace jacobian {

    int ssd(const gsl_vector *p_gsl, void *data, gsl_matrix *J);
    int nssd(const gsl_vector *p_gsl, void *data, gsl_matrix *J);
    int znssd(const gsl_vector *p_gsl, void *data, gsl_matrix *J);

}

#endif //DICJACOBIAN_H