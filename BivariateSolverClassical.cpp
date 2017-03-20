#include <algorithm>
#include "BivariateSolverClassical.hpp"
#include <gsl/gsl_blas.h>
#include <iostream>

BivariateSolverClassical::BivariateSolverClassical(double sigma_x,
						       double sigma_y,
						       double rho,
						       double a,
						       double b,
						       double c,
						       double d,
						       double x_0,
						       double y_0)
  : sigma_x_(sigma_x),
    sigma_y_(sigma_y),
    rho_(rho),
    a_(a),
    b_(b),
    c_(c),
    d_(d),
    x_0_(x_0),
    y_0_(y_0)
{
  double cc = std::sin(M_PI/4.0);
}
