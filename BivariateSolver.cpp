#include <algorithm>
#include "BivariateSolver.hpp"
#include <gsl/gsl_blas.h>
#include <iostream>

BivariateSolver::BivariateSolver(const BivariateBasis& basis,
				 double sigma_x,
				 double sigma_y,
				 double rho,
				 double x_0,
				 double y_0,
				 double t,
				 double dx)
  : sigma_x_(sigma_x),
    sigma_y_(sigma_y),
    rho_(rho),
    x_0_(x_0),
    y_0_(y_0),
    mvtnorm_(MultivariateNormal()),
    basis_(basis),
    small_t_solution_(BivariateSolverClassical(sigma_x,
					       sigma_y,
					       rho,
					       x_0,
					       y_0)),
    t_(t),
    dx_(dx)
{
  if (x_0_ < 0.0 || x_0_ > 1.0 || y_0_ < 0.0 || y_0_ > 1.0) {
    std::cout << "ERROR: IC out of range" << std::endl;
  }
  std::cout << "small tt = " << small_t_solution_.get_t() << std::endl;
  small_t_solution_.set_function_grid(dx_);
}

BivariateSolver::~BivariateSolver()
{
  // freeing vectors
  // freeing matrices
}

double BivariateSolver::
operator()(const gsl_vector* input) const
{
  if ((t_ - small_t_solution_.get_t()) <= 1e-32) {
    return small_t_solution_(input, t_);
  } else {
    unsigned number_basis_elements = 
      basis_.get_orthonormal_elements().size();
    std::vector<double> IC_coefs = std::vector<double> (number_basis_elements);

    for (unsigned i=0; i<number_basis_elements; ++i) {
      IC_coefs[i] = basis_.project(small_t_solution_,
				   basis_.get_orthonormal_element(i));
      std::cout << "IC_coefs[" << i << "] = " <<IC_coefs[i] << std::endl;
    }

    double out = basis_.project(small_t_solution_,
				basis_.get_orthonormal_element(0));
    return out;
  }
}

