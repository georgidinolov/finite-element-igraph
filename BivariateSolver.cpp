#include <algorithm>
#include "BivariateSolver.hpp"
#include <gsl/gsl_blas.h>
#include <iostream>

BivariateSolver::BivariateSolver(const BivariateBasis& basis,
				 double sigma_x,
				 double sigma_y,
				 double rho,
				 double x_0,
				 double y_0)
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
					       y_0))
{
  if (x_0_ < 0.0 || x_0_ > 1.0 || y_0_ < 0.0 || y_0_ > 1.0) {
    std::cout << "ERROR: IC out of range" << std::endl;
  }
  std::cout << "small tt = " << small_t_solution_.get_t() << std::endl;

  
}

BivariateSolver::~BivariateSolver()
{
  // freeing vectors
  

  // freeing matrices
  
}

double BivariateSolver::
operator()(const igraph_vector_t& input) const
{
  
  double out = 0;
  return out;
}

double BivariateSolver::
operator()(const gsl_vector* input) const
{
  return 0.0;
}

