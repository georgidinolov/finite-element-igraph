#include "BasisTypes.hpp"

// =================== BASE BASIS CLASS ======================
BaseBasis::~BaseBasis()
{}

// ============== GAUSSIAN KERNEL BASIS CLASS ==============
GaussianKernelBasis::GaussianKernelBasis(double rho,
					 double sigma,
					 double power,
					 double std_dev_factor)
{
  igraph_matrix_init(&system_matrix_, 2, 2);
  igraph_matrix_fill(&system_matrix_, 1);
  
  igraph_matrix_init(&mass_matrix_, 2, 2);
  igraph_matrix_fill(&mass_matrix_, 1);
}

GaussianKernelBasis::~GaussianKernelBasis()
{
  igraph_matrix_destroy(&system_matrix_);
  igraph_matrix_destroy(&mass_matrix_);
}

const igraph_matrix_t& GaussianKernelBasis::get_system_matrix() const
{
  return system_matrix_;
}

const igraph_matrix_t& GaussianKernelBasis::get_mass_matrix() const
{
  return mass_matrix_;
}
