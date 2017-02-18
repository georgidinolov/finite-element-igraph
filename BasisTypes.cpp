#include "BasisTypes.hpp"
#include <cmath>
#include "MultivariateNormal.hpp"

// =================== BASIS ELEMENT CLASS ===================
BasisElement::~BasisElement()
{}

GaussianKernelElement::
GaussianKernelElement(long unsigned dimension,
		      double exponent_power,
		      const igraph_vector_t& mean_vector,
		      const igraph_matrix_t& covariance_matrix)
  : dimension_(dimension),
    exponent_power_(exponent_power)
{
  igraph_vector_init(&mean_vector_, dimension_);
  igraph_vector_update(&mean_vector_, &mean_vector);
  
  igraph_matrix_init(&covariance_matrix_, dimension_, dimension_);
  igraph_matrix_update(&covariance_matrix_, &covariance_matrix);
}

GaussianKernelElement::~GaussianKernelElement()
{
  igraph_vector_destroy(&mean_vector_);
  igraph_matrix_destroy(&covariance_matrix_);
}

double GaussianKernelElement::
operator()(const igraph_vector_t& input) const
{
  gls_matrix *covariance_matrix_gsl = gls_matrix_alloc(dimension_,
  						       dimension_);
  gls_vector *mean_vector_gsl = gsl_vector_alloc(dimension_);



  double out = dmvnorm(dimension_,
		       mean_vector_gsl,
		       mean_vector_gsl,
		       covariance_matrix_gsl);

    gsl_matrix_free(covariance_matrix_gsl);
  gsl_vector_free(mean_vector_gsl);
  
  return 0;
}

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
