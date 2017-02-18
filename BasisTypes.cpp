#include <algorithm>
#include "BasisTypes.hpp"
#include <gsl/gsl_blas.h>
#include <iostream>

// =================== BASIS ELEMENT CLASS ===================
BasisElement::~BasisElement()
{}

GaussianKernelElement::
GaussianKernelElement(long unsigned dimension,
		      double exponent_power,
		      const igraph_vector_t& mean_vector,
		      const igraph_matrix_t& covariance_matrix)
  : dimension_(dimension),
    exponent_power_(exponent_power),
    mvtnorm_(MultivariateNormal())
{
  igraph_vector_init(&mean_vector_, dimension_);
  igraph_vector_update(&mean_vector_, &mean_vector);
  
  igraph_matrix_init(&covariance_matrix_, dimension_, dimension_);
  igraph_matrix_update(&covariance_matrix_, &covariance_matrix);
}

GaussianKernelElement::
GaussianKernelElement(const GaussianKernelElement& element)
  : dimension_(element.dimension_),
    exponent_power_(element.exponent_power_),
    mvtnorm_(MultivariateNormal())
{
  igraph_vector_init(&mean_vector_, dimension_);
  igraph_vector_update(&mean_vector_, &element.mean_vector_);
  
  igraph_matrix_init(&covariance_matrix_, dimension_, dimension_);
  igraph_matrix_update(&covariance_matrix_, &element.covariance_matrix_);
}

GaussianKernelElement::~GaussianKernelElement()
{
  igraph_vector_destroy(&mean_vector_);
  igraph_matrix_destroy(&covariance_matrix_);
}

double GaussianKernelElement::
operator()(const igraph_vector_t& input) const
{
  if (igraph_vector_size(&input) == dimension_) {
    double mollifier = 1;
    
    gsl_matrix *covariance_matrix_gsl = gsl_matrix_alloc(dimension_,
							 dimension_);
    gsl_vector *mean_vector_gsl = gsl_vector_alloc(dimension_);
    gsl_vector *input_gsl = gsl_vector_alloc(dimension_);
    
    for (unsigned i=0; i<dimension_; ++i) {
      mollifier = mollifier *
	std::pow(igraph_vector_e(&input, i), exponent_power_) *
	std::pow((1-igraph_vector_e(&input, i)), exponent_power_);
      
      gsl_vector_set(mean_vector_gsl, i, igraph_vector_e(&mean_vector_, i));
      gsl_vector_set(input_gsl, i, igraph_vector_e(&input, i));
      
      gsl_matrix_set(covariance_matrix_gsl, i, i,
		     igraph_matrix_e(&covariance_matrix_,
				     i, i));
      for (unsigned j=i+1; j<dimension_; ++j) {
	gsl_matrix_set(covariance_matrix_gsl, i, j,
		       igraph_matrix_e(&covariance_matrix_,
				       i, j));
	gsl_matrix_set(covariance_matrix_gsl, j, i,
		       igraph_matrix_e(&covariance_matrix_,
				       j, i));
      }
    }
    
    
    double out = mvtnorm_.dmvnorm(dimension_,
				  input_gsl,
				  mean_vector_gsl,
				  covariance_matrix_gsl) *
	mollifier;
    
    gsl_matrix_free(covariance_matrix_gsl);
    gsl_vector_free(mean_vector_gsl);
    gsl_vector_free(input_gsl);

    return out;
  } else {
    std::cout << "INPUT SIZE WRONG" << std::endl;
    return 0;
  }
}

const igraph_vector_t& GaussianKernelElement::get_mean_vector() const
{
  return mean_vector_;
}

const igraph_matrix_t& GaussianKernelElement::get_covariance_matrix() const
{
  return covariance_matrix_;
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
  // first create the list of basis elements
  set_basis_functions(rho,sigma,power,std_dev_factor);

  // second create the orthonormal list of elements
  
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

void GaussianKernelBasis::set_basis_functions(double rho,
					      double sigma,
					      double power,
					      double std_dev_factor)
{
  // creating the x-nodes
  double by = std_dev_factor * sigma * std::sqrt(1-rho);
  double current = 0.5 - std::sqrt(2.0);

  std::vector<double> x_nodes;
  while ((current - 0.5) <= 1e-32) {
    x_nodes.push_back(current);
    current = current + by;
  }

  current = 0.5;
  while ( (current-(0.5+std::sqrt(2))) <= 1e-32 ) {
    x_nodes.push_back(current);
    current = current + by;
  }

  // x_nodes is already sorted
  auto last = std::unique(x_nodes.begin(), x_nodes.end());
  x_nodes.erase(last, x_nodes.end());
  

  // creating the y-nodes
  by = std_dev_factor * sigma * std::sqrt(1+rho);
  current = 0.5 - std::sqrt(2.0);

  std::vector<double> y_nodes;
  while ((current - 0.5) <= 1e-32) {
    y_nodes.push_back(current);
    current = current + by;
  }

  current = 0.5;
  while ( (current-(0.5+std::sqrt(2))) <= 1e-32 ) {
    y_nodes.push_back(current);
    current = current + by;
  }

  // y_nodes is already sorted
  last = std::unique(y_nodes.begin(), y_nodes.end());
  y_nodes.erase(last, y_nodes.end());
  
  gsl_matrix *xy_nodes = gsl_matrix_alloc(2, x_nodes.size()*y_nodes.size());
  gsl_matrix *xieta_nodes = gsl_matrix_alloc(2, x_nodes.size()*y_nodes.size());

  for (unsigned i=0; i<x_nodes.size(); ++i) {
    for (unsigned j=0; j<y_nodes.size(); ++j) {
      gsl_matrix_set(xy_nodes, 0, i*y_nodes.size()+j, x_nodes[i] - 0.5);
      gsl_matrix_set(xy_nodes, 1, i*y_nodes.size()+j, y_nodes[j] - 0.5);
    }
  }

  double theta = M_PI/4.0;

  gsl_matrix *Rotation_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_set(Rotation_matrix, 0, 0, std::sin(theta));
  gsl_matrix_set(Rotation_matrix, 1, 0, -std::cos(theta));
  gsl_matrix_set(Rotation_matrix, 0, 1, std::cos(theta));
  gsl_matrix_set(Rotation_matrix, 1, 1, std::sin(theta));

  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		 1.0, Rotation_matrix, xy_nodes, 0.0,
		 xieta_nodes);

  std::vector<long unsigned> indeces_within_boundary;
  for (unsigned j=0; j<x_nodes.size()*y_nodes.size(); ++j) {
    if ( (gsl_matrix_get(xieta_nodes, 0, j) >= 1e-32) &&
	 (gsl_matrix_get(xieta_nodes, 0, j) <= 1.0-1e-32) &&
	 (gsl_matrix_get(xieta_nodes, 1, j) >= 1e-32) &&
	 (gsl_matrix_get(xieta_nodes, 1, j) <= 1.0-1e-32) )
      {
	indeces_within_boundary.push_back(j);
      }
  }

  igraph_vector_t mean_vector;
  igraph_matrix_t covariance_matrix;
  
  igraph_vector_init(&mean_vector, 2);
  igraph_matrix_init(&covariance_matrix, 2, 2);
  igraph_matrix_set(&covariance_matrix, 0, 0, std::pow(sigma, 2));
  igraph_matrix_set(&covariance_matrix, 1, 0, rho*std::pow(sigma, 2));
  igraph_matrix_set(&covariance_matrix, 0, 1, rho*std::pow(sigma, 2));
  igraph_matrix_set(&covariance_matrix, 1, 1, std::pow(sigma, 2)); 
  
  for (unsigned const& index: indeces_within_boundary) {
    igraph_vector_set(&mean_vector, 0,
  		      gsl_matrix_get(xieta_nodes, 0, index));
    igraph_vector_set(&mean_vector, 1,
  		      gsl_matrix_get(xieta_nodes, 1, index));

    basis_functions_.push_back(GaussianKernelElement(2,
						     power,
						     mean_vector,
						     covariance_matrix));
  }

  gsl_matrix_free(xy_nodes);
  gsl_matrix_free(xieta_nodes);  
  gsl_matrix_free(Rotation_matrix);

  igraph_vector_destroy(&mean_vector);
  igraph_matrix_destroy(&covariance_matrix);
}
