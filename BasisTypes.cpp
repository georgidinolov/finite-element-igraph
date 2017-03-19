#include <algorithm>
#include "BasisTypes.hpp"
#include <gsl/gsl_blas.h>
#include <iostream>

// =================== BASE BASIS CLASS ======================
BaseBasis::~BaseBasis()
{}

// ============== GAUSSIAN KERNEL BASIS CLASS ==============
BivariateGaussianKernelBasis::BivariateGaussianKernelBasis(double dx,
							   double rho,
							   double sigma,
							   double power,
							   double std_dev_factor)
  : dx_(dx)
{
  // first create the list of basis elements
  set_basis_functions(rho,sigma,power,std_dev_factor);

  // second create the orthonormal list of elements
  set_orthonormal_functions();

  //
  set_mass_matrix();
  
  igraph_matrix_init(&system_matrix_, 2, 2);
  igraph_matrix_fill(&system_matrix_, 1);
  
  igraph_matrix_init(&mass_matrix_, 2, 2);
  igraph_matrix_fill(&mass_matrix_, 1);
}

BivariateGaussianKernelBasis::~BivariateGaussianKernelBasis()
{
  igraph_matrix_destroy(&system_matrix_);
  igraph_matrix_destroy(&mass_matrix_);
  igraph_matrix_destroy(&inner_product_matrix_);
}


const LinearCombinationElement& BivariateGaussianKernelBasis::
get_orthonormal_element(unsigned i) const
{
  if (i < orthonormal_functions_.size()) {
    return orthonormal_functions_[i];
  }
  else {
    std::cout << "ERROR: orthonormal_function out of range" << std::endl;
    return orthonormal_functions_[i];
  }
}

const igraph_matrix_t& BivariateGaussianKernelBasis::get_system_matrix() const
{
  return system_matrix_;
}

const igraph_matrix_t& BivariateGaussianKernelBasis::get_mass_matrix() const
{
  return mass_matrix_;
}

double BivariateGaussianKernelBasis::
project(const BasisElement& elem_1,
	const BasisElement& elem_2) const
{
  long int N = 1.0/dx_;
  int dimension = 2;

  double integral = 0;
  igraph_vector_t input;
  igraph_vector_init(&input, dimension);
  double x;
  double y;

  for (long int i=0; i<N; ++i) {
    for (long int j=0; j<N; ++j) {
      x = i*dx_;
      y = j*dx_;
      igraph_vector_set(&input, 0, x);
      igraph_vector_set(&input, 1, y);

      integral = integral + elem_1(input)*elem_2(input);
    }
  }
  integral = integral * std::pow(dx_, 2);
  
  igraph_vector_destroy(&input);
  return integral;
}

void BivariateGaussianKernelBasis::set_basis_functions(double rho,
						       double sigma,
						       double power,
						       double std_dev_factor)
{
  std::cout << "IN set_basis_functions" << std::endl;
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

    basis_functions_.push_back(GaussianKernelElement(dx_,
						     2,
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

// Performing Gram-Schmidt Orthogonalization
void BivariateGaussianKernelBasis::set_orthonormal_functions()
{
  std::cout << "IN set_orthonormal_functions" << std::endl;
  std::cout << "Number basis elements = " << basis_functions_.size()
	    << std::endl;

  // initializing inner product matrix
  igraph_matrix_init(&inner_product_matrix_,
		     basis_functions_.size(),
		     basis_functions_.size());
  
  for (unsigned i=0; i<basis_functions_.size(); ++i) {
    for (unsigned j=i; j<basis_functions_.size(); ++j) {
      igraph_matrix_set(&inner_product_matrix_, i, j,
			project(basis_functions_[i],
				basis_functions_[j]));
      igraph_matrix_set(&inner_product_matrix_, j, i,
			  igraph_matrix_e(&inner_product_matrix_,i,j));
      
      std::cout << "projection(" << i << "," << j << ") = "
		<< igraph_matrix_e(&inner_product_matrix_,i,j)
		<< std::endl;
    }
  }
  
  for (unsigned i=0; i<basis_functions_.size(); ++i) {
    if (i==0) {
      std::cout << "(" << i << ")" << std::endl;

      std::vector<double> coefficients =
	std::vector<double> {1.0/std::sqrt(igraph_matrix_e(&inner_product_matrix_,
							   i, i))};
      std::vector<const BasisElement*> elements =
	std::vector<const BasisElement*> {&basis_functions_[i]};

      orthonormal_functions_.push_back(LinearCombinationElement(elements,
								coefficients));
    } else {
      
      std::vector<double> coefficients(i+1, 0.0);
      std::vector<const BasisElement*> elements(0);
      coefficients[i] = 1.0;

      for (unsigned j=0; j<i; ++j) {
	elements.push_back(&basis_functions_[j]);
	double projection = 0;

	for (unsigned k=0; k<j+1; ++k) {
	  projection = projection +
	    igraph_matrix_e(&inner_product_matrix_, i, k) *
	    orthonormal_functions_[j].get_coefficient(k);
	}

	for (unsigned k=0; k<j+1; ++k) {
	  coefficients[k] = coefficients[k] -
	    projection * orthonormal_functions_[j].get_coefficient(k);
	}
      }
      elements.push_back(&basis_functions_[i]);

      double current_norm = 0;
      for (unsigned j=0; j < i+1; ++j) {
	for (unsigned j_prime=0; j_prime < i+1; ++j_prime) {
	  current_norm = current_norm +
	    coefficients[j]*coefficients[j_prime]*
	    igraph_matrix_e(&inner_product_matrix_, j, j_prime);
	}
      }
      current_norm = std::sqrt(current_norm);
      
      for (unsigned k=0; k<i+1; ++k) {
	coefficients[k] = coefficients[k]/current_norm;
      }
      orthonormal_functions_.push_back(LinearCombinationElement(elements,
  								coefficients));
    }
  }

  // std::cout << "size of orthonormal_functions_ = "
  // 	    << orthonormal_functions_.size() << std::endl;
  
  // for (unsigned i=0; i<orthonormal_functions_.size(); ++i) {
  //   for (unsigned j=i; j<orthonormal_functions_.size(); ++j) {
  //         std::cout << project(orthonormal_functions_[i],
  // 			       orthonormal_functions_[j]) << std::endl;
  //   }
  // }
}
