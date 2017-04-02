#include <algorithm>
#include "BasisTypes.hpp"
#include <gsl/gsl_blas.h>
#include <iostream>

// =================== BASE BASIS CLASS ======================
BivariateBasis::~BivariateBasis()
{}

// ============== GAUSSIAN KERNEL BASIS CLASS ==============
BivariateGaussianKernelBasis::BivariateGaussianKernelBasis(double dx,
							   double rho,
							   double sigma,
							   double power,
							   double std_dev_factor)
  : dx_(dx),
    system_matrix_dx_dx_(gsl_matrix_alloc(1,1)),
    system_matrix_dx_dy_(gsl_matrix_alloc(1,1)),
    system_matrix_dy_dx_(gsl_matrix_alloc(1,1)),
    system_matrix_dy_dy_(gsl_matrix_alloc(1,1)),
    mass_matrix_(gsl_matrix_alloc(1,1)),
    inner_product_matrix_(gsl_matrix_alloc(1,1)),
    deriv_inner_product_matrix_dx_dx_(gsl_matrix_alloc(1,1)),
    deriv_inner_product_matrix_dx_dy_(gsl_matrix_alloc(1,1)),
    deriv_inner_product_matrix_dy_dx_(gsl_matrix_alloc(1,1)),
    deriv_inner_product_matrix_dy_dy_(gsl_matrix_alloc(1,1))
{
  // first create the list of basis elements
  set_basis_functions(rho,sigma,power,std_dev_factor);

  // second create the orthonormal list of elements
  set_orthonormal_functions_stable();

  //
  set_mass_matrix();

  //
  set_system_matrices_stable();
    
  // igraph_matrix_init(&system_matrix_, 2, 2);
  // igraph_matrix_fill(&system_matrix_, 1);
  
  // igraph_matrix_init(&mass_matrix_, 2, 2);
  // igraph_matrix_fill(&mass_matrix_, 1);
}

BivariateGaussianKernelBasis::~BivariateGaussianKernelBasis()
{
  gsl_matrix_free(mass_matrix_);
  gsl_matrix_free(inner_product_matrix_);

  gsl_matrix_free(deriv_inner_product_matrix_dx_dx_);
  gsl_matrix_free(deriv_inner_product_matrix_dx_dy_);
  gsl_matrix_free(deriv_inner_product_matrix_dy_dx_);
  gsl_matrix_free(deriv_inner_product_matrix_dy_dy_);

  gsl_matrix_free(system_matrix_dx_dx_);
  gsl_matrix_free(system_matrix_dx_dy_);
  gsl_matrix_free(system_matrix_dy_dx_);
  gsl_matrix_free(system_matrix_dy_dy_);
}

const BivariateGaussianKernelElement& BivariateGaussianKernelBasis::
get_basis_element(unsigned i) const
{
  return basis_functions_[i];
}

const BivariateLinearCombinationElement& BivariateGaussianKernelBasis::
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

const std::vector<BivariateLinearCombinationElement>& BivariateGaussianKernelBasis::
get_orthonormal_elements() const
{
  return orthonormal_functions_;
}

const gsl_matrix* BivariateGaussianKernelBasis::
get_system_matrix_dx_dx() const
{
  return system_matrix_dx_dx_;
}
const gsl_matrix* BivariateGaussianKernelBasis::
get_system_matrix_dx_dy() const
{
  return system_matrix_dx_dy_;
}
const gsl_matrix* BivariateGaussianKernelBasis::
get_system_matrix_dy_dx() const
{
  return system_matrix_dy_dx_;
}
const gsl_matrix* BivariateGaussianKernelBasis::
get_system_matrix_dy_dy() const
{
  return system_matrix_dy_dy_;
}

const gsl_matrix* BivariateGaussianKernelBasis::get_mass_matrix() const
{
  return mass_matrix_;
}

double BivariateGaussianKernelBasis::
project(const BivariateElement& elem_1,
	const BivariateElement& elem_2) const
{
  long int N = 1.0/dx_;
  double integral = 0;
  for (long int i=0; i<N; ++i) {
    for (long int j=0; j<N; ++j) {
      // integral = integral + elem_1(input)*elem_2(input);
      integral = integral + 
	gsl_matrix_get(elem_1.get_function_grid(), i,j)*
	gsl_matrix_get(elem_2.get_function_grid(), i,j);
    }
  }
  integral = integral * std::pow(dx_, 2);
  return integral;
}

double BivariateGaussianKernelBasis::
project(const BivariateGaussianKernelElement& elem_1,
	const BivariateGaussianKernelElement& elem_2) const
{
  int N = 1.0/dx_;
  int dimension = 2;

  double integral = 0;
  gsl_vector* input = gsl_vector_alloc(2);
  double x;
  double y;

  gsl_matrix * function_grid_matrix = gsl_matrix_alloc(N,N);
  gsl_matrix_memcpy(function_grid_matrix, elem_1.get_function_grid());
  gsl_matrix_mul_elements(function_grid_matrix, elem_2.get_function_grid());
  
  for (int i=0; i<N; ++i) {
    for (int j=0; j<N; ++j) {
      // x = i*dx_;
      // y = j*dx_;
      // gsl_vector_set(input, 0, x);
      // gsl_vector_set(input, 1, y);
      // integral = integral + elem_1(input)*elem_2(input);
      integral = integral + gsl_matrix_get(function_grid_matrix,i,j);
    }
  }
  integral = integral * std::pow(dx_, 2);
  
  gsl_vector_free(input);
  gsl_matrix_free(function_grid_matrix);
  return integral;
}

double BivariateGaussianKernelBasis::
project_deriv_analytic(const BivariateElement& elem_1,
		       long int coord_indeex_1, 
		       const BivariateElement& elem_2,
		       long int coord_indeex_2) const
{
  int N = 1.0/dx_;
  int dimension = 2;

  double integral = 0;
  gsl_vector* input = gsl_vector_alloc(2);
  double x;
  double y;

  for (int i=0; i<N; ++i) {
    x = i*dx_;
    gsl_vector_set(input, 0, x);
    
    for (int j=0; j<N; ++j) {
      y = j*dx_;
      gsl_vector_set(input, 1, y);

      integral = integral +
	elem_1.first_derivative(input, coord_indeex_1)*
	elem_2.first_derivative(input, coord_indeex_2);
    }
  }
  integral = integral * std::pow(dx_, 2);
  
  gsl_vector_free(input);
  return integral;
}

double BivariateGaussianKernelBasis::
project_deriv(const BivariateElement& elem_1,
	      int coord_indeex_1, 
	      const BivariateElement& elem_2,
	      int coord_indeex_2) const
{
  int N = 1.0/dx_;
  int dimension = 2;

  const gsl_matrix* elem_1_deriv_mat = NULL;
  const gsl_matrix* elem_2_deriv_mat = NULL;

  if (coord_indeex_1 == 0) {
    elem_1_deriv_mat = elem_1.get_deriv_function_grid_dx();
  } else if (coord_indeex_1 == 1) {
    elem_1_deriv_mat = elem_1.get_deriv_function_grid_dy();
  } else {
    std::cout << "WRONG COORD INPUT!" << std::endl;
  }

  if (coord_indeex_2 == 0) {
    elem_2_deriv_mat = elem_2.get_deriv_function_grid_dx();
  } else if (coord_indeex_2 == 1) {
    elem_2_deriv_mat = elem_2.get_deriv_function_grid_dy();
  } else {
    std::cout << "WRONG COORD INPUT!" << std::endl;
  }

  double integral = 0;
  for (int i=0; i<N; ++i) {
    for (int j=0; j<N; ++j) {
      integral = integral +
	gsl_matrix_get(elem_1_deriv_mat, i, j)*
	gsl_matrix_get(elem_2_deriv_mat, i, j);
    }
  }
  integral = integral * std::pow(dx_, 2);
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

  std::vector<double> x_nodes (0);
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

  std::cout << "Allocating matrices" << std::endl;
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

  std::cout << "Allocating matrices 2" << std::endl;  
  gsl_vector* mean_vector = gsl_vector_alloc(2);
  gsl_matrix* covariance_matrix = gsl_matrix_alloc(2,2);
  
  gsl_matrix_set(covariance_matrix, 0, 0, std::pow(sigma, 2));
  gsl_matrix_set(covariance_matrix, 1, 0, rho*std::pow(sigma, 2));
  gsl_matrix_set(covariance_matrix, 0, 1, rho*std::pow(sigma, 2));
  gsl_matrix_set(covariance_matrix, 1, 1, std::pow(sigma, 2)); 

  basis_functions_ =
    std::vector<BivariateGaussianKernelElement> (indeces_within_boundary.size());
  
  for (unsigned i=0; i<indeces_within_boundary.size(); ++i) {
    unsigned const& index = indeces_within_boundary[i];
    gsl_vector_set(mean_vector, 0,
		   gsl_matrix_get(xieta_nodes, 0, index));
    gsl_vector_set(mean_vector, 1,
		   gsl_matrix_get(xieta_nodes, 1, index));
    basis_functions_[i] = BivariateGaussianKernelElement(dx_,
    							 power,
    							 mean_vector,
    							 covariance_matrix);
  }
  
  gsl_matrix_free(xy_nodes);
  gsl_matrix_free(xieta_nodes);  
  gsl_matrix_free(Rotation_matrix);

  gsl_vector_free(mean_vector);
  gsl_matrix_free(covariance_matrix);
}

// Performing Gram-Schmidt Orthogonalization
void BivariateGaussianKernelBasis::set_orthonormal_functions()
{
  std::cout << "IN set_orthonormal_functions" << std::endl;
  std::cout << "Number basis elements = " << basis_functions_.size()
	    << std::endl;

  // initializing inner product matrix
  gsl_matrix_free(inner_product_matrix_);
  inner_product_matrix_ = gsl_matrix_alloc(basis_functions_.size(),
					   basis_functions_.size());

  for (unsigned i=0; i<basis_functions_.size(); ++i) {
    for (unsigned j=i; j<basis_functions_.size(); ++j) {
      gsl_matrix_set(inner_product_matrix_, i, j,
  		     project(basis_functions_[i],
  			     basis_functions_[j]));
      gsl_matrix_set(inner_product_matrix_, j, i,
  		     gsl_matrix_get(inner_product_matrix_,i,j));
      
      std::cout << "projection(" << i << "," << j << ") = "
  		<< gsl_matrix_get(inner_product_matrix_,i,j)
  		<< std::endl;
    }
  }
  
  for (unsigned i=0; i<basis_functions_.size(); ++i) {
    if (i==0) {
      std::cout << "(" << i << ")" << std::endl;

      std::vector<double> coefficients =
	std::vector<double> {1.0/std::sqrt(gsl_matrix_get(inner_product_matrix_,
							  i, i))};
      std::vector<const BivariateElement*> elements =
	std::vector<const BivariateElement*> {&basis_functions_[i]};

      orthonormal_functions_.push_back(BivariateLinearCombinationElement(elements,
								coefficients));
    } else {
      std::cout << "(" << i << ")" << std::endl;
      
      std::vector<double> coefficients(i+1, 0.0);
      std::vector<const BivariateElement*> elements(0);
      coefficients[i] = 1.0;

      for (unsigned j=0; j<i; ++j) {
	elements.push_back(&basis_functions_[j]);
	double projection = 0;

	for (unsigned k=0; k<j+1; ++k) {
	  projection = projection +
	    gsl_matrix_get(inner_product_matrix_, i, k) *
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
	    gsl_matrix_get(inner_product_matrix_, j, j_prime);
	}
      }
      current_norm = std::sqrt(current_norm);
      
      for (unsigned k=0; k<i+1; ++k) {
	coefficients[k] = coefficients[k]/current_norm;
      }
      orthonormal_functions_.push_back(BivariateLinearCombinationElement(elements,
									 coefficients));
    }
  }
}

// Performing Gram-Schmidt Orthogonalization
void BivariateGaussianKernelBasis::set_orthonormal_functions_stable()
{
  std::cout << "IN set_orthonormal_functions_stable" << std::endl;
  std::cout << "Number basis elements = " << basis_functions_.size()
	    << std::endl;

  for (unsigned i=0; i<basis_functions_.size(); ++i) {
    std::cout << "(" << i << ")" << std::endl;
    std::vector<double> coefficients(i+1, 0.0);
    std::vector<const BivariateElement*> elements(i+1);
    
    for (unsigned j=0; j<i+1; ++j) {
      elements[j] = &basis_functions_[j];
    }
    coefficients[i] = 1.0;
    double projection = 0.0;
    
    BivariateLinearCombinationElement current_orthonormal_element =
      BivariateLinearCombinationElement(elements,
					coefficients);
    
    for (unsigned j=0; j<i; ++j) {
      projection = project(current_orthonormal_element,
			   orthonormal_functions_[j]);
      
      for (unsigned k=0; k<j+1; ++k) {
	coefficients[k] = coefficients[k] - projection*
	  orthonormal_functions_[j].get_coefficient(k);
      }
      current_orthonormal_element.set_coefficients(coefficients);
    }
    
    double current_norm = current_orthonormal_element.norm();
    for (unsigned j=0; j < i+1; ++j) {
      coefficients[j] = coefficients[j]/current_norm;
    }
    
    current_orthonormal_element.set_coefficients(coefficients);
    orthonormal_functions_.push_back(current_orthonormal_element);
  }
}

void BivariateGaussianKernelBasis::set_mass_matrix()
{
  std::cout << "IN set_mass_matrix()" << std::endl;

  gsl_matrix_free(mass_matrix_);
  mass_matrix_ = gsl_matrix_alloc(orthonormal_functions_.size(),
				  orthonormal_functions_.size());
  double entry = 0;
   
  for (unsigned i=0; i<orthonormal_functions_.size(); ++i) {
    for (unsigned j=i; j<orthonormal_functions_.size(); ++j) {

      // double entry = 0;
      // for (unsigned k=0; k<orthonormal_functions_[i].get_elements().size(); ++k) {
      // 	for (unsigned l=0; l<orthonormal_functions_[j].get_elements().size(); ++l) {
      // 	  entry = entry +
      // 	    orthonormal_functions_[i].get_coefficient(k)*
      // 	    orthonormal_functions_[j].get_coefficient(l)*
      // 	    gsl_matrix_get(inner_product_matrix_, k, l);
      // 	}
      // }
      entry = project(orthonormal_functions_[i], 
		      orthonormal_functions_[j]);
      
      gsl_matrix_set(mass_matrix_,
  			i, j,
			entry);
      gsl_matrix_set(mass_matrix_,
  			j, i,
			entry);
      std::cout << "mass_matrix_[" << i
  		<< "," << j << "] = "
  		<< gsl_matrix_get(mass_matrix_, i, j) << std::endl;
    }
  }
}

void BivariateGaussianKernelBasis::set_system_matrices()
{
  gsl_matrix_free(deriv_inner_product_matrix_dx_dx_);
  deriv_inner_product_matrix_dx_dx_ =
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());

  gsl_matrix_free(deriv_inner_product_matrix_dx_dy_);
  deriv_inner_product_matrix_dx_dy_ = 
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());
  
  gsl_matrix_free(deriv_inner_product_matrix_dy_dx_);
  deriv_inner_product_matrix_dy_dx_ = 
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());
  
  gsl_matrix_free(deriv_inner_product_matrix_dy_dy_);
  deriv_inner_product_matrix_dy_dy_ =
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());
  
  for (unsigned i=0; i<basis_functions_.size(); ++i) {
    for (unsigned j=i; j<basis_functions_.size(); ++j) {
      gsl_matrix_set(deriv_inner_product_matrix_dx_dx_,
		     i, j,
		     project_deriv(basis_functions_[i], 0,
				   basis_functions_[j], 0));
      gsl_matrix_set(deriv_inner_product_matrix_dx_dx_,
		     j, i,
		     gsl_matrix_get(deriv_inner_product_matrix_dx_dx_,
				    i,j));
      
      gsl_matrix_set(deriv_inner_product_matrix_dx_dy_,
		     i, j,
		     project_deriv(basis_functions_[i], 0,
				       basis_functions_[j], 1));
      gsl_matrix_set(deriv_inner_product_matrix_dx_dy_,
		     j, i,
		     gsl_matrix_get(deriv_inner_product_matrix_dx_dy_,
				    i,j));
      
      
      gsl_matrix_set(deriv_inner_product_matrix_dy_dx_,
		     i, j,
		     project_deriv(basis_functions_[i], 0,
				   basis_functions_[j], 1));
      gsl_matrix_set(deriv_inner_product_matrix_dy_dx_,
		     j, i,
		     gsl_matrix_get(deriv_inner_product_matrix_dy_dx_,
				    i,j));
      
      gsl_matrix_set(deriv_inner_product_matrix_dy_dy_,
		     i, j,
		     project_deriv(basis_functions_[i], 0,
				   basis_functions_[j], 1));
      gsl_matrix_set(deriv_inner_product_matrix_dy_dy_,
		     j, i,
		     gsl_matrix_get(deriv_inner_product_matrix_dy_dy_,
				    i,j));
      
      std::cout << "deriv_inner_product_matrix_dx_dx_[" << i
		<< "," << j << "] = "
		<< gsl_matrix_get(deriv_inner_product_matrix_dx_dx_,
				  i,j)
		<< " ";
    }
    std::cout << std::endl;
  }
  
  gsl_matrix_free(system_matrix_dx_dx_);
  system_matrix_dx_dx_ = 
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());
  
  gsl_matrix_free(system_matrix_dx_dy_);
  system_matrix_dx_dy_ =
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());
  
  gsl_matrix_free(system_matrix_dy_dx_);
  system_matrix_dy_dx_ =
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());
  
  gsl_matrix_free(system_matrix_dy_dy_);
  system_matrix_dy_dy_ = 
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());

   for (unsigned i=0; i<basis_functions_.size(); ++i) {
     for (unsigned j=i; j<basis_functions_.size(); ++j) {

       // system_matrix_dx_dx_
       double entry = 0;
       for (unsigned k=0; k<orthonormal_functions_[i].get_elements().size(); ++k) {
	 for (unsigned l=0; l<orthonormal_functions_[j].get_elements().size(); ++l) {
	  entry = entry +
	    orthonormal_functions_[i].get_coefficient(k)*
	    orthonormal_functions_[j].get_coefficient(l)*
	    gsl_matrix_get(deriv_inner_product_matrix_dx_dx_, k, l);
	 }
       }
       gsl_matrix_set(system_matrix_dx_dx_,
  			i, j,
			entry);
       gsl_matrix_set(system_matrix_dx_dx_,
			 j, i,
			 entry);

       // system_matrix_dx_dy_
       entry = 0;
       for (unsigned k=0; k<orthonormal_functions_[i].get_elements().size(); ++k) {
	 for (unsigned l=0; l<orthonormal_functions_[j].get_elements().size(); ++l) {
	  entry = entry +
	    orthonormal_functions_[i].get_coefficient(k)*
	    orthonormal_functions_[j].get_coefficient(l)*
	    gsl_matrix_get(deriv_inner_product_matrix_dx_dy_, k, l);
	 }
       }
       gsl_matrix_set(system_matrix_dx_dy_,
  			i, j,
			entry);
       gsl_matrix_set(system_matrix_dx_dy_,
			 j, i,
			 entry);

       // system_matrix_dy_dx_
       entry = 0;
       for (unsigned k=0; k<orthonormal_functions_[i].get_elements().size(); ++k) {
	 for (unsigned l=0; l<orthonormal_functions_[j].get_elements().size(); ++l) {
	  entry = entry +
	    orthonormal_functions_[i].get_coefficient(k)*
	    orthonormal_functions_[j].get_coefficient(l)*
	    gsl_matrix_get(deriv_inner_product_matrix_dy_dx_, k, l);
	 }
       }
       gsl_matrix_set(system_matrix_dy_dx_,
  			i, j,
			entry);
       gsl_matrix_set(system_matrix_dy_dx_,
			 j, i,
			 entry);

       // system_matrix_dy_dy_
       entry = 0;
       for (unsigned k=0; k<orthonormal_functions_[i].get_elements().size(); ++k) {
	 for (unsigned l=0; l<orthonormal_functions_[j].get_elements().size(); ++l) {
	  entry = entry +
	    orthonormal_functions_[i].get_coefficient(k)*
	    orthonormal_functions_[j].get_coefficient(l)*
	    gsl_matrix_get(deriv_inner_product_matrix_dy_dy_, k, l);
	 }
       }
       gsl_matrix_set(system_matrix_dy_dy_,
  			i, j,
			entry);
       gsl_matrix_set(system_matrix_dy_dy_,
			 j, i,
			 entry);
       
       std::cout << "system_matrix_dx_dx_[" << i
		 << "," << j << "] = "
		 << gsl_matrix_get(system_matrix_dx_dx_,
				    i,j)
		 << " ";
     }
     std::cout << std::endl;
   }
}


void BivariateGaussianKernelBasis::set_system_matrices_stable()
{
  gsl_matrix_free(deriv_inner_product_matrix_dx_dx_);
  deriv_inner_product_matrix_dx_dx_ =
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());

  gsl_matrix_free(deriv_inner_product_matrix_dx_dy_);
  deriv_inner_product_matrix_dx_dy_ = 
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());
  
  gsl_matrix_free(deriv_inner_product_matrix_dy_dx_);
  deriv_inner_product_matrix_dy_dx_ = 
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());
  
  gsl_matrix_free(deriv_inner_product_matrix_dy_dy_);
  deriv_inner_product_matrix_dy_dy_ =
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());
  
  gsl_matrix_free(system_matrix_dx_dx_);
  system_matrix_dx_dx_ = 
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());
  
  gsl_matrix_free(system_matrix_dx_dy_);
  system_matrix_dx_dy_ =
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());
  
  gsl_matrix_free(system_matrix_dy_dx_);
  system_matrix_dy_dx_ =
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());
  
  gsl_matrix_free(system_matrix_dy_dy_);
  system_matrix_dy_dy_ = 
    gsl_matrix_alloc(basis_functions_.size(),
		     basis_functions_.size());

  for (unsigned i=0; i<orthonormal_functions_.size(); ++i) {
    for (unsigned j=i; j<orthonormal_functions_.size(); ++j) {

      // system_matrix_dx_dx_
      double entry = project_deriv(orthonormal_functions_[i], 0,
				   orthonormal_functions_[j], 0);
      gsl_matrix_set(system_matrix_dx_dx_,
		     i, j,
		     entry);
      gsl_matrix_set(system_matrix_dx_dx_,
		     j, i,
		     entry);
      
      // system_matrix_dx_dy_
      entry = project_deriv(orthonormal_functions_[i], 0,
			    orthonormal_functions_[j], 1);
      gsl_matrix_set(system_matrix_dx_dy_,
		     i, j,
		     entry);
      gsl_matrix_set(system_matrix_dx_dy_,
		     j, i,
		     entry);
      
      // system_matrix_dy_dx_
      entry = project_deriv(orthonormal_functions_[i], 1,
			    orthonormal_functions_[j], 0);
      gsl_matrix_set(system_matrix_dy_dx_,
		     i, j,
		     entry);
      gsl_matrix_set(system_matrix_dy_dx_,
		     j, i,
		     entry);
      
      // system_matrix_dy_dy_
      entry = project_deriv(orthonormal_functions_[i], 1,
			    orthonormal_functions_[j], 1);
      gsl_matrix_set(system_matrix_dy_dy_,
		     i, j,
		     entry);
      gsl_matrix_set(system_matrix_dy_dy_,
		     j, i,
		     entry);
      
      std::cout << "system_matrix_dx_dx_[" << i
		<< "," << j << "] = "
		<< gsl_matrix_get(system_matrix_dx_dx_,
				  i,j)
		<< " ";
    }
    std::cout << std::endl;
  }
}
