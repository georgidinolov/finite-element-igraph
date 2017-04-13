#include <algorithm>
#include "BasisTypes.hpp"
#include <cmath>
#include <fstream>
#include <gsl/gsl_blas.h>
#include <iostream>
#include <iomanip>
#include <string>

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
  int N = 1.0/dx_ + 1;
  double integral = 0;
  double row_sum = 0;

  for (int i=0; i<N; ++i) {
    gsl_vector_const_view row_i_1 =
      gsl_matrix_const_row(elem_1.get_function_grid(),
			   i);
    gsl_vector_const_view row_i_2 =
      gsl_matrix_const_row(elem_2.get_function_grid(),
			   i);
    gsl_blas_ddot(&row_i_1.vector, &row_i_2.vector, &row_sum);
    integral = integral + row_sum;
  }
  if (std::signbit(integral)) {
    integral = -1.0*std::exp(std::log(std::abs(integral)) + 2*std::log(dx_));
  } else {
    integral = std::exp(std::log(std::abs(integral)) + 2*std::log(dx_));
  }
  return integral;
}

double BivariateGaussianKernelBasis::
project_deriv(const BivariateElement& elem_1,
	      int coord_indeex_1, 
	      const BivariateElement& elem_2,
	      int coord_indeex_2) const
{
  int N = 1.0/dx_ + 1;

  const gsl_matrix* elem_1_deriv_mat = NULL;
  const gsl_matrix* elem_2_deriv_mat = NULL;
  const gsl_matrix* elem_1_mat = elem_1.get_function_grid();
  const gsl_matrix* elem_2_mat = elem_2.get_function_grid();  

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
  double row_sum = 0;

  gsl_vector* left_1 = gsl_vector_alloc(N-1);
  gsl_vector* left_2 = gsl_vector_alloc(N-1);
  gsl_vector* right = gsl_vector_alloc(N-1);

  // // Both derivatives wrt x, so that we take differences of *ROWS*,
  // // since each row corresponds to constant x
  // if (coord_indeex_1 == 0 && coord_indeex_2 == 0) {
  //   for (int i=0; i<N-1; ++i) {
  //     gsl_vector_const_view row_i_1 =
  //     gsl_matrix_const_row(elem_1.get_function_grid(),
  // 			   i);
  //     gsl_vector_const_view row_i_1_plus_dx =
  //     gsl_matrix_const_row(elem_1.get_function_grid(),
  // 			   i+1);
  //     gsl_vector_memcpy(left_1, &row_i_1_plus_dx.vector);
  //     gsl_vector_sub(left_1, &row_i_1.vector);

  //     gsl_vector_const_view row_i_2 =
  //     gsl_matrix_const_row(elem_2.get_function_grid(),
  //   			      i);
  //     gsl_vector_const_view row_i_2_plus_dx =
  //     gsl_matrix_const_row(elem_2.get_function_grid(),
  //   			      i+1);
  //     gsl_vector_memcpy(left_2, &row_i_2_plus_dx.vector);
  //     gsl_vector_sub(left_2, &row_i_2.vector);

  //     gsl_blas_ddot(left_1, left_2, &row_sum);
  //     integral = integral + row_sum / (dx_*dx_);
  //   }
  //   // Both derivatives wrt x and y, so that we take differences of
  //   // *ROWS* and *COLUMNS*
  // } else if (coord_indeex_1 == 0 && coord_indeex_2 == 1) {
  //   for (int i=0; i<N-1; ++i) {

  //     gsl_vector_const_view row_i_1 =
  //     gsl_matrix_const_row(elem_1.get_function_grid(),
  // 			   i);
  //     gsl_vector_const_view row_i_1_plus_dx =
  //     gsl_matrix_const_row(elem_1.get_function_grid(),
  // 			   i+1);
  //     gsl_vector_memcpy(left_1, &row_i_1_plus_dx.vector);
  //     gsl_vector_sub(left_1, &row_i_1.vector);

  //     gsl_vector_const_view col_i_2 =
  //     gsl_matrix_const_column(elem_2.get_function_grid(),
  //   			      i);
  //     gsl_vector_const_view col_i_2_plus_dy =
  //     gsl_matrix_const_column(elem_2.get_function_grid(),
  //   			      i+1);
  //     gsl_vector_memcpy(left_2, &col_i_2_plus_dy.vector);
  //     gsl_vector_sub(left_2, &col_i_2.vector);

  //     gsl_blas_ddot(left_1, left_2, &row_sum);
  //     integral = integral + row_sum / (dx_*dx_);
  //   }
  //   // Both derivatives wrt x and y, so that we take differences of
  //   // *ROWS* and *COLUMNS*
  // } else if (coord_indeex_1 == 1 && coord_indeex_2 == 0) {

  //   for (int i=0; i<N-1; ++i) {
  //     gsl_vector_const_view col_i_1 =
  //     gsl_matrix_const_column(elem_1.get_function_grid(),
  //   			      i);
  //     gsl_vector_const_view col_i_1_plus_dy =
  //     gsl_matrix_const_column(elem_1.get_function_grid(),
  //   			      i+1);
  //     gsl_vector_memcpy(left_1, &col_i_1_plus_dy.vector);
  //     gsl_vector_sub(left_1, &col_i_1.vector);

  //     gsl_vector_const_view row_i_2 =
  //     gsl_matrix_const_row(elem_2.get_function_grid(),
  //   			      i);
  //     gsl_vector_const_view row_i_2_plus_dx =
  //     gsl_matrix_const_row(elem_2.get_function_grid(),
  //   			      i+1);
  //     gsl_vector_memcpy(left_2, &row_i_2_plus_dx.vector);
  //     gsl_vector_sub(left_2, &row_i_2.vector);

  //     gsl_blas_ddot(left_1, left_2, &row_sum);
  //     integral = integral + row_sum / (dx_*dx_);
  //   }
    
  //   // Both derivatives wrt x, so that we take differences of *COLUMNS*
  // } else if (coord_indeex_1 == 1 && coord_indeex_2 == 1) {
  //   for (int i=0; i<N-1; ++i) {
  //     gsl_vector_const_view col_i_1 =
  //     gsl_matrix_const_column(elem_1.get_function_grid(),
  //   			      i);
  //     gsl_vector_const_view col_i_1_plus_dy =
  //     gsl_matrix_const_column(elem_1.get_function_grid(),
  //   			      i+1);
  //     gsl_vector_memcpy(left_1, &col_i_1_plus_dy.vector);
  //     gsl_vector_sub(left_1, &col_i_1.vector);

  //     gsl_vector_const_view col_i_2 =
  //     gsl_matrix_const_column(elem_2.get_function_grid(),
  //   			      i);
  //     gsl_vector_const_view col_i_2_plus_dy =
  //     gsl_matrix_const_column(elem_2.get_function_grid(),
  //   			      i+1);
  //     gsl_vector_memcpy(left_2, &col_i_2_plus_dy.vector);
  //     gsl_vector_sub(left_2, &col_i_2.vector);

  //     gsl_blas_ddot(left_1, left_2, &row_sum);
  //     integral = integral + row_sum / (dx_*dx_);
  //   } 
  // } else {
  //   std::cout << "WRONG COORD INPUT!" << std::endl;    
  // }

  if (coord_indeex_1 == 0 && coord_indeex_2 == 0) {
    for (int i=0; i<N-1; ++i) {
      for (int j=0; j<N-1; ++j) {
    	integral = integral +
    	  (gsl_matrix_get(elem_1_mat,
    			  i + 1,
    			  j) -
    	 gsl_matrix_get(elem_1_mat, i, j))*
    	  (gsl_matrix_get(elem_2_mat,
    			  i + 1, 
    			  j) -
    	   gsl_matrix_get(elem_2_mat, i, j));
      }
    } 
  } else if (coord_indeex_1 == 0 && coord_indeex_2 == 1) {
    for (int i=0; i<N-1; ++i) {
      for (int j=0; j<N-1; ++j) {
    	integral = integral +
    	  (gsl_matrix_get(elem_1_mat,
    			  i + 1,
    			  j) -
    	 gsl_matrix_get(elem_1_mat, i, j))*
    	  (gsl_matrix_get(elem_2_mat,
    			  i, 
    			  j + 1) -
    	   gsl_matrix_get(elem_2_mat, i, j));
      }
    } 
  } else if (coord_indeex_1 == 1 && coord_indeex_2 == 0) {
    for (int i=0; i<N-1; ++i) {
      for (int j=0; j<N-1; ++j) {
  	integral = integral +
  	  (gsl_matrix_get(elem_1_mat,
  			  i,
  			  j + 1) -
  	 gsl_matrix_get(elem_1_mat, i, j))*
  	  (gsl_matrix_get(elem_2_mat,
  			  i + 1, 
  			  j) -
  	   gsl_matrix_get(elem_2_mat, i, j));
      }
    } 
  } else if (coord_indeex_1 == 1 && coord_indeex_2 == 1) {
    for (int i=0; i<N-1; ++i) {
      for (int j=0; j<N-1; ++j) {
  	integral = integral +
  	  (gsl_matrix_get(elem_1_mat,
  			  i,
  			  j + 1) -
  	 gsl_matrix_get(elem_1_mat, i, j))*
  	  (gsl_matrix_get(elem_2_mat,
  			  i, 
  			  j + 1) -
  	   gsl_matrix_get(elem_2_mat, i, j));
      }
    } 
  } else {
    std::cout << "WRONG COORD INPUT!" << std::endl;    
  }

  gsl_vector_free(left_1);
  gsl_vector_free(left_2);
  gsl_vector_free(right);
  
  integral = integral;
  return integral;
}

void BivariateGaussianKernelBasis::save_matrix(const gsl_matrix* mat,
					       std::string file_name) const
{
  std::ofstream output_file;
  output_file.open(file_name);
  output_file << std::fixed << std::setprecision(32);
  for (int i=0; i<mat->size1; ++i) {
    for (int j=0; j<mat->size2; ++j) {
      if (j==mat->size2-1) 
	{
	  output_file << gsl_matrix_get(mat, i,j) 
		      << "\n";
	} else {
	output_file << gsl_matrix_get(mat, i,j) << ",";
      }
    }
  }
  output_file.close();
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
      gsl_matrix_set(xy_nodes, 0, i*y_nodes.size()+j, x_nodes[i]);
      gsl_matrix_set(xy_nodes, 1, i*y_nodes.size()+j, y_nodes[j]);
    }
  }

  double theta = M_PI/4.0;
  gsl_matrix *Rotation_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_set(Rotation_matrix, 0, 0, std::sin(theta));
  gsl_matrix_set(Rotation_matrix, 1, 0, -std::cos(theta));
  gsl_matrix_set(Rotation_matrix, 0, 1, std::cos(theta));
  gsl_matrix_set(Rotation_matrix, 1, 1, std::sin(theta));

  gsl_matrix_add_constant(xy_nodes, -0.5);

  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		 1.0, Rotation_matrix, xy_nodes, 0.0,
		 xieta_nodes);
  gsl_matrix_add_constant(xieta_nodes, 0.5);


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
    std::string file_name = "basis_function_" + std::to_string(i) + ".csv";
    basis_functions_[i].save_function_grid(file_name);
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

  // HAVE A MATRIX VIEW HERE ON THE STACK!
  gsl_matrix* workspace_left = gsl_matrix_alloc(1/dx_ + 1, 1/dx_ + 1);
  gsl_matrix* workspace_right = gsl_matrix_alloc(1/dx_ + 1, 1/dx_ + 1);

  for (unsigned i=0; i<basis_functions_.size(); ++i) {
    std::vector<double> coefficients(i+1, 0.0);
    std::vector<const BivariateElement*> elements(i+1);
    
    for (unsigned j=0; j<i+1; ++j) {
      elements[j] = &basis_functions_[j];
      coefficients[j] = 0.0;
    }
    coefficients[i] = 1.0;
    double projection = 0.0;
    
    BivariateLinearCombinationElement current_orthonormal_element =
      BivariateLinearCombinationElement(elements,
					coefficients);
    // This is where the work happens:
    for (unsigned j=0; j<i; ++j) {
      projection = project(current_orthonormal_element,
			   orthonormal_functions_[j]);
      // SETTING FUNCTION_GRID_
      gsl_matrix_memcpy(workspace_right,
      			orthonormal_functions_[j].get_function_grid());
      gsl_matrix_scale(workspace_right, projection);
      gsl_matrix_memcpy(workspace_left, 
      			current_orthonormal_element.get_function_grid());
      // left workspace is current_orthonormal_element function
      // grid. Setting it as so.
      gsl_matrix_sub(workspace_left, workspace_right);
      current_orthonormal_element.set_function_grid(workspace_left);

      double current_norm = current_orthonormal_element.norm();
      gsl_matrix_scale(workspace_left, 1.0/current_norm);
      current_orthonormal_element.set_function_grid(workspace_left);

      // SETTING DERIV_FUNCTION_GRID_DX_
      gsl_matrix_memcpy(workspace_right, 
      			orthonormal_functions_[j].get_deriv_function_grid_dx());
      gsl_matrix_scale(workspace_right, projection / current_norm);
      gsl_matrix_memcpy(workspace_left, 
      			current_orthonormal_element.get_deriv_function_grid_dx());
      gsl_matrix_scale(workspace_left, 1.0/current_norm);
      // left workspace is current_orthonormal_element function
      // grid. Setting it as so.
      gsl_matrix_sub(workspace_left, workspace_right);
      current_orthonormal_element.set_deriv_function_grid_dx(workspace_left);

      // SETTING DERIV_FUNCTION_GRID_DY__
      gsl_matrix_memcpy(workspace_right, 
      			orthonormal_functions_[j].get_deriv_function_grid_dy());
      gsl_matrix_scale(workspace_right, -1.0*projection / current_norm);
      gsl_matrix_memcpy(workspace_left, 
      			current_orthonormal_element.get_deriv_function_grid_dy());
      gsl_matrix_scale(workspace_left, 1.0 / current_norm);
      // left workspace is current_orthonormal_element function
      // grid. Setting it as so.
      gsl_matrix_add(workspace_left, workspace_right);
      current_orthonormal_element.set_deriv_function_grid_dy(workspace_left);
      
      // current_orthonormal_element = current_orthonormal_element -
      // project(current_orthonormal_element,orthonormal_functions_[j])*
      //         orthonormal_functions_[j]
      for (unsigned k=0; k<j+1; ++k) {
	coefficients[k] = coefficients[k] - projection*
	  orthonormal_functions_[j].get_coefficient(k);
      }

      for (unsigned k=0; k<i+1; ++k) {
	coefficients[k] = coefficients[k] / current_norm;
      }

      current_orthonormal_element.
      	set_coefficients_without_function_grids(coefficients);
    }

    double current_norm = current_orthonormal_element.norm();
    if (i==0) {
      std::cout << std::fixed << std::setprecision(32);
      std::cout << "current_norm = " << current_norm << "\n";
    }

    std::string file_name = "orthonormal_function_" + std::to_string(i)
      + ".csv";

    gsl_matrix_memcpy(workspace_left, current_orthonormal_element.
		      get_function_grid());
    for (int i=0; i<1/dx_+1; ++i) {
      for (int j=0; j<1/dx_+1; ++j) {
    	double entry = gsl_matrix_get(workspace_left,i,j);

    	if (std::signbit(entry)) {
    	  entry = -1.0*exp( log(std::abs(entry)) - log(current_norm) );
    	} else {
    	  entry = exp( log(entry) - log(current_norm) );
    	}
	gsl_matrix_set(workspace_left, i,j, entry);
      }
    }
    // gsl_matrix_memcpy(workspace_left, current_orthonormal_element.
    // 		      get_function_grid());
    // gsl_matrix_scale(workspace_left, 1.0/current_norm);
    current_orthonormal_element.set_function_grid(workspace_left);

    gsl_matrix_memcpy(workspace_left, current_orthonormal_element.
		      get_deriv_function_grid_dx());
    gsl_matrix_scale(workspace_left, 1.0/current_norm);
    current_orthonormal_element.set_deriv_function_grid_dx(workspace_left);

    gsl_matrix_memcpy(workspace_left, current_orthonormal_element.
		      get_deriv_function_grid_dy());
    gsl_matrix_scale(workspace_left, 1.0/current_norm);
    current_orthonormal_element.set_deriv_function_grid_dy(workspace_left);

    for (unsigned j=0; j < i+1; ++j) {
      coefficients[j] = coefficients[j]/current_norm;
    }
    current_orthonormal_element.set_coefficients_without_function_grids(coefficients);

    current_orthonormal_element.save_function_grid(file_name);
    orthonormal_functions_.push_back(current_orthonormal_element);
    
    // gsl_matrix_memcpy(workspace_left, current_orthonormal_element.get_function_grid());
    // gsl_matrix_scale(workspace_left, 1.0/current_norm);
    // current_orthonormal_element.set_function_grid(workspace_left);

    // gsl_matrix_memcpy(workspace_left, current_orthonormal_element.
    // 		      get_deriv_function_grid_dx());
    // gsl_matrix_scale(workspace_left, 1.0/current_norm);
    // current_orthonormal_element.set_deriv_function_grid_dx(workspace_left);

    // gsl_matrix_memcpy(workspace_left, current_orthonormal_element.
    // 		      get_deriv_function_grid_dy());
    // gsl_matrix_scale(workspace_left, 1.0/current_norm);
    // current_orthonormal_element.set_deriv_function_grid_dy(workspace_left);



    double sum = 0;
    if (i==30) {
      std::cout << "i = " << i << std::endl;
      std::cout << std::endl;
      for (int j=0; j<1/dx_+1; ++j) {
    	for (int k=0; k<1/dx_+1; ++k) {
    	  // sum = sum + std::pow(gsl_matrix_get(orthonormal_functions_[i].
    	  // 				      get_deriv_function_grid_dx(),
    	  // 				      j,k), 2);
    	  if (j==180 && k==180) {
    	    std::cout << "ENTRY = "
		      << gsl_matrix_get(basis_functions_[i].
    	  				get_function_grid(),
    	  				j,k)
    	  	      << " ";
    	  }
    	  sum = sum + gsl_matrix_get(orthonormal_functions_[i].
    				     get_function_grid(),j,k)*
    	    gsl_matrix_get(orthonormal_functions_[i].
    			   get_function_grid(),j,k);
		       
    	}
    	//std::cout << std::endl;
      }
      std::cout << "\n sum = " << sum;
      std::cout << "\n sum*dx_*dx_ = " << sum*dx_*dx_;
      std::cout << "\n dx dx element [" << i << "," << i << "] = " 
      		<< project_deriv(orthonormal_functions_[i], 0,
      				 orthonormal_functions_[i], 0);
      std::cout << "\n dy dy element [" << i << "," << i << "] = " 
      		<< project_deriv(orthonormal_functions_[i], 1,
      				 orthonormal_functions_[i], 1);
      std::cout << "\n dx dy element [" << i << "," << i << "] = " 
      		<< project_deriv(orthonormal_functions_[i], 0,
      				 orthonormal_functions_[i], 1);
      std::cout << "\n dy dx element [" << i << "," << i << "] = " 
      		<< project_deriv(orthonormal_functions_[i], 1,
      				 orthonormal_functions_[i], 0);
      std::cout << "\n element [" << i << "," << i << "] = " 
      		<< project(orthonormal_functions_[i],
      			   orthonormal_functions_[i])
      		<< std::endl;
    }
  }

  gsl_matrix_free(workspace_left);
  gsl_matrix_free(workspace_right);
}

void BivariateGaussianKernelBasis::set_mass_matrix()
{
  std::cout << "IN set_mass_matrix()" << std::endl;

  gsl_matrix_free(mass_matrix_);
  mass_matrix_ = gsl_matrix_alloc(orthonormal_functions_.size(),
				  orthonormal_functions_.size());
  double entry = 0;
   
  for (unsigned i=0; i<orthonormal_functions_.size(); ++i) {
    if (i==50) {
      std::cout << "row " << i << " of mass_matrix_ = \n";
    }
    for (unsigned j=i; j<orthonormal_functions_.size(); ++j) {

      entry = project(orthonormal_functions_[i], 
      		      orthonormal_functions_[j]);
      
      // gsl_vector* input = gsl_vector_alloc(2);
      // entry = 0;
      // for (int k=0; k<1/dx_+1; ++k) {
      // 	gsl_vector_set(input, 0, k*dx_);

      // 	for (int l=0; l<1/dx_+1; ++l) {
      // 	  gsl_vector_set(input, 1, l*dx_);
      // 	  entry = entry + (orthonormal_functions_[i])(input)*
      // 	    (orthonormal_functions_[j])(input);
      // 	}
      // }
      // entry = entry * std::pow(dx_, 2);

      if (i==50) { 
	std::cout << entry << "\n";
      }
      gsl_matrix_set(mass_matrix_,
  			i, j,
			entry);
      gsl_matrix_set(mass_matrix_,
  			j, i,
			entry);
    }
  }
  save_matrix(mass_matrix_,
	      "mass_matrix.csv");
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
  std::cout << "setting system matrices" << std::endl;
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
    for (unsigned j=0; j<orthonormal_functions_.size(); ++j) {

      // system_matrix_dx_dx_
      double entry = project_deriv(orthonormal_functions_[i], 0,
				   orthonormal_functions_[j], 0);
      gsl_matrix_set(system_matrix_dx_dx_,
		     i, j,
		     entry);
      // gsl_matrix_set(system_matrix_dx_dx_,
      // 		     j, i,
      // 		     entry);

      
      // system_matrix_dx_dy_
      entry = project_deriv(orthonormal_functions_[i], 0,
			    orthonormal_functions_[j], 1);
      gsl_matrix_set(system_matrix_dx_dy_,
		     i, j,
		     entry);
      // gsl_matrix_set(system_matrix_dx_dy_,
      // 		     j, i,
      // 		     entry);
      
      // system_matrix_dy_dx_
      entry = project_deriv(orthonormal_functions_[i], 1,
			    orthonormal_functions_[j], 0);
      gsl_matrix_set(system_matrix_dy_dx_,
		     i, j,
		     entry);
      // gsl_matrix_set(system_matrix_dy_dx_,
      // 		     j, i,
      // 		     entry);
      
      // system_matrix_dy_dy_
      entry = project_deriv(orthonormal_functions_[i], 1,
			    orthonormal_functions_[j], 1);
      gsl_matrix_set(system_matrix_dy_dy_,
		     i, j,
		     entry);
      // gsl_matrix_set(system_matrix_dy_dy_,
      // 		     j, i,
      // 		     entry);
    }
  }

  save_matrix(system_matrix_dx_dx_,
	      "system_matrix_dx_dx.csv");
  save_matrix(system_matrix_dx_dy_,
	      "system_matrix_dx_dy.csv");
  save_matrix(system_matrix_dy_dx_,
	      "system_matrix_dy_dx.csv");
  save_matrix(system_matrix_dy_dy_,
	      "system_matixr_dy_dy.csv");
}
