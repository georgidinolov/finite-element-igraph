#include <algorithm>
#include "BasisTypes.hpp"
#include <chrono>
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
BivariateGaussianKernelBasis::BivariateGaussianKernelBasis()
  : dx_(0.1),
    system_matrix_dx_dx_(gsl_matrix_alloc(1,1)),
    system_matrix_dx_dy_(gsl_matrix_alloc(1,1)),
    system_matrix_dy_dx_(gsl_matrix_alloc(1,1)),
    system_matrix_dy_dy_(gsl_matrix_alloc(1,1)),
    mass_matrix_(gsl_matrix_alloc(1,1)),
    inner_product_matrix_(gsl_matrix_alloc(1,1)),
    deriv_inner_product_matrix_dx_dx_(gsl_matrix_alloc(1,1)),
    deriv_inner_product_matrix_dx_dy_(gsl_matrix_alloc(1,1)),
    deriv_inner_product_matrix_dy_dx_(gsl_matrix_alloc(1,1)),
    deriv_inner_product_matrix_dy_dy_(gsl_matrix_alloc(1,1)),
    integration_rule_multiplier_(gsl_matrix_alloc(1,1))
{
  gsl_matrix_set(integration_rule_multiplier_,0,0,1.0);
  // first create the list of basis elements
  set_basis_functions(0.0,1.0,1.0,0.5);

  //
  set_mass_matrix();

  //
  set_system_matrices_stable();
}


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
    deriv_inner_product_matrix_dy_dy_(gsl_matrix_alloc(1,1)),
    integration_rule_multiplier_(gsl_matrix_alloc(1,1))
{
  set_integration_rule_multiplier();
  // first create the list of basis elements
  set_basis_functions(rho,sigma,power,std_dev_factor);
  std::cout << "done with basis functions" << std::endl;
  //
  set_mass_matrix();
  std::cout << "done with mass matrix" << std::endl;

  //
  set_system_matrices_stable();
  std::cout << "done with system matrices" << std::endl;

  // igraph_matrix_init(&system_matrix_, 2, 2);
  // igraph_matrix_fill(&system_matrix_, 1);

  // igraph_matrix_init(&mass_matrix_, 2, 2);
  // igraph_matrix_fill(&mass_matrix_, 1);
}

BivariateGaussianKernelBasis::BivariateGaussianKernelBasis(double dx,
							   double rho,
							   double sigma_x,
							   double sigma_y,
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
    deriv_inner_product_matrix_dy_dy_(gsl_matrix_alloc(1,1)),
    integration_rule_multiplier_(gsl_matrix_alloc(1,1))
{
  set_integration_rule_multiplier();
  // first create the list of basis elements
  set_basis_functions(rho,
		      sigma_x,
		      sigma_y,
		      power,
		      std_dev_factor);

    //
  set_mass_matrix();

  //
  set_system_matrices_stable();

  // igraph_matrix_init(&system_matrix_, 2, 2);
  // igraph_matrix_fill(&system_matrix_, 1);

  // igraph_matrix_init(&mass_matrix_, 2, 2);
  // igraph_matrix_fill(&mass_matrix_, 1);
}

BivariateGaussianKernelBasis::
BivariateGaussianKernelBasis(const BivariateGaussianKernelBasis& basis)
  : dx_(basis.dx_),
    system_matrix_dx_dx_(gsl_matrix_alloc(basis.system_matrix_dx_dx_->size1,
					  basis.system_matrix_dx_dx_->size2)),
    system_matrix_dx_dy_(gsl_matrix_alloc(basis.system_matrix_dx_dy_->size1,
					  basis.system_matrix_dx_dy_->size2)),
    system_matrix_dy_dx_(gsl_matrix_alloc(basis.system_matrix_dy_dx_->size1,
					  basis.system_matrix_dy_dx_->size2)),
    system_matrix_dy_dy_(gsl_matrix_alloc(basis.system_matrix_dy_dy_->size1,
					  basis.system_matrix_dy_dy_->size2)),
    //
    mass_matrix_(gsl_matrix_alloc(basis.mass_matrix_->size1,
				  basis.mass_matrix_->size2)),
    inner_product_matrix_(gsl_matrix_alloc(basis.inner_product_matrix_->size1,
					   basis.inner_product_matrix_->size2)),
    //
    deriv_inner_product_matrix_dx_dx_(gsl_matrix_alloc(basis.deriv_inner_product_matrix_dx_dx_->size1,
						       basis.deriv_inner_product_matrix_dx_dx_->size2)),
    deriv_inner_product_matrix_dx_dy_(gsl_matrix_alloc(basis.deriv_inner_product_matrix_dx_dy_->size1,
						       basis.deriv_inner_product_matrix_dx_dy_->size2)),
    deriv_inner_product_matrix_dy_dx_(gsl_matrix_alloc(basis.deriv_inner_product_matrix_dy_dx_->size1,
						       basis.deriv_inner_product_matrix_dy_dx_->size2)),
    deriv_inner_product_matrix_dy_dy_(gsl_matrix_alloc(basis.deriv_inner_product_matrix_dy_dy_->size1,
						       basis.deriv_inner_product_matrix_dy_dy_->size2)),
    integration_rule_multiplier_(gsl_matrix_alloc(basis.integration_rule_multiplier_->size1,
						  basis.integration_rule_multiplier_->size2))
{
  gsl_matrix_memcpy(system_matrix_dx_dx_, basis.system_matrix_dx_dx_);
  gsl_matrix_memcpy(system_matrix_dx_dy_, basis.system_matrix_dx_dy_);
  gsl_matrix_memcpy(system_matrix_dy_dx_, basis.system_matrix_dy_dx_);
  gsl_matrix_memcpy(system_matrix_dy_dy_, basis.system_matrix_dy_dy_);

  gsl_matrix_memcpy(mass_matrix_, basis.mass_matrix_);
  gsl_matrix_memcpy(inner_product_matrix_, basis.inner_product_matrix_);

  gsl_matrix_memcpy(deriv_inner_product_matrix_dx_dx_, basis.deriv_inner_product_matrix_dx_dx_);
  gsl_matrix_memcpy(deriv_inner_product_matrix_dx_dy_, basis.deriv_inner_product_matrix_dx_dy_);
  gsl_matrix_memcpy(deriv_inner_product_matrix_dy_dx_, basis.deriv_inner_product_matrix_dy_dx_);
  gsl_matrix_memcpy(deriv_inner_product_matrix_dy_dy_, basis.deriv_inner_product_matrix_dy_dy_);

  gsl_matrix_memcpy(integration_rule_multiplier_, basis.integration_rule_multiplier_);

  orthonormal_functions_ =
    std::vector<BivariateLinearCombinationElement> (basis.orthonormal_functions_.size());

  for (unsigned i=0; i<basis.orthonormal_functions_.size(); ++i) {
    orthonormal_functions_[i] = basis.orthonormal_functions_[i];
  }
}

BivariateGaussianKernelBasis& BivariateGaussianKernelBasis::
operator=(const BivariateGaussianKernelBasis& rhs)
{
  dx_ = rhs.dx_;
  gsl_matrix_free(system_matrix_dx_dx_);
  system_matrix_dx_dx_ = gsl_matrix_alloc(rhs.system_matrix_dx_dx_->size1,
					  rhs.system_matrix_dx_dx_->size2);

  gsl_matrix_free(system_matrix_dx_dy_);
  system_matrix_dx_dy_ = gsl_matrix_alloc(rhs.system_matrix_dx_dy_->size1,
					  rhs.system_matrix_dx_dy_->size2);

  gsl_matrix_free(system_matrix_dy_dx_);
  system_matrix_dy_dx_ = gsl_matrix_alloc(rhs.system_matrix_dy_dx_->size1,
					  rhs.system_matrix_dy_dx_->size2);

  gsl_matrix_free(system_matrix_dy_dy_);
  system_matrix_dy_dy_ = gsl_matrix_alloc(rhs.system_matrix_dy_dy_->size1,
					  rhs.system_matrix_dy_dy_->size2);
  //
  gsl_matrix_free(mass_matrix_);
  mass_matrix_ = gsl_matrix_alloc(rhs.mass_matrix_->size1,
				  rhs.mass_matrix_->size2);

  gsl_matrix_free(inner_product_matrix_);
  inner_product_matrix_ = gsl_matrix_alloc(rhs.inner_product_matrix_->size1,
					   rhs.inner_product_matrix_->size2);
  //
  //
  gsl_matrix_free(deriv_inner_product_matrix_dx_dx_);
  deriv_inner_product_matrix_dx_dx_ = gsl_matrix_alloc(rhs.deriv_inner_product_matrix_dx_dx_->size1,
						       rhs.deriv_inner_product_matrix_dx_dx_->size2);

  gsl_matrix_free(deriv_inner_product_matrix_dx_dy_);
  deriv_inner_product_matrix_dx_dy_ = gsl_matrix_alloc(rhs.deriv_inner_product_matrix_dx_dy_->size1,
						       rhs.deriv_inner_product_matrix_dx_dy_->size2);

  gsl_matrix_free(deriv_inner_product_matrix_dy_dx_);
  deriv_inner_product_matrix_dy_dx_ = gsl_matrix_alloc(rhs.deriv_inner_product_matrix_dy_dx_->size1,
						       rhs.deriv_inner_product_matrix_dy_dx_->size2);

  gsl_matrix_free(deriv_inner_product_matrix_dy_dy_);
  deriv_inner_product_matrix_dy_dy_ = gsl_matrix_alloc(rhs.deriv_inner_product_matrix_dy_dy_->size1,
						       rhs.deriv_inner_product_matrix_dy_dy_->size2);
  //
  gsl_matrix_free(integration_rule_multiplier_);
  integration_rule_multiplier_ = gsl_matrix_alloc(rhs.integration_rule_multiplier_->size1,
						  rhs.integration_rule_multiplier_->size2);

  gsl_matrix_memcpy(system_matrix_dx_dx_, rhs.system_matrix_dx_dx_);
  gsl_matrix_memcpy(system_matrix_dx_dy_, rhs.system_matrix_dx_dy_);
  gsl_matrix_memcpy(system_matrix_dy_dx_, rhs.system_matrix_dy_dx_);
  gsl_matrix_memcpy(system_matrix_dy_dy_, rhs.system_matrix_dy_dy_);

  gsl_matrix_memcpy(mass_matrix_, rhs.mass_matrix_);
  gsl_matrix_memcpy(inner_product_matrix_, rhs.inner_product_matrix_);

  gsl_matrix_memcpy(deriv_inner_product_matrix_dx_dx_, rhs.deriv_inner_product_matrix_dx_dx_);
  gsl_matrix_memcpy(deriv_inner_product_matrix_dx_dy_, rhs.deriv_inner_product_matrix_dx_dy_);
  gsl_matrix_memcpy(deriv_inner_product_matrix_dy_dx_, rhs.deriv_inner_product_matrix_dy_dx_);
  gsl_matrix_memcpy(deriv_inner_product_matrix_dy_dy_, rhs.deriv_inner_product_matrix_dy_dy_);

  gsl_matrix_memcpy(integration_rule_multiplier_, rhs.integration_rule_multiplier_);

  orthonormal_functions_ =
    std::vector<BivariateLinearCombinationElement> (rhs.orthonormal_functions_.size());

  for (unsigned i=0; i<rhs.orthonormal_functions_.size(); ++i) {
    orthonormal_functions_[i] = rhs.orthonormal_functions_[i];
  }

  return *this;
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

  gsl_matrix_free(integration_rule_multiplier_);
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
  return project_omp(elem_1, elem_2);
}

double BivariateGaussianKernelBasis::
project_solver(const BivariateSolverClassical& solver_1,
	       const BivariateElement& elem_2) const
{
  int N = 1.0/dx_;
  double integral = 0;

  const gsl_matrix* mat_1 = solver_1.get_function_grid();
  const gsl_matrix* mat_2 = elem_2.get_function_grid();
  double product_mat [(N+1)*(N+1)];
  gsl_matrix_view product_mat_view = gsl_matrix_view_array(product_mat, N+1, N+1);

  gsl_matrix_memcpy(&product_mat_view.matrix, mat_1);
  gsl_matrix_mul_elements(&product_mat_view.matrix, mat_2);
  gsl_matrix_mul_elements(&product_mat_view.matrix, integration_rule_multiplier_);
  
  for (int i=0; i<N+1; ++i) {
    for (int j=0; j<N+1; ++j) {
      integral += product_mat[i*(N+1) + j];
    }
  }

  return integral;
}

double BivariateGaussianKernelBasis::
project_simple(const BivariateElement& elem_1,
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
project_omp(const BivariateElement& elem_1,
	    const BivariateElement& elem_2) const
{
  int N = 1.0/dx_;
  double integral = 0;

  // // http://mathfaculty.fullerton.edu/mathews/n2003/SimpsonsRule2DMod.html
  // // SIMPSON'S RULE START 
  // const gsl_matrix* mat_1 = elem_1.get_function_grid();
  // const gsl_matrix* mat_2 = elem_2.get_function_grid();

  // integral += gsl_matrix_get(mat_1, 0,0)*gsl_matrix_get(mat_2, 0,0) +
  //   gsl_matrix_get(mat_1, 0,N)*gsl_matrix_get(mat_2, 0,N) +
  //   gsl_matrix_get(mat_1, N,0)*gsl_matrix_get(mat_2, N,0) +
  //   gsl_matrix_get(mat_1, N,N)*gsl_matrix_get(mat_2, N,N);

  // for (int i=1; i<=N/2; ++i) {
  //   if (i<N/2) {
  //     integral += 
  // 	4.0*gsl_matrix_get(mat_1, 0,2*i-1)*gsl_matrix_get(mat_2, 0,2*i-1) +
  // 	2.0*gsl_matrix_get(mat_1, 0,2*i)*gsl_matrix_get(mat_2, 0,2*i) +
  // 	// // 
  // 	4.0*gsl_matrix_get(mat_1, N,2*i-1)*gsl_matrix_get(mat_2, N,2*i-1) +
  // 	2.0*gsl_matrix_get(mat_1, N,2*i)*gsl_matrix_get(mat_2, N,2*i) +
  // 	//
  // 	4.0*gsl_matrix_get(mat_1, 2*i-1,0)*gsl_matrix_get(mat_2, 2*i-1,0) +
  // 	2.0*gsl_matrix_get(mat_1, 2*i,0)*gsl_matrix_get(mat_2, 2*i,0) +
  // 	// // 
  // 	4.0*gsl_matrix_get(mat_1, 2*i-1,N)*gsl_matrix_get(mat_2, 2*i-1,N) +
  // 	2.0*gsl_matrix_get(mat_1, 2*i,N)*gsl_matrix_get(mat_2, 2*i,N);

  //     for (int j=1; j<=N/2; ++j) {
  // 	if (j<N/2) {
  // 	  integral +=
  // 	    16.0*gsl_matrix_get(mat_1, 2*i-1, 2*j-1)*gsl_matrix_get(mat_2, 2*i-1,2*j-1) +
  // 	    8.0*gsl_matrix_get(mat_1, 2*i, 2*j-1)*gsl_matrix_get(mat_2, 2*i,2*j-1) +
  // 	    //
  // 	    8.0*gsl_matrix_get(mat_1, 2*i-1, 2*j)*gsl_matrix_get(mat_2, 2*i-1,2*j) +
  // 	    4.0*gsl_matrix_get(mat_1, 2*i, 2*j)*gsl_matrix_get(mat_2, 2*i,2*j);
  // 	} else {
  // 	  integral +=
  // 	    16.0*gsl_matrix_get(mat_1, 2*i-1, 2*j-1)*gsl_matrix_get(mat_2, 2*i-1,2*j-1) +
  // 	    8.0*gsl_matrix_get(mat_1, 2*i-1, 2*j)*gsl_matrix_get(mat_2, 2*i-1,2*j);
  // 	}
  //     }

  //   } else {
  //     integral += 
  // 	4.0*gsl_matrix_get(mat_1, 0,2*i-1)*gsl_matrix_get(mat_2, 0,2*i-1) +
  // 	// // 
  // 	4.0*gsl_matrix_get(mat_1, N,2*i-1)*gsl_matrix_get(mat_2, N,2*i-1) +
  // 	//
  // 	4.0*gsl_matrix_get(mat_1, 2*i-1,0)*gsl_matrix_get(mat_2, 2*i-1,0) +
  // 	// // 
  // 	4.0*gsl_matrix_get(mat_1, 2*i-1,N)*gsl_matrix_get(mat_2, 2*i-1,N);

  //     for (int j=1; j<=N/2; ++j) {
  // 	if (j<N/2) {
  // 	  integral +=
  // 	    16.0*gsl_matrix_get(mat_1, 2*i-1, 2*j-1)*gsl_matrix_get(mat_2, 2*i-1,2*j-1) +
  // 	    8.0*gsl_matrix_get(mat_1, 2*i, 2*j-1)*gsl_matrix_get(mat_2, 2*i,2*j-1);
  // 	} else {
  // 	  integral +=
  // 	    16.0*gsl_matrix_get(mat_1, 2*i-1, 2*j-1)*gsl_matrix_get(mat_2, 2*i-1,2*j-1);
  // 	}
  //     }
  //   }
  // }

  // if (std::signbit(integral)) {
  //   integral = -1.0*std::exp(std::log(std::abs(integral)) + 2*std::log(dx_) - std::log(9.0));
  // } else {
  //   integral = std::exp(std::log(std::abs(integral)) + 2*std::log(dx_) - std::log(9.0));
  // }
  // return integral;
  // // SIMPSON RULE END

  
  // TRAPEZOIDAL RULE START 
  const gsl_matrix* mat_1 = elem_1.get_function_grid();
  const gsl_matrix* mat_2 = elem_2.get_function_grid();

  for (int i=0; i<N; ++i) {
    if (i==0 || i==N-1) {

      for (int j=1; j<N-1; ++j) {
  	integral += gsl_matrix_get(mat_1, i,j)*gsl_matrix_get(mat_2, i,j)*2;
      }
      integral += gsl_matrix_get(mat_1, i,0)*gsl_matrix_get(mat_2, i,0)*1;
      integral += gsl_matrix_get(mat_1, i,N-1)*gsl_matrix_get(mat_2, i,N-1)*1;

    } else {

      for (int j=1; j<N-1; ++j) {
  	integral += gsl_matrix_get(mat_1, i,j)*gsl_matrix_get(mat_2, i,j)*4;
      }
      integral += gsl_matrix_get(mat_1, i,0)*gsl_matrix_get(mat_2, i,0)*2;
      integral += gsl_matrix_get(mat_1, i,N-1)*gsl_matrix_get(mat_2, i,N-1)*2;

    }
  }

  if (std::signbit(integral)) {
    integral = -1.0*std::exp(std::log(std::abs(integral)) + 2*std::log(dx_) - std::log(4.0));
  } else {
    integral = std::exp(std::log(std::abs(integral)) + 2*std::log(dx_) - std::log(4.0));
  }
  return integral;
  // TRAPEZOIDAL RULE END

  // for (int i=0; i<N; ++i) {
  //   double row_sum = 0;
  //   double product = 0;

  //   //     //     //     //     //     //     //      //
  //   gsl_vector_const_view row_i_1 =
  //     gsl_matrix_const_row(elem_1.get_function_grid(),
  // 			   i);
  //   gsl_vector_const_view row_ip1_1 =
  //     gsl_matrix_const_row(elem_1.get_function_grid(),
  // 			   i+1);

  //   gsl_vector_const_view row_i_j_1 =
  //     gsl_vector_const_subvector(&row_i_1.vector, 0, N);
  //   gsl_vector_const_view row_i_jp1_1 =
  //     gsl_vector_const_subvector(&row_i_1.vector, 1, N);
  //   gsl_vector_const_view row_ip1_j_1 =
  //     gsl_vector_const_subvector(&row_ip1_1.vector, 0, N);
  //   gsl_vector_const_view row_ip1_jp1_1 =
  //     gsl_vector_const_subvector(&row_ip1_1.vector, 1, N);
  //   // //    //     //     //     //     //     //
  //   // //    //     //     //     //     //     //
  //   gsl_vector_const_view row_i_2 =
  //     gsl_matrix_const_row(elem_2.get_function_grid(),
  // 			   i);
  //   gsl_vector_const_view row_ip1_2 =
  //     gsl_matrix_const_row(elem_2.get_function_grid(),
  // 			   i+1);

  //   gsl_vector_const_view row_i_j_2 =
  //     gsl_vector_const_subvector(&row_i_2.vector, 0, N);
  //   gsl_vector_const_view row_i_jp1_2 =
  //     gsl_vector_const_subvector(&row_i_2.vector, 1, N);
  //   gsl_vector_const_view row_ip1_j_2 =
  //     gsl_vector_const_subvector(&row_ip1_2.vector, 0, N);
  //   gsl_vector_const_view row_ip1_jp1_2 =
  //     gsl_vector_const_subvector(&row_ip1_2.vector, 1, N);
  //   //     //     //     //     //     //

  //   // f_11
  //   // f_11 f_11
  //   gsl_blas_ddot(&row_i_j_1.vector, &row_i_j_2.vector, &product);
  //   row_sum += product * 1.0/9.0;
  //   // f_11 f_21
  //   gsl_blas_ddot(&row_i_j_1.vector, &row_ip1_j_2.vector, &product);
  //   row_sum += product * 1.0/18.0;
  //   // f_11 f_12
  //   gsl_blas_ddot(&row_i_j_1.vector, &row_i_jp1_2.vector, &product);
  //   row_sum += product * 1.0/18.0;
  //   // f_11 f_22
  //   gsl_blas_ddot(&row_i_j_1.vector, &row_ip1_jp1_2.vector, &product);
  //   row_sum += product * 1.0/36;
  //   //
  //   // f_21
  //   // f_21 f_11
  //   gsl_blas_ddot(&row_ip1_j_1.vector, &row_i_j_2.vector, &product);
  //   row_sum += product * 1.0/18.0;
  //   // f_21 f_21
  //   gsl_blas_ddot(&row_ip1_j_1.vector, &row_ip1_j_2.vector, &product);
  //   row_sum += product * 1.0/9.0;
  //   // f_21 f_12
  //   gsl_blas_ddot(&row_ip1_j_1.vector, &row_i_jp1_2.vector, &product);
  //   row_sum += product * 1.0/36;
  //   // f_21 f_22
  //   gsl_blas_ddot(&row_ip1_j_1.vector, &row_ip1_jp1_2.vector, &product);
  //   row_sum += product * 1.0/18.0;
  //   //
  //   // f_12
  //   // f_12 f_11
  //   gsl_blas_ddot(&row_i_jp1_1.vector, &row_i_j_2.vector, &product);
  //   row_sum += product * 1.0/18.0;
  //   // f_12 f_21
  //   gsl_blas_ddot(&row_i_jp1_1.vector, &row_ip1_j_2.vector, &product);
  //   row_sum += product * 1.0/36.0;
  //   // f_12 f_12
  //   gsl_blas_ddot(&row_i_jp1_1.vector, &row_i_jp1_2.vector, &product);
  //   row_sum += product * 1.0/9.0;
  //   // f_12 f_22
  //   gsl_blas_ddot(&row_i_jp1_1.vector, &row_ip1_jp1_2.vector, &product);
  //   row_sum += product * 1.0/18.0;
  //   //
  //   // f_22
  //   // f_22 f_11
  //   gsl_blas_ddot(&row_ip1_jp1_1.vector, &row_i_j_2.vector, &product);
  //   row_sum += product * 1.0/36.0;
  //   // f_22 f_21
  //   gsl_blas_ddot(&row_ip1_jp1_1.vector, &row_ip1_j_2.vector, &product);
  //   row_sum += product * 1.0/18.0;
  //   // f_22 f_12
  //   gsl_blas_ddot(&row_ip1_jp1_1.vector, &row_i_jp1_2.vector, &product);
  //   row_sum += product * 1.0/18.0;
  //   // f_22 f_22
  //   gsl_blas_ddot(&row_ip1_jp1_1.vector, &row_ip1_jp1_2.vector, &product);
  //   row_sum += product * 1.0/9.0;
  //   //

  //   integral = integral + row_sum;
  // }

  // if (std::signbit(integral)) {
  //   integral = -1.0*std::exp(std::log(std::abs(integral)) + 2*std::log(dx_));
  // } else {
  //   integral = std::exp(std::log(std::abs(integral)) + 2*std::log(dx_));
  // }
  // return integral;
}

double BivariateGaussianKernelBasis::
project_deriv(const BivariateElement& elem_1,
	      int coord_indeex_1,
	      const BivariateElement& elem_2,
	      int coord_indeex_2) const
{
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

  int N = 1.0/dx_;
  double integral = 0;

  // // http://mathfaculty.fullerton.edu/mathews/n2003/SimpsonsRule2DMod.html
  // // SIMPSON'S RULE START 
  // const gsl_matrix* mat_1 = elem_1_deriv_mat;
  // const gsl_matrix* mat_2 = elem_2_deriv_mat;

  // integral += gsl_matrix_get(mat_1, 0,0)*gsl_matrix_get(mat_2, 0,0) +
  //   gsl_matrix_get(mat_1, 0,N)*gsl_matrix_get(mat_2, 0,N) +
  //   gsl_matrix_get(mat_1, N,0)*gsl_matrix_get(mat_2, N,0) +
  //   gsl_matrix_get(mat_1, N,N)*gsl_matrix_get(mat_2, N,N);

  // for (int i=1; i<=N/2; ++i) {
  //   if (i<N/2) {
  //     integral += 
  // 	4.0*gsl_matrix_get(mat_1, 0,2*i-1)*gsl_matrix_get(mat_2, 0,2*i-1) +
  // 	2.0*gsl_matrix_get(mat_1, 0,2*i)*gsl_matrix_get(mat_2, 0,2*i) +
  // 	// // 
  // 	4.0*gsl_matrix_get(mat_1, N,2*i-1)*gsl_matrix_get(mat_2, N,2*i-1) +
  // 	2.0*gsl_matrix_get(mat_1, N,2*i)*gsl_matrix_get(mat_2, N,2*i) +
  // 	//
  // 	4.0*gsl_matrix_get(mat_1, 2*i-1,0)*gsl_matrix_get(mat_2, 2*i-1,0) +
  // 	2.0*gsl_matrix_get(mat_1, 2*i,0)*gsl_matrix_get(mat_2, 2*i,0) +
  // 	// // 
  // 	4.0*gsl_matrix_get(mat_1, 2*i-1,N)*gsl_matrix_get(mat_2, 2*i-1,N) +
  // 	2.0*gsl_matrix_get(mat_1, 2*i,N)*gsl_matrix_get(mat_2, 2*i,N);

  //     for (int j=1; j<=N/2; ++j) {
  // 	if (j<N/2) {
  // 	  integral +=
  // 	    16.0*gsl_matrix_get(mat_1, 2*i-1, 2*j-1)*gsl_matrix_get(mat_2, 2*i-1,2*j-1) +
  // 	    8.0*gsl_matrix_get(mat_1, 2*i, 2*j-1)*gsl_matrix_get(mat_2, 2*i,2*j-1) +
  // 	    //
  // 	    8.0*gsl_matrix_get(mat_1, 2*i-1, 2*j)*gsl_matrix_get(mat_2, 2*i-1,2*j) +
  // 	    4.0*gsl_matrix_get(mat_1, 2*i, 2*j)*gsl_matrix_get(mat_2, 2*i,2*j);
  // 	} else {
  // 	  integral +=
  // 	    16.0*gsl_matrix_get(mat_1, 2*i-1, 2*j-1)*gsl_matrix_get(mat_2, 2*i-1,2*j-1) +
  // 	    8.0*gsl_matrix_get(mat_1, 2*i-1, 2*j)*gsl_matrix_get(mat_2, 2*i-1,2*j);
  // 	}
  //     }

  //   } else {
  //     integral += 
  // 	4.0*gsl_matrix_get(mat_1, 0,2*i-1)*gsl_matrix_get(mat_2, 0,2*i-1) +
  // 	// // 
  // 	4.0*gsl_matrix_get(mat_1, N,2*i-1)*gsl_matrix_get(mat_2, N,2*i-1) +
  // 	//
  // 	4.0*gsl_matrix_get(mat_1, 2*i-1,0)*gsl_matrix_get(mat_2, 2*i-1,0) +
  // 	// // 
  // 	4.0*gsl_matrix_get(mat_1, 2*i-1,N)*gsl_matrix_get(mat_2, 2*i-1,N);

  //     for (int j=1; j<=N/2; ++j) {
  // 	if (j<N/2) {
  // 	  integral +=
  // 	    16.0*gsl_matrix_get(mat_1, 2*i-1, 2*j-1)*gsl_matrix_get(mat_2, 2*i-1,2*j-1) +
  // 	    8.0*gsl_matrix_get(mat_1, 2*i, 2*j-1)*gsl_matrix_get(mat_2, 2*i,2*j-1);
  // 	} else {
  // 	  integral +=
  // 	    16.0*gsl_matrix_get(mat_1, 2*i-1, 2*j-1)*gsl_matrix_get(mat_2, 2*i-1,2*j-1);
  // 	}
  //     }
  //   }
  // }

  // if (std::signbit(integral)) {
  //   integral = -1.0*std::exp(std::log(std::abs(integral)) + 2*std::log(dx_) - std::log(9.0));
  // } else {
  //   integral = std::exp(std::log(std::abs(integral)) + 2*std::log(dx_) - std::log(9.0));
  // }
  // return integral;
  // // SIMPSON RULE END

  // TRAPEZOIDAL RULE START 
  const gsl_matrix* mat_1 = elem_1_deriv_mat;
  const gsl_matrix* mat_2 = elem_2_deriv_mat;

  for (int i=0; i<N; ++i) {
    if (i==0 || i==N-1) {

      for (int j=1; j<N-1; ++j) {
  	integral += gsl_matrix_get(mat_1, i,j)*gsl_matrix_get(mat_2, i,j)*2;
      }
      integral += gsl_matrix_get(mat_1, i,0)*gsl_matrix_get(mat_2, i,0)*1;
      integral += gsl_matrix_get(mat_1, i,N-1)*gsl_matrix_get(mat_2, i,N-1)*1;

    } else {

      for (int j=1; j<N-1; ++j) {
  	integral += gsl_matrix_get(mat_1, i,j)*gsl_matrix_get(mat_2, i,j)*4;
      }
      integral += gsl_matrix_get(mat_1, i,0)*gsl_matrix_get(mat_2, i,0)*2;
      integral += gsl_matrix_get(mat_1, i,N-1)*gsl_matrix_get(mat_2, i,N-1)*2;

    }
  }

  if (std::signbit(integral)) {
    integral = -1.0*std::exp(std::log(std::abs(integral)) + 2*std::log(dx_) - std::log(4.0));
  } else {
    integral = std::exp(std::log(std::abs(integral)) + 2*std::log(dx_) - std::log(4.0));
  }
  return integral;
  // TRAPEZOIDAL RULE END

  // if (coord_indeex_1 == 0 && coord_indeex_2 == 0) {
  //   for (int i=0; i<N-1; ++i) {
  //     for (int j=0; j<N-1; ++j) {
  //   	integral = integral +
  //   	  (gsl_matrix_get(elem_1_mat,
  //   			  i + 1,
  //   			  j) -
  //   	 gsl_matrix_get(elem_1_mat, i, j))*
  //   	  (gsl_matrix_get(elem_2_mat,
  //   			  i + 1,
  //   			  j) -
  //   	   gsl_matrix_get(elem_2_mat, i, j));
  //     }
  //   }
  // } else if (coord_indeex_1 == 0 && coord_indeex_2 == 1) {
  //   for (int i=0; i<N-1; ++i) {
  //     for (int j=0; j<N-1; ++j) {
  //   	integral = integral +
  //   	  (gsl_matrix_get(elem_1_mat,
  //   			  i + 1,
  //   			  j) -
  //   	 gsl_matrix_get(elem_1_mat, i, j))*
  //   	  (gsl_matrix_get(elem_2_mat,
  //   			  i,
  //   			  j + 1) -
  //   	   gsl_matrix_get(elem_2_mat, i, j));
  //     }
  //   }
  // } else if (coord_indeex_1 == 1 && coord_indeex_2 == 0) {
  //   for (int i=0; i<N-1; ++i) {
  //     for (int j=0; j<N-1; ++j) {
  // 	integral = integral +
  // 	  (gsl_matrix_get(elem_1_mat,
  // 			  i,
  // 			  j + 1) -
  // 	 gsl_matrix_get(elem_1_mat, i, j))*
  // 	  (gsl_matrix_get(elem_2_mat,
  // 			  i + 1,
  // 			  j) -
  // 	   gsl_matrix_get(elem_2_mat, i, j));
  //     }
  //   }
  // } else if (coord_indeex_1 == 1 && coord_indeex_2 == 1) {
  //   for (int i=0; i<N-1; ++i) {
  //     for (int j=0; j<N-1; ++j) {
  // 	integral = integral +
  // 	  (gsl_matrix_get(elem_1_mat,
  // 			  i,
  // 			  j + 1) -
  // 	 gsl_matrix_get(elem_1_mat, i, j))*
  // 	  (gsl_matrix_get(elem_2_mat,
  // 			  i,
  // 			  j + 1) -
  // 	   gsl_matrix_get(elem_2_mat, i, j));
  //     }
  //   }
  // } else {
  //   std::cout << "WRONG COORD INPUT!" << std::endl;
  // }
}

void BivariateGaussianKernelBasis::save_matrix(const gsl_matrix* mat,
					       std::string file_name) const
{
  std::ofstream output_file;
  output_file.open(file_name);
  output_file << std::fixed << std::setprecision(32);
  for (unsigned i=0; i<mat->size1; ++i) {
    for (unsigned j=0; j<mat->size2; ++j) {
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
  set_basis_functions(rho,
		      sigma,
		      sigma,
		      power,
		      std_dev_factor);
}

void BivariateGaussianKernelBasis::set_basis_functions(double rho,
						       double sigma_x,
						       double sigma_y,
						       double power,
						       double std_dev_factor)
{
  // GIVEN compututational domain Omega = [0,1]\times[0,1] and a
  // kernel centered on (1/2,1/2) with parameters
  // (sigma_x,sigma_y,rho), we transform such that kernel has parameters (\sqrt{1-\rho},\sqrt{1+\rho},0) via
  // SCALE: (x,y) -> (x/\sigma_x, y/\sigma_y) := (xi(1), eta(1))
  // ROTATE: (xi(1), eta(1)) ->  \sqrt{2}/2 (xi(1)-eta(1), xi(2)+eta(2)) = \sqrt{2}/2 (x/\sigma_x-y/\sigma_y, x/\sigma_x+y/\sigma_y) := (xi(2), eta(2))

  // We lay out grid on (xi(2), eta(2)) centered on 
  //
  // \sqrt{2}/2 \cdot 1/2 \cdot (1/\sigma_x-1/\sigma_y, 1/\sigma_x+1/\sigma_y)
  //
  // [(1/2,1/2) transformed], where each nodes is separated by a std
  // dev factor \times \sqrt{1-\rho} in the xi(2) direction and dev
  // factor \times \sqrt{1+\rho} in the eta(2) direction. Further, the nodes are bounded by
  // 0 \leq \eta(2) \leq \sqrt{2}/2(1/\sigma_x + 1/\sigma_y)
  // -\sqrt{2}/2 \codt 1/\sigma_y \leq xi(2) \leq \sqrt{2}/2 \codt 1/\sigma_x
  //
  // The above inequalities ensure an upper bound on the presence of
  // nodes within \Omega when we transform back.
  //
  // Given the nodes, we transform back to (x,y) frame by rotating
  // back and re-scaling. We finally filter on the nodes so that all
  // are within \Omega.
  //
  // For computational reasons (mainly to allow for nodes at the
  // corners), we extend Omega to Omega = [-epsilon,1+epsilon] \times
  // [-epsilon,1+epsilon]

  double epsilon = 0.10; 
  // corners will include the midpoint... 
                           //low.right, low.left,   up.left,     up.right
  double corners [2*5] = {-epsilon, 1.0+epsilon, 1.0+epsilon, -epsilon, 0.5,
			  -epsilon, -epsilon,    1.0+epsilon, 1.0+epsilon, 0.5};
  gsl_matrix_view corners_view = gsl_matrix_view_array(corners,2,5);

  double corners_xi [2*5];
  gsl_matrix_view corners_xi_view = gsl_matrix_view_array(corners_xi,2,5);

  double theta = M_PI/4.0;
  gsl_matrix *Rotation_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_set(Rotation_matrix, 0, 0, 1.0/sigma_x * std::cos(theta));
  gsl_matrix_set(Rotation_matrix, 1, 0, 1.0/sigma_x * std::sin(theta));
  gsl_matrix_set(Rotation_matrix, 0, 1, 1.0/sigma_y * -1.0*std::sin(theta));
  gsl_matrix_set(Rotation_matrix, 1, 1, 1.0/sigma_y * std::cos(theta));

  gsl_blas_dgemm(CblasNoTrans, 
		 CblasNoTrans,
		 1.0, Rotation_matrix, &corners_view.matrix, 0.0,
		 &corners_xi_view.matrix);

  gsl_vector_view corner_xi_coordinates = gsl_matrix_row(&corners_xi_view.matrix, 0);
  double xi_min = gsl_vector_min(&corner_xi_coordinates.vector);
  double xi_max = gsl_vector_max(&corner_xi_coordinates.vector);

  gsl_vector_view corner_eta_coordinates = gsl_matrix_row(&corners_xi_view.matrix, 1);
  double eta_min = gsl_vector_min(&corner_eta_coordinates.vector);
  double eta_max = gsl_vector_max(&corner_eta_coordinates.vector);

  gsl_vector_view midpoint_xieta_coordinates = gsl_matrix_column(&corners_xi_view.matrix, 4);
  double xi_midpoint = gsl_vector_get(&midpoint_xieta_coordinates.vector,0);
  double eta_midpoint = gsl_vector_get(&midpoint_xieta_coordinates.vector,1);

  //  creating the x-nodes
  //
  // if rho is negative
  double by = std_dev_factor*std::sqrt(1.0 - rho);
  // if (std::signbit(rho)) {
  //   by = std_dev_factor;
  // } else {
  //   by = std_dev_factor*std::sqrt(1.0-rho)/std::sqrt(1.0+rho);
  // }
  unsigned N = std::ceil( (xi_midpoint - xi_min)/by );
  std::vector<double> xi_nodes (N);
  double xi_current = xi_midpoint + by;
  std::generate(xi_nodes.begin(),
		xi_nodes.end(),
		[&] ()->double {xi_current = xi_current - by; return xi_current; });
  N = std::ceil( (xi_max - xi_midpoint)/by );
  xi_current = xi_midpoint + by;
  for (unsigned i=0; i<N; ++i) {
    xi_nodes.push_back(xi_current);
    xi_current = xi_current + by;
  }
  std::sort(xi_nodes.begin(), xi_nodes.end());
  auto last = std::unique(xi_nodes.begin(), xi_nodes.end());
  xi_nodes.erase(last, xi_nodes.end());

  // creating the y-nodes
  // if (rho >= 0.0) {
  //   by = std_dev_factor * sigma * std::sqrt(1-rho) / std::sqrt(1+rho);
  // } else {
  //   by = std_dev_factor * sigma * std::sqrt(1+rho) / std::sqrt(1-rho);
  // }
  by = std_dev_factor*std::sqrt(1.0 + rho);
  // if (std::signbit(rho)) {
  //   by = std_dev_factor*std::sqrt(1.0+rho)/std::sqrt(1.0-rho);
  // } else {
  //   by = std_dev_factor;
  // }
  unsigned M = std::ceil( (eta_midpoint - eta_min)/by );
  std::vector<double> eta_nodes (M);
  double eta_current = eta_midpoint + by;
  std::generate(eta_nodes.begin(),
		eta_nodes.end(),
		[&] ()->double {eta_current = eta_current - by; return eta_current; });
  M = std::ceil( (eta_max - eta_midpoint)/by );
  eta_current = eta_midpoint + by;
  for (unsigned i=0; i<M; ++i) {
    eta_nodes.push_back(eta_current);
    eta_current = eta_current + by;
  }
  std::sort(eta_nodes.begin(), eta_nodes.end());
  last = std::unique(eta_nodes.begin(), eta_nodes.end());
  eta_nodes.erase(last, eta_nodes.end());

  gsl_matrix *xy_nodes = gsl_matrix_alloc(2, xi_nodes.size()*eta_nodes.size());
  gsl_matrix *xieta_nodes = gsl_matrix_alloc(2, xi_nodes.size()*eta_nodes.size());

  for (unsigned i=0; i<xi_nodes.size(); ++i) {
    for (unsigned j=0; j<eta_nodes.size(); ++j) {
      gsl_matrix_set(xieta_nodes, 0, i*eta_nodes.size()+j, xi_nodes[i]);
      gsl_matrix_set(xieta_nodes, 1, i*eta_nodes.size()+j, eta_nodes[j]);
    }
  }

  gsl_matrix_set(Rotation_matrix, 0, 0, sigma_x * std::cos(-theta));
  gsl_matrix_set(Rotation_matrix, 1, 0, sigma_y * std::sin(-theta));
  gsl_matrix_set(Rotation_matrix, 0, 1, sigma_x * -1.0*std::sin(-theta));
  gsl_matrix_set(Rotation_matrix, 1, 1, sigma_y * std::cos(-theta));

  // gsl_vector_view xi_nodes_view = gsl_matrix_row(xieta_nodes, 0);
  // gsl_vector_view eta_nodes_view = gsl_matrix_row(xieta_nodes, 1);
  // gsl_vector_add_constant(&xi_nodes_view.vector, -0.5/sigma_x);
  // gsl_vector_add_constant(&eta_nodes_view.vector, -0.5/sigma_y);

  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		 1.0, Rotation_matrix, xieta_nodes, 0.0,
		 xy_nodes);

  // gsl_vector_view x_nodes_view = gsl_matrix_row(xy_nodes, 0);
  // gsl_vector_view y_nodes_view = gsl_matrix_row(xy_nodes, 1);
  // gsl_vector_add_constant(&xi_nodes_view.vector, 0.5);
  // gsl_vector_add_constant(&eta_nodes_view.vector, 0.5);

  std::vector<long unsigned> indeces_within_boundary (0);
  for (unsigned j=0; j<xi_nodes.size()*eta_nodes.size(); ++j) {
    if ( (gsl_matrix_get(xy_nodes, 0, j) >= -epsilon) && // left
	 (gsl_matrix_get(xy_nodes, 0, j) <= (1.0 + epsilon) ) && // right
	 (gsl_matrix_get(xy_nodes, 1, j) >= -epsilon) && // bottom
	 (gsl_matrix_get(xy_nodes, 1, j) <= (1.0 + epsilon)) ) // top
      {
	indeces_within_boundary.push_back(j);
      }
  }
  printf("Number of basis elements = %lu\n", indeces_within_boundary.size());

  gsl_vector* mean_vector = gsl_vector_alloc(2);
  gsl_matrix* covariance_matrix = gsl_matrix_alloc(2,2);

  gsl_matrix_set(covariance_matrix, 0, 0, std::pow(sigma_x, 2));
  gsl_matrix_set(covariance_matrix, 1, 0, rho*sigma_x*sigma_y);
  gsl_matrix_set(covariance_matrix, 0, 1, rho*sigma_x*sigma_y);
  gsl_matrix_set(covariance_matrix, 1, 1, std::pow(sigma_y, 2));

  std::vector<BivariateGaussianKernelElement> basis_functions_ =
    std::vector<BivariateGaussianKernelElement> (indeces_within_boundary.size());

  for (unsigned i=0; i<indeces_within_boundary.size(); ++i) {
    unsigned const& index = indeces_within_boundary[i];
    gsl_vector_set(mean_vector, 0,
		   gsl_matrix_get(xy_nodes, 0, index));
    gsl_vector_set(mean_vector, 1,
		   gsl_matrix_get(xy_nodes, 1, index));
    basis_functions_[i] = BivariateGaussianKernelElement(dx_,
    							 power,
    							 mean_vector,
    							 covariance_matrix);
    
    printf("basis.%d = c(%g,%g); points(basis.%d[1], basis.%d[2], col=3, pch=20);\n",
    	   i,
    	   gsl_matrix_get(xy_nodes, 0, index),
    	   gsl_matrix_get(xy_nodes, 1, index),
    	   i,i);
  }

  // SETTING ORTHONORMAL ELEMENTS
  printf("begin Graham-Schmidt\n");
  set_orthonormal_functions_stable(basis_functions_);
  printf("end Graham-Schmidt\n");

  gsl_matrix_free(xy_nodes);
  gsl_matrix_free(xieta_nodes);
  gsl_matrix_free(Rotation_matrix);

  gsl_vector_free(mean_vector);
  gsl_matrix_free(covariance_matrix);
}

// Performing Gram-Schmidt Orthogonalization
void BivariateGaussianKernelBasis::
set_orthonormal_functions_stable(const std::vector<BivariateGaussianKernelElement>& basis_functions)
{
  // HAVE A MATRIX VIEW HERE ON THE STACK!
  gsl_matrix* workspace_left = gsl_matrix_alloc(1/dx_ + 1, 1/dx_ + 1);
  gsl_matrix* workspace_right = gsl_matrix_alloc(1/dx_ + 1, 1/dx_ + 1);

  unsigned COUNTER = 0;
  auto t1 = std::chrono::high_resolution_clock::now();
  for (unsigned i=0; i<basis_functions.size(); ++i) {
    std::vector<double> coefficients(i+1, 0.0);
    std::vector<const BivariateElement*> elements(i+1);

    for (unsigned j=0; j<i+1; ++j) {
      elements[j] = &basis_functions[j];
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
      COUNTER ++;
      if (COUNTER == basis_functions.size()) {
	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "Performing " << COUNTER << " projections took "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " milliseconds." << std::endl;
      }

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

      // double current_norm = current_orthonormal_element.norm();
      double current_norm = std::sqrt(project(current_orthonormal_element,
					      current_orthonormal_element));

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

      // SETTING DERIV_FUNCTION_GRID_DY_
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
    }

    // double current_norm = current_orthonormal_element.norm();
    double current_norm = std::sqrt(project(current_orthonormal_element,
					    current_orthonormal_element));

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

    orthonormal_functions_.push_back(current_orthonormal_element);
  }

  gsl_matrix_free(workspace_left);
  gsl_matrix_free(workspace_right);
}

void BivariateGaussianKernelBasis::set_mass_matrix()
{
  gsl_matrix_free(mass_matrix_);
  mass_matrix_ = gsl_matrix_alloc(orthonormal_functions_.size(),
				  orthonormal_functions_.size());
  double entry = 0;

  for (unsigned i=0; i<orthonormal_functions_.size(); ++i) {
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
      gsl_matrix_set(mass_matrix_,
  			i, j,
			entry);
      gsl_matrix_set(mass_matrix_,
  			j, i,
			entry);
    }
  }
}

void BivariateGaussianKernelBasis::set_system_matrices_stable()
{
  gsl_matrix_free(deriv_inner_product_matrix_dx_dx_);
  deriv_inner_product_matrix_dx_dx_ =
    gsl_matrix_alloc(orthonormal_functions_.size(),
		     orthonormal_functions_.size());

  gsl_matrix_free(deriv_inner_product_matrix_dx_dy_);
  deriv_inner_product_matrix_dx_dy_ =
    gsl_matrix_alloc(orthonormal_functions_.size(),
		     orthonormal_functions_.size());

  gsl_matrix_free(deriv_inner_product_matrix_dy_dx_);
  deriv_inner_product_matrix_dy_dx_ =
    gsl_matrix_alloc(orthonormal_functions_.size(),
		     orthonormal_functions_.size());

  gsl_matrix_free(deriv_inner_product_matrix_dy_dy_);
  deriv_inner_product_matrix_dy_dy_ =
    gsl_matrix_alloc(orthonormal_functions_.size(),
		     orthonormal_functions_.size());

  gsl_matrix_free(system_matrix_dx_dx_);
  system_matrix_dx_dx_ =
    gsl_matrix_alloc(orthonormal_functions_.size(),
		     orthonormal_functions_.size());

  gsl_matrix_free(system_matrix_dx_dy_);
  system_matrix_dx_dy_ =
    gsl_matrix_alloc(orthonormal_functions_.size(),
		     orthonormal_functions_.size());

  gsl_matrix_free(system_matrix_dy_dx_);
  system_matrix_dy_dx_ =
    gsl_matrix_alloc(orthonormal_functions_.size(),
		     orthonormal_functions_.size());

  gsl_matrix_free(system_matrix_dy_dy_);
  system_matrix_dy_dy_ =
    gsl_matrix_alloc(orthonormal_functions_.size(),
		     orthonormal_functions_.size());

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
}

std::ostream& operator<<(std::ostream& os, const BivariateGaussianKernelBasis& current_basis)
{
  // creating a list of files within a file that contains, in the
  // following order:
  // * unsigned number orthonormal functions
  // ** list of files containing files for the orthonormal functions
  // *  system_matrix_dx_dx_;
  // *  system_matrix_dx_dy_;
  // *  system_matrix_dy_dx_;
  // *  system_matrix_dy_dy_;
  // *  mass_matrix_;
  // *  inner_product_matrix_;
  // *  deriv_inner_product_matrix_dx_dx_;
  // *  deriv_inner_product_matrix_dx_dy_;
  // *  deriv_inner_product_matrix_dy_dx_;
  // *  deriv_inner_product_matrix_dy_dy_;

  os << current_basis.orthonormal_functions_.size() << "\n";

  for (unsigned i=0; i<current_basis.orthonormal_functions_.size(); ++i) {
    std::string current_orthonormal_function =
      "orthonormal_function_" + std::to_string(i) + ".csv";

    current_basis.orthonormal_functions_[i].save_function_grid(current_orthonormal_function);
    os << current_orthonormal_function << "\n";
  }

  return os;
}

void BivariateGaussianKernelBasis::set_simpsons_rule()
{
  int N = 1.0/dx_;
  gsl_matrix_free(integration_rule_multiplier_);
  integration_rule_multiplier_ = gsl_matrix_alloc(N+1,N+1);

  // http://mathfaculty.fullerton.edu/mathews/n2003/SimpsonsRule2DMod.html
  // SIMPSON'S RULE START 
  gsl_matrix_set(integration_rule_multiplier_, 0,0, 1.0);
  gsl_matrix_set(integration_rule_multiplier_, 0,N, 1.0);
  gsl_matrix_set(integration_rule_multiplier_, N,0, 1.0);
  gsl_matrix_set(integration_rule_multiplier_, N,N, 1.0);
  for (int i=1; i<=N/2; ++i) {
    if (i<N/2) {
      gsl_matrix_set(integration_rule_multiplier_, 0, 2*i-1, 4.0);
      gsl_matrix_set(integration_rule_multiplier_, 0, 2*i  , 2.0);
      // // 
      gsl_matrix_set(integration_rule_multiplier_, N, 2*i-1, 4.0);
      gsl_matrix_set(integration_rule_multiplier_, N, 2*i,   2.0);
      // //
      gsl_matrix_set(integration_rule_multiplier_, 2*i-1, 0, 4.0);
      gsl_matrix_set(integration_rule_multiplier_, 2*i,   0, 2.0);
      // // 
      gsl_matrix_set(integration_rule_multiplier_, 2*i-1, N, 4.0);
      gsl_matrix_set(integration_rule_multiplier_, 2*i,   N, 2.0);
      // //
      for (int j=1; j<=N/2; ++j) {
  	if (j<N/2) {
	  gsl_matrix_set(integration_rule_multiplier_, 2*i-1, 2*j-1, 16);
	  gsl_matrix_set(integration_rule_multiplier_, 2*i, 2*j-1, 8.0);
	  //
	  gsl_matrix_set(integration_rule_multiplier_, 2*i-1, 2*j, 8.0);
	  gsl_matrix_set(integration_rule_multiplier_, 2*i, 2*j,4.0);
  	} else {
	  gsl_matrix_set(integration_rule_multiplier_, 2*i-1, 2*j-1, 16.0);
	  gsl_matrix_set(integration_rule_multiplier_, 2*i-1, 2*j, 8.0);
  	}
      }
    } else {
      gsl_matrix_set(integration_rule_multiplier_, 0, 2*i-1, 4.0);
      gsl_matrix_set(integration_rule_multiplier_, N, 2*i-1, 4.0);
      gsl_matrix_set(integration_rule_multiplier_, 2*i-1, 0, 4.0);
      gsl_matrix_set(integration_rule_multiplier_, 2*i-1, N, 4.0);
      
      for (int j=1; j<=N/2; ++j) {
  	if (j<N/2) {
	  gsl_matrix_set(integration_rule_multiplier_, 2*i-1, 2*j-1, 16.0);
	  gsl_matrix_set(integration_rule_multiplier_, 2*i, 2*j-1, 8.0);
  	} else {
	  gsl_matrix_set(integration_rule_multiplier_, 2*i-1, 2*j-1, 16.0);
  	}
      }
    }
  }
  gsl_matrix_scale(integration_rule_multiplier_, dx_*dx_ / 9.0);
  // SIMPSON RULE END  
}
