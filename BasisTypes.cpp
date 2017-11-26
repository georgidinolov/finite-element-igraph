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
    deriv_inner_product_matrix_dy_dy_(gsl_matrix_alloc(1,1))
{
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
    deriv_inner_product_matrix_dy_dy_(gsl_matrix_alloc(1,1))
{
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
    deriv_inner_product_matrix_dy_dy_(gsl_matrix_alloc(1,1))
{
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
						       basis.deriv_inner_product_matrix_dy_dy_->size2))
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
  for (int i=0; i<N; ++i) {
    double row_sum = 0;
    double product = 0;

    //     //     //     //     //     //     //      //
    gsl_vector_const_view row_i_1 =
      gsl_matrix_const_row(elem_1.get_function_grid(),
			   i);
    gsl_vector_const_view row_ip1_1 =
      gsl_matrix_const_row(elem_1.get_function_grid(),
			   i+1);

    gsl_vector_const_view row_i_j_1 =
      gsl_vector_const_subvector(&row_i_1.vector, 0, N);
    gsl_vector_const_view row_i_jp1_1 =
      gsl_vector_const_subvector(&row_i_1.vector, 1, N);
    gsl_vector_const_view row_ip1_j_1 =
      gsl_vector_const_subvector(&row_ip1_1.vector, 0, N);
    gsl_vector_const_view row_ip1_jp1_1 =
      gsl_vector_const_subvector(&row_ip1_1.vector, 1, N);
    // //    //     //     //     //     //     //
    // //    //     //     //     //     //     //
    gsl_vector_const_view row_i_2 =
      gsl_matrix_const_row(elem_2.get_function_grid(),
			   i);
    gsl_vector_const_view row_ip1_2 =
      gsl_matrix_const_row(elem_2.get_function_grid(),
			   i+1);

    gsl_vector_const_view row_i_j_2 =
      gsl_vector_const_subvector(&row_i_2.vector, 0, N);
    gsl_vector_const_view row_i_jp1_2 =
      gsl_vector_const_subvector(&row_i_2.vector, 1, N);
    gsl_vector_const_view row_ip1_j_2 =
      gsl_vector_const_subvector(&row_ip1_2.vector, 0, N);
    gsl_vector_const_view row_ip1_jp1_2 =
      gsl_vector_const_subvector(&row_ip1_2.vector, 1, N);
    //     //     //     //     //     //

    // f_11
    // f_11 f_11
    gsl_blas_ddot(&row_i_j_1.vector, &row_i_j_2.vector, &product);
    row_sum += product * 1.0/9.0;
    // f_11 f_21
    gsl_blas_ddot(&row_i_j_1.vector, &row_ip1_j_2.vector, &product);
    row_sum += product * 1.0/18.0;
    // f_11 f_12
    gsl_blas_ddot(&row_i_j_1.vector, &row_i_jp1_2.vector, &product);
    row_sum += product * 1.0/18.0;
    // f_11 f_22
    gsl_blas_ddot(&row_i_j_1.vector, &row_ip1_jp1_2.vector, &product);
    row_sum += product * 1.0/36;
    //
    // f_21
    // f_21 f_11
    gsl_blas_ddot(&row_ip1_j_1.vector, &row_i_j_2.vector, &product);
    row_sum += product * 1.0/18.0;
    // f_21 f_21
    gsl_blas_ddot(&row_ip1_j_1.vector, &row_ip1_j_2.vector, &product);
    row_sum += product * 1.0/9.0;
    // f_21 f_12
    gsl_blas_ddot(&row_ip1_j_1.vector, &row_i_jp1_2.vector, &product);
    row_sum += product * 1.0/36;
    // f_21 f_22
    gsl_blas_ddot(&row_ip1_j_1.vector, &row_ip1_jp1_2.vector, &product);
    row_sum += product * 1.0/18.0;
    //
    // f_12
    // f_12 f_11
    gsl_blas_ddot(&row_i_jp1_1.vector, &row_i_j_2.vector, &product);
    row_sum += product * 1.0/18.0;
    // f_12 f_21
    gsl_blas_ddot(&row_i_jp1_1.vector, &row_ip1_j_2.vector, &product);
    row_sum += product * 1.0/36.0;
    // f_12 f_12
    gsl_blas_ddot(&row_i_jp1_1.vector, &row_i_jp1_2.vector, &product);
    row_sum += product * 1.0/9.0;
    // f_12 f_22
    gsl_blas_ddot(&row_i_jp1_1.vector, &row_ip1_jp1_2.vector, &product);
    row_sum += product * 1.0/18.0;
    //
    // f_22
    // f_22 f_11
    gsl_blas_ddot(&row_ip1_jp1_1.vector, &row_i_j_2.vector, &product);
    row_sum += product * 1.0/36.0;
    // f_22 f_21
    gsl_blas_ddot(&row_ip1_jp1_1.vector, &row_ip1_j_2.vector, &product);
    row_sum += product * 1.0/18.0;
    // f_22 f_12
    gsl_blas_ddot(&row_ip1_jp1_1.vector, &row_i_jp1_2.vector, &product);
    row_sum += product * 1.0/18.0;
    // f_22 f_22
    gsl_blas_ddot(&row_ip1_jp1_1.vector, &row_ip1_jp1_2.vector, &product);
    row_sum += product * 1.0/9.0;
    //

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
  // creating the x-nodes

  // if rho is negative
  double by = 0;
  double current = 0.5/sigma_x;
  if (std::signbit(rho)) {
    by = std_dev_factor;
  } else {
    by = std_dev_factor*std::sqrt(1.0-rho)/std::sqrt(1.0+rho);
  }
  // by = std_dev_factor*std::sqrt(1.0 - rho);

  std::vector<double> x_nodes (0);
  while ((current - (0.5/sigma_x -
		     std::sqrt( (1.0/std::pow(sigma_x,2)+1.0/std::pow(sigma_y,2))))) >= 1e-32) {
    x_nodes.push_back(current);
    current = current - by;
  }
  std::reverse(x_nodes.begin(), x_nodes.end());

  current = 0.5/sigma_x;
  while ( (current-(0.5/sigma_x +
		    std::sqrt( (1.0/std::pow(sigma_x,2)+1.0/std::pow(sigma_y,2)) ))) <= 1e-32 ) {
    x_nodes.push_back(current);
    current = current + by;
  }

  // x_nodes is already sorted but sorting again
  std::sort(x_nodes.begin(), x_nodes.end());
  auto last = std::unique(x_nodes.begin(), x_nodes.end());
  x_nodes.erase(last, x_nodes.end());

  // creating the y-nodes
  // if (rho >= 0.0) {
  //   by = std_dev_factor * sigma * std::sqrt(1-rho) / std::sqrt(1+rho);
  // } else {
  //   by = std_dev_factor * sigma * std::sqrt(1+rho) / std::sqrt(1-rho);
  // }
  if (std::signbit(rho)) {
    by = std_dev_factor*std::sqrt(1.0+rho)/std::sqrt(1.0-rho);
  } else {
    by = std_dev_factor;
  }
  // by = std_dev_factor*std::sqrt(1.0 + rho);
  current = 0.5/sigma_y;

  std::vector<double> y_nodes;
  while ((current - (0.5/sigma_y -
		     std::sqrt( (1.0/std::pow(sigma_x,2)+1.0/std::pow(sigma_y,2))  ) )) >= 1e-32) {
    y_nodes.push_back(current);
    current = current - by;
  }
  std::reverse(y_nodes.begin(), y_nodes.end());

  current = 0.5/sigma_y;
  while ( (current-(0.5/sigma_y +
		    std::sqrt( (1.0/std::pow(sigma_x,2)+1.0/std::pow(sigma_y,2))))) <= 1e-32 ) {
    y_nodes.push_back(current);
    current = current + by;
  }

  // y_nodes is already sorted but sorting it anyway
  std::sort(y_nodes.begin(), y_nodes.end());
  last = std::unique(y_nodes.begin(), y_nodes.end());
  y_nodes.erase(last, y_nodes.end());

  gsl_matrix *xy_nodes = gsl_matrix_alloc(2, x_nodes.size()*y_nodes.size());
  gsl_matrix *xieta_nodes = gsl_matrix_alloc(2, x_nodes.size()*y_nodes.size());

  for (unsigned i=0; i<x_nodes.size(); ++i) {
    for (unsigned j=0; j<y_nodes.size(); ++j) {
      gsl_matrix_set(xy_nodes, 0, i*y_nodes.size()+j, x_nodes[i]);
      gsl_matrix_set(xy_nodes, 1, i*y_nodes.size()+j, y_nodes[j]);
    }
  }

  double theta = M_PI/4.0;
  // if (rho >= 0.0) {
  //   theta = 1.0*M_PI/4.0;
  // } else {
  //   theta = -1.0*M_PI/4.0;
  // }

  gsl_matrix *Rotation_matrix = gsl_matrix_alloc(2,2);
  // gsl_matrix_set(Rotation_matrix, 0, 0,
  // 		 sigma_x/
  // 		 (std::sqrt(2.0)*std::sqrt(1-rho)));
  // gsl_matrix_set(Rotation_matrix, 0, 1,
  // 		 sigma_x/
  // 		 (std::sqrt(2.0)*std::sqrt(1+rho)));
  // gsl_matrix_set(Rotation_matrix, 1, 0,
  // 		 -1.0*sigma_y/
  // 		 (std::sqrt(2.0)*std::sqrt(1-rho)));
  // gsl_matrix_set(Rotation_matrix, 1, 1,
  // 		 sigma_y/
  // 		 (std::sqrt(2.0)*std::sqrt(1+rho)));
  gsl_matrix_set(Rotation_matrix, 0, 0, sigma_x * std::cos(-theta));
  gsl_matrix_set(Rotation_matrix, 1, 0, sigma_y * std::sin(-theta));
  gsl_matrix_set(Rotation_matrix, 0, 1, sigma_x * -1.0*std::sin(-theta));
  gsl_matrix_set(Rotation_matrix, 1, 1, sigma_y * std::cos(-theta));

  gsl_vector_view x_nodes_view = gsl_matrix_row(xy_nodes, 0);
  gsl_vector_view y_nodes_view = gsl_matrix_row(xy_nodes, 1);

  gsl_vector_add_constant(&x_nodes_view.vector, -0.5/sigma_x);
  gsl_vector_add_constant(&y_nodes_view.vector, -0.5/sigma_y);

  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		 1.0, Rotation_matrix, xy_nodes, 0.0,
		 xieta_nodes);

  gsl_vector_view xi_nodes_view = gsl_matrix_row(xieta_nodes, 0);
  gsl_vector_view eta_nodes_view = gsl_matrix_row(xieta_nodes, 1);

  gsl_vector_add_constant(&xi_nodes_view.vector, 0.5);
  gsl_vector_add_constant(&eta_nodes_view.vector, 0.5);

  std::vector<long unsigned> indeces_within_boundary;
  for (unsigned j=0; j<x_nodes.size()*y_nodes.size(); ++j) {
    if ( (gsl_matrix_get(xieta_nodes, 0, j) >= 1e-32 - 0.5) &&
	 (gsl_matrix_get(xieta_nodes, 0, j) <= 1.0 + 0.5 - 1e-32) &&
	 (gsl_matrix_get(xieta_nodes, 1, j) >= 1e-32 - 0.5) &&
	 (gsl_matrix_get(xieta_nodes, 1, j) <= 1.0 + 0.5 - 1e-32) )
      {
	indeces_within_boundary.push_back(j);
      }
  }
  printf("Number of basis elements = %d\n", indeces_within_boundary.size());

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
		   gsl_matrix_get(xieta_nodes, 0, index));
    gsl_vector_set(mean_vector, 1,
		   gsl_matrix_get(xieta_nodes, 1, index));
    basis_functions_[i] = BivariateGaussianKernelElement(dx_,
    							 power,
    							 mean_vector,
    							 covariance_matrix);
  }

  // SETTING ORTHONORMAL ELEMENTS
  set_orthonormal_functions_stable(basis_functions_);

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
  save_matrix(mass_matrix_,
	      "mass_matrix.csv");
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

  save_matrix(system_matrix_dx_dx_,
	      "system_matrix_dx_dx.csv");
  save_matrix(system_matrix_dx_dy_,
	      "system_matrix_dx_dy.csv");
  save_matrix(system_matrix_dy_dx_,
	      "system_matrix_dy_dx.csv");
  save_matrix(system_matrix_dy_dy_,
	      "system_matixr_dy_dy.csv");
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
