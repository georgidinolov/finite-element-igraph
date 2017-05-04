#include <algorithm>
#include "BasisTypes.hpp"
#include <fstream>
#include <iostream>
#include <vector>

double project_simple(const BivariateElement& elem_1,
		      const BivariateElement& elem_2)
{
  if (elem_1.get_dx() != elem_2.get_dx()) {
    std::cout << "ERROR: in project_simple, dx not same for elem_1 and elem_2"
	      << std::endl;
  }
  double dx_ = elem_1.get_dx();
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

double project_omp(const BivariateElement& elem_1,
		   const BivariateElement& elem_2)
{
  if (elem_1.get_dx() != elem_2.get_dx()) {
    std::cout << "ERROR: in project_omp, dx not same for elem_1 and elem_2"
	      << std::endl;
  }
  double dx_ = elem_1.get_dx();
  
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

int main() {
  double dx = 5e-4;
  double exponent_power = 1;
  long unsigned dimension = 2;
  gsl_vector* mean = gsl_vector_alloc(dimension);
  gsl_vector* input = gsl_vector_alloc(dimension);
  gsl_vector_set_all(mean, 0.1);
  gsl_vector_set_all(input, 0.5);

  gsl_matrix* cov = gsl_matrix_alloc(dimension, dimension);
  gsl_matrix_set_all(cov, 0.5);

  for (unsigned i=0; i<dimension; ++i) {
    gsl_matrix_set(cov, i, i, 1.0);
  }

  BivariateGaussianKernelElement kernel_element =
    BivariateGaussianKernelElement(dx,
				   exponent_power,
				   mean,
				   cov);
  kernel_element.save_function_grid("test-basis.csv");

  printf("norm1 = %.32f\n", (project_simple(kernel_element,
  						  kernel_element)));
  printf("norm2 = %.32f\n", (project_omp(kernel_element,
  					    kernel_element)));
  
  // BivariateGaussianKernelBasis basis = BivariateGaussianKernelBasis(dx,
  // 								    0.9,
  // 								    0.3,
  // 								    1,
  // 								    0.5);
  
  // const BivariateLinearCombinationElement& ortho_e_1 =
  //   basis.get_orthonormal_element(0);
  // const BivariateLinearCombinationElement& ortho_e_2 =
  //   basis.get_orthonormal_element(1);
  // const BivariateLinearCombinationElement& ortho_e_3 =
  //   basis.get_orthonormal_element(2);

  // const BivariateLinearCombinationElement& ortho_e_100 =
  //   basis.get_orthonormal_element(99);
  // const BivariateLinearCombinationElement& ortho_e_99 =
  //   basis.get_orthonormal_element(98);

  // const BivariateLinearCombinationElement& last_ortho_elem =
  //   basis.get_orthonormal_element(basis.
  // 				  get_orthonormal_elements().
  // 				  size()-1);

  // std::cout << "<ortho_e_1 | ortho_e_1> = "
  // 	    << basis.project(ortho_e_1, ortho_e_1) << std::endl;
  
  // std::cout << "<ortho_e_1 | ortho_e_2> = "
  // 	    << basis.project(ortho_e_1, ortho_e_2) << std::endl;

  // std::cout << "<ortho_e_1 | ortho_e_3> = "
  // 	    << basis.project(ortho_e_1, ortho_e_3) << std::endl;

  // std::cout << "<ortho_e_2 | ortho_e_3> = "
  // 	    << basis.project(ortho_e_2, ortho_e_3) << std::endl;
  
  // std::cout << "<ortho_e_2 | ortho_e_2> = "
  // 	    << basis.project(ortho_e_2, ortho_e_2) << std::endl;

  // std::cout << "<ortho_e_3 | ortho_e_3> = "
  // 	    << basis.project(ortho_e_3, ortho_e_3) << std::endl;

  // std::cout << "<ortho_e_99 | ortho_e_100> = "
  // 	    << basis.project(ortho_e_99, ortho_e_100) << std::endl;

  // // OUTPUTTING FUNCTION GRID START
  // const BivariateElement& printed_elem = basis.get_orthonormal_element(10);
  // std::cout << "norm of printed_elem = " << printed_elem.norm() << std::endl;

  // int N = 1/dx + 1;
  // double x = 0.0;
  // double y = 0.0;
  // double min_out = 0;
  // double max_out = 0;
  
  // gsl_matrix_minmax(printed_elem.get_function_grid(), &min_out, &max_out);
  // std::cout << "min_out = " << min_out << "\n";
  // std::cout << "max_out = " << max_out << std::endl;
  // std::vector<double> minmax = std::vector<double> {std::abs(min_out),
  // 						    std::abs(max_out)};
  // double abs_max = *std::max_element(minmax.begin(),
  // 				     minmax.end());
  // std::ofstream output_file;
  // output_file.open("orthonormal-element.csv");

  // // header
  // output_file << "x, y, function.val\n";
  
  // for ( unsigned i=0; i<N; ++i) {
  //   x = dx*i;
  //   for (unsigned j=0; j<N; ++j) {
  //     y = dx*j;

  //     output_file << x << ","
  // 		  << y << ","
  // 		  << (gsl_matrix_get(printed_elem.get_function_grid(),
  // 				     i, j) - min_out) / (max_out-min_out) << "\n";
  //   }
  // }
  // output_file.close();
  // // OUTPUTTING FUNCTION GRID END

  // // const igraph_matrix_t& mass_matrix = basis.get_mass_matrix();
  // // std::cout << &mass_matrix << std::endl;
  // // std::cout << igraph_matrix_e(&mass_matrix, 0, 0) << " ";
  // // std::cout << igraph_matrix_e(&mass_matrix, 0, 1) << std::endl;
  // // std::cout << igraph_matrix_e(&mass_matrix, 1, 0) << " ";
  // // std::cout << igraph_matrix_e(&mass_matrix, 1, 1) << "\n" << std::endl;

  // // igraph_vector_t first_col;
  // // igraph_vector_init(&first_col, 1);
  // // igraph_matrix_get_col(&mass_matrix, &first_col, 1);
  // // igraph_vector_print(&first_col);

  // // igraph_vector_destroy(&first_col);

  return 0;
}
