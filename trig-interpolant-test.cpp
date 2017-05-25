#include "BivariateSolver.hpp"
#include <fstream>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_fft_real.h>
#include <iostream>
#include <iomanip>
#include <vector>


#define REAL(z,i) ((z)[2*(i)]) //complex arrays stored as    
#define IMAG(z,i) ((z)[2*(i)+1])

void save_function_grid(std::string file_name, 
			const gsl_matrix * function_grid) {

  std::ofstream output_file;
  output_file.open(file_name);
  output_file << std::fixed << std::setprecision(32);

  for (unsigned i=0; i<function_grid->size1; ++i) {
    for (unsigned j=0; j<function_grid->size2; ++j) {
      if (j==function_grid->size2-1) 
	{
	  output_file << gsl_matrix_get(function_grid, i,j) 
		      << "\n";
	} else {
	output_file << gsl_matrix_get(function_grid, i,j) << ",";
      }
    }
  }
  output_file.close();
}

int main() {
  // double N = std::pow(2,3); //power of 2 for Cooley-Tukey algorithm
  // int n = (int) N;
  // printf("n = %d\n", n);
  
  // double f[2*n];
  // double dx = 1.0/N;
  // double x = 0.0;
  // std::ofstream fileo("out.csv");

  // for (int i=0; i<n; ++i){      //initialize gaussian
  //   f[2*i]=std::exp(-0.5 * (x-0.5)*(x-0.5) / 0.01);
  //   // f[2*i] = x*(1-x);
  //   f[2*i + 1]=0.0;
  //   x+=dx;
  // }
  // std::cout << std::endl;
  
  // gsl_fft_complex_radix2_forward(f, 1, n);  //Fourier transform

  // for (int i=0; i<n; ++i){
  //   fileo << i <<","<<f[2*i] << "," << f[2*i+1] << "\n";  //plot frequency distribution
  // }
  // fileo.close();


  // double N = std::pow(2,8); //power of 2 for Cooley-Tukey algorithm
  // int n = (int) N;
  // printf("n = %d\n", n);
  
  // double f[n];
  // double dx = 1.0/N;
  // double x = 0.0;
  // std::ofstream fileo("out.csv");

  // for (int i=0; i<n; ++i){      //initialize gaussian
  //   f[i]=std::exp(-0.5 * (x-0.5)*(x-0.5) / 0.01);
  //   // f[i] = x*(1-x);
  //   x+=dx;
  // }
  // std::cout << std::endl;
  
  // gsl_fft_real_radix2_transform(f, 1, n);  //Fourier transform

  // for (int i=0; i<n; ++i){
  //   fileo << i <<","<<f[i] << "," << f[i] << "\n";  //plot frequency distribution
  // }
  // fileo.close();

  // double N = std::pow(2,3); //power of 2 for Cooley-Tukey algorithm
  // int n = (int) N;
  // printf("n = %d\n", n);

  // gsl_matrix * test_mat = gsl_matrix_alloc(1,n);
  // double dx = 1.0/N;
  // double x = 0.0;
  // std::ofstream fileo("out.csv");

  // for (int i=0; i<n; ++i){      //initialize gaussian
  //   // gsl_matrix_set(test_mat, 0, i, std::exp(-0.5 * (x-0.5)*(x-0.5) / 0.01));
  //   gsl_matrix_set(test_mat, 0, i, std::sin(2*M_PI*x));
  //   x+=dx;
  // }
  // std::cout << std::endl;
  // gsl_vector_view row = gsl_matrix_row(test_mat, 0);  

  // gsl_fft_real_radix2_transform(row.vector.data, row.vector.stride, row.vector.size);  //Fourier transform

  // for (int i=0; i<n; ++i){
  //   fileo << i <<","<< row.vector.data[i] << "," << row.vector.data[i] << "\n";  //plot frequency distribution
  // }
  // fileo.close();


  // double dx = 1.0/16;
  // double N = std::round(1.0/dx);
  // int n = (int) N;
  // printf("n = %d\n", n);

  // gsl_matrix * test_mat = gsl_matrix_alloc(1,n+1);
  // double x = 0.0;
  // std::ofstream fileo("out.csv");

  // for (int i=0; i<n+1; ++i){      //initialize gaussian
  //   //gsl_matrix_set(test_mat, 0, i, std::exp(-0.5 * (x-0.5)*(x-0.5) / 0.01));
  //   // gsl_matrix_set(test_mat, 0, i, std::sin(M_PI*x));
  //   gsl_matrix_set(test_mat, 0, i, x*(1-x));
  //   x+=dx;
  // }
  // std::cout << std::endl;
  // gsl_vector_view row = gsl_matrix_row(test_mat, 0);  
  // double odd_extension [2*(2*n+1)];

  // for (int i=0; i<n; ++i) {
  //   odd_extension[2*i] = row.vector.data[i];
  //   odd_extension[2*i + 1] = 0.0;

  //   odd_extension[2*(-i + 2*n)] = -row.vector.data[i];
  //   odd_extension[2*(-i + 2*n) + 1] = 0.0;
  // }
  // odd_extension[2*n] = row.vector.data[n];
  // odd_extension[2*n + 1] = 0.0;

  // for (int i=0; i<2*n+1; ++i) {
  //   std::cout << odd_extension[2*i] << ",";
  // }
  // std::cout << std::endl;

  // gsl_fft_complex_radix2_forward(odd_extension, 1, 2*n);  //Fourier transform

  // for (int i=0; i<2*n; ++i){
  //   fileo << i <<","<< odd_extension[2*i] << "," << odd_extension[2*i + 1] << "\n";  //plot frequency distribution
  // }
  // fileo.close();

  double dx = 1.0/64.0;
  unsigned dxinv = std::round(1.0/dx);
  unsigned n = dxinv;

  BivariateGaussianKernelBasis basis = BivariateGaussianKernelBasis(dx,
								    0.6,
								    0.3,
								    1,
								    0.5);
  std::cout << std::fixed << std::setprecision(32);
  double x_T = 0.0;
  double y_T = 0.0;
  double x_0 = 0.0;
  double y_0 = 0.0;
  double sigma_x = 0.88008638461644062012112499360228;
  double sigma_y = 0.94621168768833074924629045199254;
  double rho = 0.60;
  // BivariateSolverClassical classical_solver =
  //   BivariateSolverClassical(sigma_x, sigma_y, rho,
  // 			     x_0, y_0);
  // classical_solver.set_function_grid(dx);
  // // classical_solver.save_FFT_grid("ic.csv");

  BivariateSolver solver = BivariateSolver(&basis,
					   sigma_x,
					   sigma_y,
					   rho, 
					   -1.0, x_0, 1.0,
					   -1.0, y_0, 1.0,
					   1,
					   dx);
  gsl_vector * input = gsl_vector_alloc(2);
  gsl_vector_set(input, 0, x_T);
  gsl_vector_set(input, 0, y_T);
  printf("solver(input) = %.16f\n", solver(input));

  // printf("dxdx for kernel elem = %.16f\n", basis.project_deriv(kernel_element,
  // 							       0,
  // 							       kernel_element,
  // 							       0));

  // double f [(n+1)*(n+1)];
  // gsl_matrix_view fun_grid_odd_extension = gsl_matrix_view_array(f, n+1, n+1);

  // // for (int i=0; i<2*n+1; ++i) {
  // //   double x = i*dx;
  // //   for (int j=0; j<2*n+1; ++j) {
  // //     double y = j*dx;
  // //     double fval = std::sin(2*M_PI*0.5*x)*std::sin(2*M_PI*0.5*y);
  // //     gsl_matrix_set(&fun_grid_odd_extension.matrix, i,j, fval);
  // //   }
  // // }
    
  
  // for (unsigned i=0; i<n+1; ++i) {
  //   for (unsigned j=0; j<n+1; ++j) {
  //     gsl_matrix_set(&fun_grid_odd_extension.matrix, i, j,
  // 		     gsl_matrix_get(kernel_element.get_function_grid(), i,j));
  //   }
  // }
  // //  save_function_grid("odd-extension-1.csv", &fun_grid_odd_extension.matrix);
  // //  save_function_grid("odd-extension.csv", &fun_grid_odd_extension.matrix);

  // gsl_matrix * fft_mat = gsl_matrix_alloc(2 * n, n);
  
  // // FFT FIRST PASS START
  // for (unsigned i=0; i<n; ++i) {
  //   double fft_row [2 * n];
  //   for (unsigned j=0; j<n; ++j) {
  //     fft_row[2*j] = gsl_matrix_get(&fun_grid_odd_extension.matrix, i, j);
  //     fft_row[2*j + 1] = 0.0;
  //   }

  //   gsl_fft_complex_radix2_forward(fft_row, 1, n);

  //   for (unsigned j=0; j<n; ++j) {
  //     gsl_matrix_set(fft_mat, 2*i, j, fft_row[2*j]);
  //     gsl_matrix_set(fft_mat, 2*i + 1, j, fft_row[2*j + 1]);
  //   }

  // }
  // // FFT FIRST PASS END
  // //  save_function_grid("odd-extension-fft-first-pass.csv", fft_mat);

  // // FFT SECOND PASS START
  // for (unsigned j=0; j<n; ++j) {
  //   double fft_col [2 * n];
  //   gsl_vector_view fft_col_view = gsl_vector_view_array(fft_col, 2*n);
  //   gsl_vector_view fft_col_rhs = gsl_matrix_column(fft_mat,j);
  //   gsl_vector_memcpy(&fft_col_view.vector, &fft_col_rhs.vector);
  //   // for (unsigned i=0; i<2*n; ++i) {
  //   //   fft_col[i] = gsl_matrix_get(fft_mat, i, j);
  //   // }

  //   gsl_fft_complex_radix2_forward(fft_col, 1, n);
    
  //   gsl_vector_memcpy(&fft_col_rhs.vector, &fft_col_view.vector);
  //   // for (unsigned i=0; i<2*n; ++i) {
  //   //   gsl_matrix_set(fft_mat, i, j, fft_col[i]);
  //   // }
  // }
  // // FFT SECOND PASS END
  
  // double approximate_norm = 0;
  // for (unsigned i=0; i<n; ++i) {
  //   for (unsigned j=0; j<n; ++j) {
  //     double A = 
  // 	std::sqrt(
  // 	gsl_matrix_get(fft_mat, 2*i, j)*
  // 	gsl_matrix_get(fft_mat, 2*i, j) +
  // 	gsl_matrix_get(fft_mat, 2*i+1, j)*
  // 	gsl_matrix_get(fft_mat, 2*i+1, j));

  //     approximate_norm += A*A;
  //   }
  // }
  // std::cout << "approximate_norm = " << std::sqrt(approximate_norm / std::pow(n,4)) << std::endl;

  // save_function_grid("odd-extension-fft.csv", fft_mat);
  

  // // odd_extension[2*n] = row.vector.data[n];
  // // odd_extension[2*n + 1] = 0.0;





  // // gsl_vector_view row = gsl_matrix_row(fun_grid, 3);
  // // double f [row.vector.size];
  // // double x = 0;
  // // for (unsigned i=0; i<row.vector.size; ++i) {
  // //   //    f[i] = gsl_vector_get(&row.vector, i);
  // //   f[i] = std::sin(M_PI*x);
  // //   x += dx;
  // //   std::cout << f[i] << ",";
  // // }
  // // std::cout << std::endl;

  // // gsl_fft_real_radix2_transform(f, 1, dxinv);

  // // std::ofstream fileo("out.csv");
  // // for (unsigned i=0; i<row.vector.size; ++i) {
  // //   fileo << i <<","<< f[i] << "," << f[i] << "\n";  //plot frequency distribution
  // //   std::cout << f[i] << ",";
  // // }
  // // std::cout << std::endl;
  // // fileo.close();

  return 0;
}
