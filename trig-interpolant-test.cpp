#include "BivariateSolver.hpp"
#include <chrono>
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


  double dx = 1.0/256.0;
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
  // classical_solver.save_FFT_grid("ic.csv");
  // BivariateLinearCombinationElementFourier kernel_element = 
  //   basis.get_orthonormal_element(1);
  // kernel_element.save_FFT_grid("trig-test.csv");

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
  auto t1 = std::chrono::high_resolution_clock::now();    
  printf("solver.numerical_likelihood(input, dx) = %.16f\n",
	 solver.numerical_likelihood(input, dx));
  auto t2 = std::chrono::high_resolution_clock::now();    
  std::cout << "likelihood duration = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " milliseconds\n";

  return 0;
}
