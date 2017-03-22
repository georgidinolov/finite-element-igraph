#include "BivariateSolver.hpp"
#include <iostream>
#include <fstream>
#include <vector>

int main() {
  double dx = 1e-2;
  BivariateGaussianKernelBasis basis = BivariateGaussianKernelBasis(dx,
								    0.0,
								    0.5,
								    1,
								    0.5);
  BivariateSolver FEM_solver =
    BivariateSolver(basis,
		    0.1, 0.1, 0.0,
		    0.5, 0.5,
		    1.7);
  double x = 0;
  double y = 0;
  dx = 0.01;
  double dy = 0.01;

  unsigned N = 1.0/dx;
  unsigned M = 1.0/dx;

  gsl_vector* input = gsl_vector_alloc(2);

  std::ofstream output_file;
  output_file.open("/home/georgi/research/PDE-solvers/bivariate-solution.csv");
  // header
  output_file << "x, y, solution\n";
  
  for ( unsigned i=0; i<N; ++i) {
    x = dx*i;
    gsl_vector_set(input, 0, x);

    for (unsigned j=0; j<M; ++j) {
      y = dy*j;
      gsl_vector_set(input, 1, y);

      output_file << x << ","
		  << y << ","
		  << FEM_solver(input) << "\n";
    }
  }
  output_file.close();
  gsl_vector_free(input);

  
  // double x = 0;
  // double y = 0;
  
  // double dx = 0.002;
  // double dy = 0.002;

  // unsigned N = 1.0/dx;
  // unsigned M = 1.0/dx;

  // gsl_vector* input = gsl_vector_alloc(2);

  // std::ofstream output_file;
  // output_file.open("/home/georgi/research/PDE-solvers/classical-solution.csv");
  // // header
  // output_file << "x, y, solution\n";
  
  // for ( unsigned i=0; i<N; ++i) {
  //   x = dx*i;
  //   gsl_vector_set(input, 0, x);

  //   for (unsigned j=0; j<M; ++j) {
  //     y = dy*j;
  //     gsl_vector_set(input, 1, y);

  //     output_file << x << ","
  // 		  << y << ","
  // 		  << classical_solver(input) << "\n";
  //   }
  // }
  // output_file.close();
  // gsl_vector_free(input);
  return 0;
}


