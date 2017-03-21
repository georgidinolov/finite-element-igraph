#include "BasisElementTypes.hpp"
#include <iostream>
#include <fstream>
#include <vector>

int main() {
  BivariateSolverClassical classical_solver =
    BivariateSolverClassical(1.0, 0.1, 0.9,
			     0.5, 0.5);
  double x = 0;
  double y = 0;
  
  double dx = 0.002;
  double dy = 0.002;

  unsigned N = 1.0/dx;
  unsigned M = 1.0/dx;

  gsl_vector* input = gsl_vector_alloc(2);

  std::ofstream output_file;
  output_file.open("/home/georgi/research/PDE-solvers/classical-solution.csv");
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
		  << classical_solver(input) << "\n";
    }
  }
  output_file.close();
  gsl_vector_free(input);
  return 0;
}


