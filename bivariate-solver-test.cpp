#include "2DBrownianMotionPath.hpp"
#include "BivariateSolver.hpp"
#include <iostream>
#include <fstream>
#include <vector>

int main() {
  unsigned order = 1e6;
  double sigma_x_data_gen = 1.0;
  double sigma_y_data_gen = 0.1;
  double rho_data_gen = 0.8;
  double x_0 = 0;
  double y_0 = 0;
  double t = 1;
  long unsigned seed = 100;

  BrownianMotion BM = BrownianMotion(seed,
				     order,
				     rho_data_gen,
				     sigma_x_data_gen,
				     sigma_y_data_gen,
				     x_0,
				     y_0,
				     t);
  // Transformation 1: Scaling so that both boundary tuples are 1 unit
  // apart. Diffusion parameters, including t, stay the same.
  double L_x = BM.get_b() - BM.get_a();
  double L_y = BM.get_d() - BM.get_c();

  double a_1 = BM.get_a() / L_x;
  double b_1 = BM.get_b() / L_x;
  double x_T_1 = BM.get_x_T() / L_x;

  double c_1 = BM.get_c() / L_y;
  double d_1 = BM.get_d() / L_y;
  double y_T_1 = BM.get_y_T() / L_y;

  // Transformation 2: Shifting so that lower bounds are at
  // 0. Diffusion parameters, including t, stay the same.
  double a_2 = a_1 - a_1;
  double b_2 = b_1 - a_1;
  double x_T_2 = x_T_1 - a_1;

  double c_2 = c_1 - c_1;
  double d_2 = d_1 - c_1;
  double y_T_2 = y_T_1 - c_1;


  double dx = 2e-3;
  BivariateGaussianKernelBasis basis = BivariateGaussianKernelBasis(dx,
								    rho_data_gen,
								    0.20,
								    1,
								    0.5);
  BivariateSolver FEM_solver = BivariateSolver(basis,
  					       sigma_x_data_gen, 
					       sigma_y_data_gen,
					       rho_data_gen,
  					       x_T_2, y_T_2,
  					       t,
  					       dx);
  double x = 0;
  double y = 0;

  unsigned N = 1.0/dx;
  gsl_vector* input = gsl_vector_alloc(2);
  gsl_vector_set(input, 0, 0.5);
  gsl_vector_set(input, 1, 0.5);
  std::cout << "FEM_solver(input) = " 
	    << FEM_solver(input) << std::endl;

  std::ofstream output_file;
  output_file.open("bivariate-solution.csv");
  // header
  output_file << "x, y, solution\n";
  
  for ( unsigned i=0; i<N; ++i) {
    x = dx*i;
    gsl_vector_set(input, 0, x);

    for (unsigned j=0; j<N; ++j) {
      y = dx*j;
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


