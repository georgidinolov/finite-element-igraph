#include "2DBrownianMotionPath.hpp"
#include "BivariateSolver.hpp"
#include <iostream>
#include <fstream>
#include <vector>

int main() {
  unsigned order = 1e6;
  double sigma_x_data_gen = 1.0;
  double sigma_y_data_gen = 0.25;
  double rho_data_gen = 0.4;
  double x_0 = 0;
  double y_0 = 0;
  double t = 1;
  long unsigned seed = 200;

  BrownianMotion BM = BrownianMotion(seed,
				     order,
				     rho_data_gen,
				     sigma_x_data_gen,
				     sigma_y_data_gen,
				     x_0,
				     y_0,
				     t);

  std::cout << "ax = " << BM.get_a() << std::endl;
  std::cout << "x_T = " << BM.get_x_T() << std::endl;
  std::cout << "bx = " << BM.get_b() << std::endl;

  std::cout << "ay = " << BM.get_c() << std::endl;
  std::cout << "y_T = " << BM.get_y_T() << std::endl;
  std::cout << "by = " << BM.get_d() << std::endl;

  // Transformation 1: Scaling so that both boundary tuples are 1 unit
  // apart. Diffusion parameters, including t, stay the same.
  double L_x = BM.get_b() - BM.get_a();
  double L_y = BM.get_d() - BM.get_c();

  double a_1 = BM.get_a() / L_x;
  double b_1 = BM.get_b() / L_x;
  double x_T_1 = BM.get_x_T() / L_x;
  double x_0_1 = BM.get_x_0() / L_x;
  double sigma_x_scaled = sigma_x_data_gen/std::pow(L_x,2);

  double c_1 = BM.get_c() / L_y;
  double d_1 = BM.get_d() / L_y;
  double y_T_1 = BM.get_y_T() / L_y;
  double y_0_1 = BM.get_y_0() / L_y;
  double sigma_y_scaled = sigma_y_data_gen/std::pow(L_y,2);

  // Transformation 2: Shifting so that lower bounds are at
  // 0. Diffusion parameters, including t, stay the same.
  double a_2 = a_1 - a_1;
  double b_2 = b_1 - a_1;
  double x_T_2 = x_T_1 - a_1;
  double x_0_2 = x_0_1 - a_1;

  double c_2 = c_1 - c_1;
  double d_2 = d_1 - c_1;
  double y_T_2 = y_T_1 - c_1;
  double y_0_2 = y_0_1 - c_1;

  double dx = 5e-3;
  BivariateGaussianKernelBasis basis = BivariateGaussianKernelBasis(dx,
  								    rho_data_gen,
  								    0.30,
  								    1,
  								    0.5);
  BivariateSolver FEM_solver = BivariateSolver(basis,
  					       sigma_x_scaled, 
  					       sigma_y_scaled,
  					       rho_data_gen,
  					       x_0_2, y_0_2,
  					       t,
  					       dx);
  double x = 0;
  double y = 0;

  unsigned N = 1.0/dx + 1;
  gsl_vector* input = gsl_vector_alloc(2);
  gsl_vector_set(input, 0, 0.5);
  gsl_vector_set(input, 1, 0.5);
  std::cout << "FEM_solver(input) = " 
  	    << FEM_solver(input) << std::endl;
  std::cout << "x_T_2 = " << x_T_2 << "\n"
  	    << "y_T_2 = " << y_T_2 << std::endl;
  std::cout << "x_0_2 = " << x_0_2 << "\n"
  	    << "y_0_2 = " << y_0_2 << std::endl;

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
  return 0;
}


