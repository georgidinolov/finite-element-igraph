#include "2DBrownianMotionPath.hpp"
#include "BivariateSolver.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>

int main() {
  std::cout << std::fixed << std::setprecision(32);
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

  double ax = BM.get_a();
  double x_T = BM.get_x_T();
  double bx = BM.get_b();

  double ay = BM.get_c();
  double y_T = BM.get_y_T();
  double by = BM.get_d();

  // STEP 1
  double x_0_1 = 0 - BM.get_a();
  double x_T_1 = BM.get_x_T() - BM.get_a();
  double b_1 = BM.get_b() - BM.get_a();
  double a_1 = BM.get_a() - BM.get_a();

  double y_0_1 = 0 - BM.get_c();
  double y_T_1 = BM.get_y_T() - BM.get_c();
  double c_1 = BM.get_c() - BM.get_c();
  double d_1 = BM.get_d() - BM.get_c();

  // STEP 2
  double Lx_2 = b_1 - a_1;
  double x_0_2 =  x_0_1 / Lx_2;
  double  x_T_2 = x_T_1 / Lx_2;
  double  a_2 = a_1 / Lx_2;
  double b_2 = b_1 / Lx_2;
  double sigma_x_2 = sigma_x_data_gen / Lx_2;

  double Ly_2 = d_1 - c_1;
  double y_0_2 =  y_0_1 / Ly_2;
  double y_T_2 = y_T_1 / Ly_2;
  double c_2 = c_1 / Ly_2;
  double d_2 = d_1 / Ly_2;
  double sigma_y_2 = sigma_y_data_gen / Ly_2;

  double dx = 5e-3;
  BivariateGaussianKernelBasis basis = BivariateGaussianKernelBasis(dx,
  								    rho_data_gen,
  								    0.30,
  								    1,
  								    0.5);
  BivariateSolver FEM_solver = BivariateSolver(basis,
  					       sigma_x_data_gen, 
  					       sigma_y_data_gen,
  					       rho_data_gen,
  					       BM.get_a(),
					       BM.get_x_0(),
					       BM.get_b(),
					       BM.get_c(),
					       BM.get_y_0(),
					       BM.get_d(),
					       t,
  					       dx);
  double x = 0;
  double y = 0;

  unsigned N = 1.0/dx + 1;
  gsl_vector* input = gsl_vector_alloc(2);
  gsl_vector_set(input, 0, BM.get_x_T());
  gsl_vector_set(input, 1, BM.get_y_T());
  std::cout << "FEM_solver(input) = " 
  	    << FEM_solver(input) << std::endl;

  std::cout << "x_T_2 = " << x_T_2 << "\n"
  	    << "y_T_2 = " << y_T_2 << std::endl;
  std::cout << "x_0_2 = " << x_0_2 << "\n"
  	    << "y_0_2 = " << y_0_2 << std::endl;

  std::cout << "FEM_solver.get_evals[0] = "
	    << gsl_vector_get(FEM_solver.get_evals(),0)
	    << std::endl;
  std::cout << "FEM_solver.get_ic_coefs[56] = "
	    << gsl_vector_get(FEM_solver.get_ic_coefs(),55)
	      << std::endl;
  std::cout << "FEM_solver.get_solution_coefs[56] = "
	    << gsl_vector_get(FEM_solver.get_solution_coefs(),0)
	    << std::endl;
    

  gsl_matrix* left = gsl_matrix_alloc(N,N);
  gsl_matrix* right = gsl_matrix_alloc(N,N);
  for (unsigned i=0; i<basis.get_orthonormal_elements().size(); ++i)
    {
      if (i==0)
	{
	  gsl_matrix_memcpy(left, basis.get_orthonormal_element(i).
			    get_function_grid());
	  gsl_matrix_scale(left, gsl_vector_get(FEM_solver.get_solution_coefs(),
						i));
	}
      else
	{
	  gsl_matrix_memcpy(right, basis.get_orthonormal_element(i).
			    get_function_grid());
	  gsl_matrix_scale(right, gsl_vector_get(FEM_solver.get_solution_coefs(),
						 i));
	  gsl_matrix_add(left, right);
	}
    }

  basis.save_matrix(left, "bivariate-solution.csv");
  
  gsl_vector_free(input);
  gsl_matrix_free(left);
  gsl_matrix_free(right);

  return 0;
}


