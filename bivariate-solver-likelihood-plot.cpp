#include "2DAdvectionDiffusionSolverImages.hpp"
#include "2DBrownianMotionPath.hpp"
#include "BivariateSolver.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <string>
#include <stdio.h>
#include <vector>

int main() {
  std::ofstream output_file;


  double h = 1/32;
  double sigma_x = 1.00;
  double sigma_y = 0.30;
  double rho = 0.0;

  double ax=-0.018268/(0.007022 - -0.018268);
  double x_T=0.0/(0.007022 - -0.018268);
  double bx=0.007022/(0.007022 - -0.018268);
  //
  double ay=-0.014171/(0.008332 - -0.014171);
  double y_T=0.0/(0.008332 - -0.014171);
  double by=0.008332/(0.008332 - -0.014171);
  //
  double t = 0.08;

  double dx = 1.0/500.0;
  double dx_likelihood = 1.0/16.0;
  double rho_basis = 0.0;
  double sigma_x_basis = 0.30;
  double sigma_y_basis = 0.30;
  double power = 1.0;
  double std_dev_factor = 1.0;
  output_file.open("bivariate-solver-likelihood-plot-" +
		   std::to_string(sigma_x_basis) + "-" +
		   std::to_string(sigma_y_basis) + "-" +
		   std::to_string(rho_basis) + ".csv");

  BivariateGaussianKernelBasis basis_positive =
    BivariateGaussianKernelBasis(dx,
  				 rho_basis,
  				 sigma_x_basis,
				 sigma_y_basis,
  				 power,
  				 std_dev_factor);

  BivariateSolver solver = BivariateSolver(&basis_positive,
  					   sigma_x,
  					   sigma_y,
  					   rho,
  					   ax, 0.0, bx,
  					   ay, 0.0, by,
  					   t,
  					   dx);

  double dsigma = 0.1;
  double current_xy [2];
  current_xy[0] = x_T;
  current_xy[1] = y_T;
  gsl_vector_view gsl_current_xy = gsl_vector_view_array(current_xy, 2);

  double current_sigma_y = 0.2;
  for (double i=0; current_sigma_y <= 1.0; ++i) {
    current_sigma_y = 0.2 + i*dsigma;
    solver.set_diffusion_parameters(1.0,
				    current_sigma_y,
				    rho);
    output_file << solver.numerical_likelihood(&gsl_current_xy.vector,h) << ",";
  }
  output_file << "\n";

  for (unsigned i=0; current_sigma_y <= 1.0; ++i) {
    current_sigma_y = 0.2 + i*dsigma;

    TwoDAdvectionDiffusionSolverImages images_solver = 
      TwoDAdvectionDiffusionSolverImages(0, // mu_x,
					 0, // mu_y,
					 sigma_x,
					 current_sigma_y,
					 100, // order,
					 0, // ic_x
					 0, // ic_y
					 ax, // ax
					 bx, // bx
					 ay, // ay
					 by); // by
    output_file << images_solver.numerical_likelihood(t, current_xy[0], current_xy[1], h);
  }
  output_file << "\n";
  output_file.close();
  
  return 0;
}
