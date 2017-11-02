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
  output_file.open("bivariate-solver-contour-plot.csv");

  double sigma_x=0.008695;
  double sigma_y=0.009606;
  double rho=0;
  double ax=-0.005448;
  double x_T=0.005183;
  double bx=0.015722;
  double ay=-0.001143;
  double y_T=0.004881;
  double by=0.014078;

  double t = 1.0;

  // NORMALIZING
  double Lx = bx - ax;
  double Ly = by - ay;

  double tau_x = sigma_x/Lx;
  double tau_y = sigma_y/Ly;

  ax = ax/Lx;
  x_T = x_T/Lx;
  bx = bx/Lx;
  
  ay = ay/Ly;
  y_T = y_T/Ly;
  by = by/Ly;

  double t_tilde = t*tau_x*tau_x;
  double beta_y = tau_y/tau_x;
  double beta_x = 1.0;
  
  double dx = 1.0/500.0;
  double dx_likelihood = 1.0/256.0;
  double rho_basis = 0.0;
  double sigma_x_basis = 0.30;
  double sigma_y_basis = 0.05;
  double power = 1.0;
  double std_dev_factor = 1.0;

  BivariateGaussianKernelBasis basis_positive =
    BivariateGaussianKernelBasis(dx,
  				 rho_basis,
  				 sigma_x_basis,
				 sigma_y_basis,
  				 power,
  				 std_dev_factor);

  // manual derivative at t_2_ = 1.0
  
  BivariateSolver solver = BivariateSolver(&basis_positive,
  					   beta_x,
  					   beta_y,
  					   rho,
  					   ax, 0.0, bx,
  					   ay, 0.0, by,
  					   t_tilde*1.0,
  					   dx);

  double current_xy [2] {x_T, y_T};


  gsl_vector_view gsl_current_xy = gsl_vector_view_array(current_xy, 2);

  double like = solver.numerical_likelihood_extended(&gsl_current_xy.vector, dx_likelihood*2.0);
  double likelihood = like/( std::pow((bx-ax),3) * std::pow((by-ay),3) );
  printf("likelihood = %f,\n", like/( std::pow((bx-ax),3) * std::pow((by-ay),3) ) );
  printf("log likelihood = %f,\n", log(likelihood));
  std::cout << "sign(like) = " << std::signbit(like) << std::endl;
  std::cout << "sign(-1) = " << std::signbit(-1) << std::endl;
  std::cout << "like > 0 = " << ( like > 0.0 ) << std::endl;

    std::cout << std::endl;
  like = solver.numerical_likelihood_extended(&gsl_current_xy.vector, dx_likelihood);
  likelihood = like/( std::pow((bx-ax),3) * std::pow((by-ay),3) );
  printf("likelihood = %f,\n", like/( std::pow((bx-ax),3) * std::pow((by-ay),3) ) );
  printf("log likelihood = %f,\n", log(likelihood));
  std::cout << "sign(like) = " << std::signbit(like) << std::endl;
  std::cout << "sign(-1) = " << std::signbit(-1) << std::endl;
  std::cout << "like > 0 = " << ( like > 0.0 ) << std::endl;

  std::cout << std::endl;
  like = solver.numerical_likelihood_extended(&gsl_current_xy.vector, dx_likelihood/2.0);
  likelihood = like/( std::pow((bx-ax),3) * std::pow((by-ay),3) );
  printf("likelihood = %f,\n", like/( std::pow((bx-ax),3) * std::pow((by-ay),3) ) );
  printf("log likelihood = %f,\n", log(likelihood));
  std::cout << "sign(like) = " << std::signbit(like) << std::endl;
  std::cout << "sign(-1) = " << std::signbit(-1) << std::endl;
  std::cout << "like > 0 = " << ( like > 0.0 ) << std::endl;

  std::cout << std::endl;
  like = solver.numerical_likelihood_extended(&gsl_current_xy.vector, dx_likelihood/4.0);
  likelihood = like/( std::pow((bx-ax),3) * std::pow((by-ay),3) );
  printf("likelihood = %f,\n", like/( std::pow((bx-ax),3) * std::pow((by-ay),3) ) );
  printf("log likelihood = %f,\n", log(likelihood));
  std::cout << "sign(like) = " << std::signbit(like) << std::endl;
  std::cout << "sign(-1) = " << std::signbit(-1) << std::endl;
  std::cout << "like > 0 = " << ( like > 0.0 ) << std::endl;

    std::cout << std::endl;
  like = solver.numerical_likelihood_extended(&gsl_current_xy.vector, dx_likelihood/8.0);
  likelihood = like/( std::pow((bx-ax),3) * std::pow((by-ay),3) );
  printf("likelihood = %f,\n", like/( std::pow((bx-ax),3) * std::pow((by-ay),3) ) );
  printf("log likelihood = %f,\n", log(likelihood));
  std::cout << "sign(like) = " << std::signbit(like) << std::endl;
  std::cout << "sign(-1) = " << std::signbit(-1) << std::endl;
  std::cout << "like > 0 = " << ( like > 0.0 ) << std::endl;
  
  // double dx_interation = 1.0/400 * (bx - ax);
  // double dy_interation = 1.0/400 * (by - ay);

  // gsl_vector_view gsl_current_xy = gsl_vector_view_array(current_xy, 2);
  // printf("current_xy = (%f, %f)\n", current_xy[0], current_xy[1]);
  // printf("current_xy = (%f, %f)\n",
  // 	 gsl_vector_get(&gsl_current_xy.vector, 0),
  // 	 gsl_vector_get(&gsl_current_xy.vector, 1));
  // printf("solver(current_xy) = %f\n", solver(&gsl_current_xy.vector));

  // // for (unsigned i=1; i<2; ++i) {
  // //   current_xy[0] = ax + i*dx_interation;
    
  // //   for (unsigned j=1; j<2; ++j) {
  // //     current_xy[1] = ay + j*dy_interation;
  // double like = solver.numerical_likelihood_extended(&gsl_current_xy.vector, dx_likelihood);
  //     output_file << like << ",";
  //     printf("likelihood = %f\n", like);
  // //   }

  // //   output_file << "\n";
  // // }
  // output_file.close();

  // output_file.open("classical-solver-contour-plot.csv");

  // TwoDAdvectionDiffusionSolverImages* images_solver = 
  //   new TwoDAdvectionDiffusionSolverImages(0, // mu_x,
  // 					   0, // mu_y,
  // 					   sigma_x,
  // 					   sigma_y,
  // 					   100, // order,
  // 					   0, // ic x
  // 					   0, // ic y
  // 					   ax, // ax
  // 					   bx, // bx
  // 					   ay, // ay
  // 					   by); // by

  // for (unsigned i=1; i<399; ++i) {
  //   current_xy[0] = ax + i*dx_interation;
    
  //   for (unsigned j=1; j<399; ++j) {
  //     current_xy[1] = ay + j*dy_interation;
  //     output_file << images_solver->solve(t, current_xy[0], current_xy[1]) << ",";
  //   }
  //   output_file << "\n";
  // }
  
  // output_file.close();

  
  // // double input [2];
  // // input[0] = x_T; input[1] = y_T;
  
  // // double h = 1/32;

  // delete images_solver;
  return 0;
}
