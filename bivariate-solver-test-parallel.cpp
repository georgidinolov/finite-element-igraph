#include "2DBrownianMotionPath.hpp"
#include "BivariateSolver.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <string>
#include <vector>

int main() {
  std::cout << std::fixed << std::setprecision(32);
  unsigned order = 1e6;
  double sigma_x_data_gen = 1.0;
  double sigma_y_data_gen = 0.25;
  double rho_data_gen = 0.1;
  double x_0 = 0.0;
  double y_0 = 0.0;
  double t = 1;
  long unsigned seed = 2000;
  gsl_vector * input = gsl_vector_alloc(2);

  BrownianMotion BM = BrownianMotion(seed,
				     order,
				     rho_data_gen,
				     sigma_x_data_gen,
				     sigma_y_data_gen,
				     x_0,
				     y_0,
				     t);

  double dx = 5e-3;
  BivariateGaussianKernelBasis* basis = new BivariateGaussianKernelBasis();

  // LIKELIHOOD LOOP
  std::ofstream output_file;
  output_file.open("likelihood-rho.txt");
  output_file << std::fixed << std::setprecision(32);
  output_file << "log.likelihood\n";
  
  long unsigned seed_init = 2000;
  unsigned N = 1000;
  // GENERATE DATA
  std::vector<BrownianMotion> BMs (0);
  for (unsigned i=0; i<N; ++i) {
    seed = seed_init + i;
    BM = BrownianMotion(seed,
			order,
			rho_data_gen,
			sigma_x_data_gen,
			sigma_y_data_gen,
			x_0,
			y_0,
			t);
    BMs.push_back(BM);
  }
  std::vector<double> neg_log_likelihoods (N, 0.0);
  
  unsigned R = 3;
  double dr = 0.1;
  double rho_init = -0.3;
  for (unsigned r=0; r<R; ++r) {
    double rho = rho_init + dr*r;
    delete basis;
    basis = new BivariateGaussianKernelBasis(dx,
					     rho,
					     0.30,
					     1,
					     0.5);
    
    BivariateGaussianKernelBasis* null_basis = NULL;

    unsigned i = 0;
    double FEM_likelihood = 0;

    std::cout << "allocating solvers" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<BivariateSolver*> FEM_solvers (N);

//     omp_set_dynamic(1);
// #pragma omp parallel private(i) shared(FEM_solvers, basis, sigma_x_data_gen, sigma_y_data_gen, rho, BMs, t, dx, N)
//  {
// #pragma omp for
    for (i=0; i<N; ++i) {
      delete FEM_solvers[i];
      FEM_solvers[i] = new BivariateSolver(basis,
					   sigma_x_data_gen, 
					   sigma_y_data_gen,
					   rho,
					   BMs[i].get_a(),
					   BMs[i].get_x_0(),
					   BMs[i].get_b(),
					   BMs[i].get_c(),
					   BMs[i].get_y_0(),
					   BMs[i].get_d(),
					   t,
					   dx);
    }
 // }
 
    for (i=0; i<N; ++i) {
      delete FEM_solvers[i];
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "T to alloc " << N << " solvers = "
	      << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
	      << " millisecnds." << std::endl;

    
//     BivariateSolver FEM_solver_2 = BivariateSolver(*basis,
// 						   sigma_x_data_gen, 
// 						   sigma_y_data_gen,
// 						   rho,
// 						   BMs[i].get_a(),
// 						   BMs[i].get_x_0(),
// 						   BMs[i].get_b(),
// 						   BMs[i].get_c(),
// 						   BMs[i].get_y_0(),
// 						   BMs[i].get_d(),
// 						   t,
// 						   dx);
//     for (i=0; i<N; ++i) {
//       gsl_vector_set(input, 0, BMs[i].get_x_T());
//       gsl_vector_set(input, 1, BMs[i].get_y_T());
      
//       BivariateSolver FEM_solver_2 = BivariateSolver(*basis,
// 						     sigma_x_data_gen, 
// 						     sigma_y_data_gen,
// 						     rho,
// 						     BMs[i].get_a(),
// 						     BMs[i].get_x_0(),
// 						     BMs[i].get_b(),
// 						     BMs[i].get_c(),
// 						     BMs[i].get_y_0(),
// 						     BMs[i].get_d(),
// 						     t,
// 						     dx);
      
//       FEM_likelihood = FEM_solver_2.
// 	numerical_likelihood_first_order(input, dx);
      
//       std::cout << "i=" << i << "; ";
//       std::cout << "FEM.numerical_likelihood(input,dx) = "
//   		<< FEM_likelihood;
//       std::cout << std::endl;

//       if (FEM_likelihood > 1e-16) {
//   	neg_log_likelihoods[i] = -log(FEM_likelihood);
//       }
//     }

  
  }
  
  gsl_vector_free(input);
  output_file.close();
  return 0;
}


