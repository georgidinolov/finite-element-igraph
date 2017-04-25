#include "2DAdvectionDiffusionSolverImages.hpp"
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
  double rho_data_gen = 0.5;
  double x_0 = 0.0;
  double y_0 = 0.0;
  double t = 1;
  long unsigned seed = 2000;
  double dx = 4e-3;
  gsl_vector* input = gsl_vector_alloc(2);

  // LIKELIHOOD LOOP
  std::ofstream output_file;
  output_file.open("likelihood-rho.txt");
  output_file << std::fixed << std::setprecision(32);
  output_file << "log.likelihood, rho\n";
  
  long unsigned seed_init = 2000;
  unsigned N = 50;
  // GENERATE DATA
  std::vector<BrownianMotion> BMs (0);
  for (unsigned i=0; i<N; ++i) {
    seed = seed_init + i;
    BrownianMotion BM = BrownianMotion(seed,
				       order,
				       rho_data_gen,
				       sigma_x_data_gen,
				       sigma_y_data_gen,
				       x_0,
				       y_0,
				       t);
    BMs.push_back(BM);
  }

  std::cout << "throwing bases on heap\n";
  std::vector<BivariateGaussianKernelBasis*> bases (3);
  bases[0] = new BivariateGaussianKernelBasis(dx,
					      0.0,
					      0.3,
					      1,
					      0.5);
  std::cout << "done with bases[0]\n";

  bases[1] = new BivariateGaussianKernelBasis(dx,
					      0.5,
					      0.3,
					      1,
					      0.5);
  std::cout << "done with bases[1]\n";

  bases[2] = new BivariateGaussianKernelBasis(dx,
					      0.9,
					      0.3,
					      1,
					      0.5);
  std::cout << "done with bases[2]\n";
  
  unsigned R = 11;
  double dr = 0.1;
  double rho_init = -0.2;
  for (unsigned r=0; r<R; ++r) {
    double rho = rho_init + dr*r;
    BivariateGaussianKernelBasis * basis;
    if (abs(rho-0.0) <= abs(rho - 0.5) && abs(rho-0) <= abs(rho-0.9)) {
      basis = bases[0];
    } else if (abs(rho-0.5) <= abs(rho-0.0) && abs(rho-0) <= abs(rho-0.9)) {
      basis = bases[1];
    } else if (abs(rho-0.9) <= abs(rho-0.0) && abs(rho-0) <= abs(rho-0.5)) {
      basis = bases[2];
    }

    double log_likelihood = 0;
    for (unsigned i=0; i<N; ++i) {
      gsl_vector_set(input, 0, BMs[i].get_x_T());
      gsl_vector_set(input, 1, BMs[i].get_y_T());
      
      BivariateSolver FEM_solver_2 = BivariateSolver(basis,
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
      double FEM_likelihood = 0.0;
      FEM_likelihood = FEM_solver_2.numerical_likelihood_first_order(input, dx);
      
      std::cout << "i=" << i << "; ";
      std::cout << "FEM.numerical_likelihood(input,dx) = "
  		<< FEM_likelihood << "; "
		<< "r = " << r;
      std::cout << std::endl;

      if (FEM_likelihood > 1e-16) {
  	log_likelihood = log_likelihood + log(FEM_likelihood);
      }
    }
    output_file << log_likelihood << ", " << rho << "\n";
  }

  gsl_vector_free(input);
  output_file.close();
  return 0;
}


