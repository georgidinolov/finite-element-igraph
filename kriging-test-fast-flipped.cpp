#include <algorithm>
#include "BivariateSolver.hpp"
#include "GaussianInterpolator.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <iostream>
#include <sstream>
#include <limits>
#include "nlopt.hpp"
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc < 5 || argc > 5) {
    printf("You must provide input\n");
    printf("The input is: \n\nnumber data points;\nrho;\ninput file for likelihoods;\nnumber_points_for_integration;\n");
    exit(0);
  }

  unsigned N = std::stoi(argv[1]);
  double rho = std::stod(argv[2]);
  std::string input_file_name = argv[3];
  int M = std::stoi(argv[4]);

  static int counter = 0;
  static gsl_rng* r_ptr_threadprivate;

#pragma omp threadprivate(counter, r_ptr_threadprivate)
  omp_set_dynamic(0);
  omp_set_num_threads(20);

  long unsigned seed_init = 10;
  gsl_rng * r_ptr_local;
  const gsl_rng_type * Type;
  gsl_rng_env_setup();
  Type = gsl_rng_default;
  r_ptr_local = gsl_rng_alloc(Type);
  gsl_rng_set(r_ptr_local, seed_init + N);

  int tid=0;
  unsigned i=0;
#pragma omp parallel default(none) private(tid, i) shared(r_ptr_local)
  {
    tid = omp_get_thread_num();

    r_ptr_threadprivate = gsl_rng_clone(r_ptr_local);
    gsl_rng_set(r_ptr_threadprivate, tid);

    printf("Thread %d: counter %d\n", tid, counter);
  }

  std::vector<likelihood_point> points_for_kriging (0);
  std::vector<likelihood_point> points_for_integration (M);

  auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(none) private(i) shared(points_for_integration, N, M, seed_init, rho)
  {
#pragma omp for
    for (i=0; i<M; ++i) {
      long unsigned seed = seed_init + N + i;

      double log_sigma_x = 1 + gsl_ran_gaussian(r_ptr_threadprivate, 1.0);
      double log_sigma_y = 1 + gsl_ran_gaussian(r_ptr_threadprivate, 1.0);

      BrownianMotion BM = BrownianMotion(seed,
					 10e6,
					 rho,
					 exp(log_sigma_x),
					 exp(log_sigma_y),
					 0.0,
					 0.0,
					 1.0);

      points_for_integration[i] = BM;
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();

  std::cout << "Generating " << M << " points for integration took "
	    << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << " seconds." << std::endl;

  std::ifstream input_file(input_file_name);

  if (input_file.is_open()) {
    for (i=0; i<N; ++i) {
      likelihood_point current_lp = likelihood_point();
      input_file >> current_lp;
      if ((i >= 0) && (current_lp.FLIPPED)) { 
	points_for_kriging.push_back(current_lp); 
      }

    }
  }

  parameters_nominal params = parameters_nominal();
  std::vector<double> L =
    std::vector<double> {0.1,
			 0.0,0.1,
			 0.0,0.0,0.1,
			 0.0,0.0,0.0,0.1,
			 0.0,0.0,0.0,0.0,0.1,
			 0.0,0.0,0.0,0.0,0.0,0.1,
			 0.0,0.0,0.0,0.0,0.0,0.0,0.1};

  params.lower_triag_mat_as_vec = L;
  params.sigma_2 = 0.001;
  params.phi = -10.01;
  params.tau_2 = 1.0;

  std::cout << "points_for_kriging.size() = " << points_for_kriging.size() << std::endl;
  std::cout << "points_for_integration.size() = " << points_for_integration.size() << std::endl;

  GaussianInterpolator GP_prior = GaussianInterpolator(points_for_integration,
  						       points_for_kriging,
  						       params);

  GP_prior.optimize_parameters();
  params = GP_prior.parameters;
  std::cout << "optimal parameters = \n" << params << std::endl;

  double integral = 0;
  for (unsigned i=0; i<points_for_integration.size(); ++i) {
    double add = GP_prior(points_for_integration[i]);
    integral = integral +
      1.0/M * add;
  }
  std::cout << "Integral = " << integral << std::endl;

  std::string output_file_name = "optimal-parameters-" + std::to_string(N) + "-" + input_file_name;
  std::ofstream output_file(output_file_name);

  if (output_file.is_open()) {
    output_file << params;
  }
  output_file.close();

  return 0;
}
