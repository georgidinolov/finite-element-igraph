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
  omp_set_num_threads(3);

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

  std::vector<likelihood_point> points_for_kriging (N);
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
      input_file >> points_for_kriging[i];
      points_for_kriging[i].likelihood = log(points_for_kriging[i].likelihood);
    }
  }

  parameters_nominal params = parameters_nominal();
  std::vector<double> L =
    std::vector<double> {10.0,
			 0.0, 10.0,
			 0.0, 0.0, 10.0,
			 0.0, 0.0, 0.0, 10.0,
			 0.0, 0.0, 0.0, 0.0, 10.0,
			 0.0, 0.0, 0.0, 0.0, 0.0, 10.0,
			 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0};

  params.lower_triag_mat_as_vec = L;
  params.sigma_2 = 1;
  params.phi = -10;
  params.tau_2 = 10;
  
  GaussianInterpolator GP_prior = GaussianInterpolator(points_for_integration,
						       points_for_kriging,
						       params);

  nlopt::opt opt(nlopt::LN_NELDERMEAD, 32);
  std::vector<double> lb = {0.00001, -HUGE_VAL, 5, 0.0001,
  			    0.00001, // diag element 1 min
  			    -HUGE_VAL, 0.00001,
  			    -HUGE_VAL, -HUGE_VAL, 0.00001,
  			    -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, 0.00001,
  			    -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, 0.00001,
  			    -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, 0.00001,
  			    -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, 0.00001};

  std::vector<double> ub = {HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL,
  			    HUGE_VAL, // diag element 1 max
  			    HUGE_VAL, HUGE_VAL,
  			    HUGE_VAL, HUGE_VAL, HUGE_VAL,
  			    HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL,
  			    HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL,
  			    HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL,
  			    HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL};

  std::cout << std::endl;
  std::cout << "starting parameters = " << std::endl;
  std::cout << GP_prior.parameters << std::endl;
  

  std::vector<double> optimal_params = GP_prior.parameters.as_vector();
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);
  opt.set_ftol_rel(0.0001);
  double maxf;
  //  opt.set_maxeval(1000);
  opt.set_max_objective(optimization_wrapper,
  			&GP_prior);

  opt.optimize(optimal_params, maxf);

  params = parameters_nominal(optimal_params);
  GP_prior.set_parameters(params);
  std::cout << "optimal parameters = \n" << params << std::endl;
  
  double integral = 0;
  for (unsigned i=0; i<points_for_integration.size(); ++i) {
    double add = GP_prior(points_for_integration[i]);
    integral = integral +
      1.0/M * add;
  }
  std::cout << "Integral = " << integral << std::endl;

  double dt = 0.1;
  unsigned m = 150;
  likelihood_point lp = points_for_kriging[7];
  std::vector<double> log_likelihoods (0);
  std::vector<double> log_likelihoods_var = std::vector<double> (m, 0.0);
  std::vector<double> ts (0);
  for (unsigned i=0; i<m; ++i) {
    lp.t_tilde = (i+1)*dt;
    ts.push_back(lp.t_tilde);
    log_likelihoods.push_back(GP_prior(lp));
    log_likelihoods_var[i] = GP_prior.prediction_variance(lp);
  }

  std::cout << "ts = c(";
  for (unsigned i=0; i<ts.size(); ++i) {
    std::cout << ts[i] << ",";
  }
  std::cout << ");" << std::endl;

  std::cout << "lls.t = c(";
  for (unsigned i=0; i<log_likelihoods.size(); ++i) {
    std::cout << log_likelihoods[i] << ",";
  }
  std::cout << ");" << std::endl;

  std::cout << "lls.t.var = c(";
  for (unsigned i=0; i<log_likelihoods.size(); ++i) {
    std::cout << log_likelihoods_var[i] << ",";
  }
  std::cout << ");" << std::endl;

  double dsigma = 0.1;
  m = 150;
  lp = points_for_kriging[7];
  log_likelihoods = std::vector<double> (m, 0.0);
  log_likelihoods_var = std::vector<double> (m, 0.0);
  ts = std::vector<double> (m, 0.0);
  for (unsigned i=0; i<m; ++i) {
    lp.sigma_y_tilde = (i+1)*dsigma;
    ts[i] = lp.sigma_y_tilde;
    log_likelihoods[i] = GP_prior(lp);
    log_likelihoods_var[i] = GP_prior.prediction_variance(lp);
  }

  std::cout << "sigma_y_tiles = c(";
  for (unsigned i=0; i<ts.size(); ++i) {
    std::cout << ts[i] << ",";
  }
  std::cout << ");" << std::endl;

  std::cout << "lls.sigma = c(";
  for (unsigned i=0; i<log_likelihoods.size(); ++i) {
    std::cout << log_likelihoods[i] << ",";
  }
  std::cout << ");" << std::endl;

  std::cout << "lls.sigma.var = c(";
  for (unsigned i=0; i<log_likelihoods_var.size(); ++i) {
    std::cout << log_likelihoods_var[i] << ",";
  }
  std::cout << ");" << std::endl;

  // -------------------------------
  double drho = 0.01;
  m = 200;
  lp = points_for_kriging[7];
  log_likelihoods = std::vector<double> (m, 0.0);
  ts = std::vector<double> (m, 0.0);
  for (unsigned i=0; i<m; ++i) {
    lp.rho = -1.0 + (i+1)*drho;
    ts[i] = lp.rho;
    log_likelihoods[i] = GP_prior(lp);
  }

  std::cout << "rhos = c(";
  for (unsigned i=0; i<ts.size(); ++i) {
    std::cout << ts[i] << ",";
  }
  std::cout << ");" << std::endl;

  std::cout << "lls.rho = c(";
  for (unsigned i=0; i<log_likelihoods.size(); ++i) {
    std::cout << log_likelihoods[i] << ",";
  }
  std::cout << ");" << std::endl;
  
  return 0;
}
