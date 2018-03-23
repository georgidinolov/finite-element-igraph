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
  if (argc < 4 || argc > 4) {
    printf("You must provide input\n");
    printf("The input is: \n\nnumber of points of data;\nnumber of samples per datum;\ninput file for likelihoods;\n");
    exit(0);
  }

  unsigned number_points_of_data = std::stoi(argv[1]);
  unsigned number_points_per_datum = std::stoi(argv[2]);
  std::string input_file_name = argv[3];

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
  gsl_rng_set(r_ptr_local, seed_init + number_points_of_data);

  int tid=0;
  unsigned i=0;
#pragma omp parallel default(none) private(tid, i) shared(r_ptr_local)
  {
    tid = omp_get_thread_num();

    r_ptr_threadprivate = gsl_rng_clone(r_ptr_local);
    gsl_rng_set(r_ptr_threadprivate, tid);

    printf("Thread %d: counter %d\n", tid, counter);
  }

  std::vector<std::vector<likelihood_point>> points_for_kriging_not_flipped (number_points_of_data);
  std::vector<std::vector<likelihood_point>> points_for_kriging_flipped (number_points_of_data);
  std::vector<likelihood_point> points_for_integration = std::vector<likelihood_point> (1);

  std::ifstream input_file(input_file_name);
  if (input_file.is_open()) {
    for (i=0; i<number_points_of_data; ++i) {
      for (unsigned m=0; m<number_points_per_datum; ++m) {
	likelihood_point current_lp = likelihood_point();
	input_file >> current_lp;

	if (current_lp.FLIPPED) {
	  points_for_kriging_flipped[i].push_back(current_lp);
	} else {
	  points_for_kriging_not_flipped[i].push_back(current_lp);
	}

      }
      std::cout << points_for_kriging_not_flipped[i].size() << "\n";
      std::cout << points_for_kriging_flipped[i].size() << "\n";
      std::cout << std::endl;
    }
  }

  std::vector<GaussianInterpolatorWithChecker> GPs_not_flipped(number_points_of_data);
  std::vector<GaussianInterpolatorWithChecker> GPs_flipped(number_points_of_data);

  auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(none) private(i) shared(number_points_of_data, number_points_per_datum, seed_init, points_for_kriging_flipped, points_for_kriging_not_flipped, GPs_flipped, GPs_not_flipped, points_for_integration)
  {
#pragma omp for
    for (i=0; i<number_points_of_data; ++i) {
      
      if (points_for_kriging_not_flipped[i].size() > 6) {
	GPs_not_flipped[i] = GaussianInterpolatorWithChecker(points_for_integration,
							     points_for_kriging_not_flipped[i]);
      } else {
	GPs_not_flipped[i] = GaussianInterpolatorWithChecker(points_for_integration,
							     points_for_kriging_flipped[i]);
      }
      GPs_not_flipped[i].optimize_parameters();

      if (points_for_kriging_flipped[i].size() > 6) {
	GPs_flipped[i] = GaussianInterpolatorWithChecker(points_for_integration,
							 points_for_kriging_flipped[i]);
      } else {
	GPs_flipped[i] = GaussianInterpolatorWithChecker(points_for_integration,
							 points_for_kriging_not_flipped[i]);
      }
      GPs_flipped[i].optimize_parameters();
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "Generating " << number_points_of_data << " points for integration took "
	    << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << " seconds." << std::endl;

  for (i=0; i<number_points_of_data; ++i) {
    std::string output_file_name = "optimal-parameters-data-point-" + std::to_string(i) + "-" +
      "not-flipped-" +
      input_file_name;

    std::ofstream output_file(output_file_name);
    if (output_file.is_open()) {
      output_file << GPs_not_flipped[i].parameters;
    }
    output_file.close();

    output_file_name = "optimal-parameters-data-point-" + std::to_string(i) + "-" +
      "flipped-" +
      input_file_name;

    output_file = std::ofstream(output_file_name);
    if (output_file.is_open()) {
      output_file << GPs_flipped[i].parameters;
    }
    output_file.close();
  }
  
  // parameters_nominal params = parameters_nominal();
  // std::vector<double> L =
  //   std::vector<double> {0.1,
  // 			 0.0,0.1,
  // 			 0.0,0.0,0.1,
  // 			 0.0,0.0,0.0,0.1,
  // 			 0.0,0.0,0.0,0.0,0.1,
  // 			 0.0,0.0,0.0,0.0,0.0,0.1,
  // 			 0.0,0.0,0.0,0.0,0.0,0.0,0.1};

  // params.lower_triag_mat_as_vec = L;
  // params.sigma_2 = 0.001;
  // params.phi = -10.01;
  // params.tau_2 = 1.0;

  // std::cout << "points_for_kriging.size() = " << points_for_kriging.size() << std::endl;
  // std::cout << "points_for_integration.size() = " << points_for_integration.size() << std::endl;

  // GaussianInterpolator GP_prior = GaussianInterpolator(points_for_integration,
  // 						       points_for_kriging,
  // 						       params);

  // GP_prior.optimize_parameters();
  // params = GP_prior.parameters;
  // std::cout << "optimal parameters = \n" << params << std::endl;

  // double integral = 0;
  // for (unsigned i=0; i<points_for_integration.size(); ++i) {
  //   double add = GP_prior(points_for_integration[i]);
  //   integral = integral +
  //     1.0/M * add;
  // }
  // std::cout << "Integral = " << integral << std::endl;

  // std::string output_file_name = "optimal-parameters-" + std::to_string(N) + "-" + input_file_name;
  // std::ofstream output_file(output_file_name);

  // if (output_file.is_open()) {
  //   output_file << params;
  // }
  // output_file.close();

  return 0;
}
