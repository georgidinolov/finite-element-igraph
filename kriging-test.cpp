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
  if (argc < 8 || argc > 8) {
    printf("You must provide input\n");
    printf("The input is: \n\nnumber data points;\nrho_basis;\nsigma_x;\nsigma_y;\ndx_likelihood;\nfile prefix;\ndata_file for points;\n");
    exit(0);
  }

  unsigned N = std::stoi(argv[1]);
  double rho_basis = std::stod(argv[2]);
  double sigma_x_basis = std::stod(argv[3]);
  double sigma_y_basis = std::stod(argv[4]);
  double dx_likelihood = std::stod(argv[5]);
  std::string file_prefix = argv[6];
  std::string input_file_name = argv[7];
  double dx = 1.0/1000.0;

  static int counter = 0;
  static BivariateGaussianKernelBasis* private_bases;
  static gsl_rng* r_ptr_threadprivate;

#pragma omp threadprivate(private_bases, counter, r_ptr_threadprivate)
  omp_set_dynamic(0);
  omp_set_num_threads(3);

  BivariateGaussianKernelBasis basis_positive =
    BivariateGaussianKernelBasis(dx,
				 rho_basis,
				 sigma_x_basis,
				 sigma_y_basis,
				 1.0,
				 1.0);

  long unsigned seed_init = 10;
  gsl_rng * r_ptr_local;
  const gsl_rng_type * Type;
  gsl_rng_env_setup();
  Type = gsl_rng_default;
  r_ptr_local = gsl_rng_alloc(Type);
  gsl_rng_set(r_ptr_local, seed_init + N);

  int tid=0;
  unsigned i=0;
#pragma omp parallel default(none) private(tid, i) shared(basis_positive, r_ptr_local)
  {
    tid = omp_get_thread_num();

    private_bases = new BivariateGaussianKernelBasis();
    (*private_bases) = basis_positive;

    r_ptr_threadprivate = gsl_rng_clone(r_ptr_local);
    gsl_rng_set(r_ptr_threadprivate, tid);

    printf("Thread %d: counter %d\n", tid, counter);
  }


  std::vector<likelihood_point> points_for_kriging (N);
  std::ifstream input_file(input_file_name);

  if (input_file.is_open()) {
    for (i=0; i<N; ++i) {
      input_file >> points_for_kriging[i];
      points_for_kriging[i].log_likelihood = 1.0;
    }
  }

  std::vector<likelihood_point> points_for_integration (1);

  auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(none) private(i) shared(points_for_kriging, N, seed_init, r_ptr_local) firstprivate(dx_likelihood, dx)
    {
#pragma omp for
      for (i=0; i<N; ++i) {
	double raw_input_array [2] = {points_for_kriging[i].x_t_tilde,
				      points_for_kriging[i].y_t_tilde};
	gsl_vector_view raw_input = gsl_vector_view_array(raw_input_array,2);
	
	double rho = points_for_kriging[i].rho;
	double likelihood = 0.0;
	if (!std::signbit(rho)) {
	  BivariateSolver solver = BivariateSolver(private_bases,
						   1.0,
						   points_for_kriging[i].sigma_y_tilde,
						   rho,
						   0.0,
						   points_for_kriging[i].x_0_tilde,
						   1.0,
						   0.0,
						   points_for_kriging[i].y_0_tilde,
						   1.0,
						   points_for_kriging[i].t_tilde,
						   dx);

	  raw_input_array[0] = points_for_kriging[i].x_t_tilde;
	  raw_input_array[1] = points_for_kriging[i].y_t_tilde;
	  
	  std::vector<BivariateImageWithTime> small_positions =
		  solver.small_t_image_positions_1_3(false);
	  std::vector<double> current_modes (small_positions.size());
	  for (unsigned ii=0; ii<small_positions.size(); ++ii) {
	    const BivariateImageWithTime& image = small_positions[ii];
	    double x = gsl_vector_get(&raw_input.vector, 0);
	    double y = gsl_vector_get(&raw_input.vector, 1);
	    double x_0 = gsl_vector_get(image.get_position(), 0);
	    double y_0 = gsl_vector_get(image.get_position(), 1);
		  
	    double sigma = points_for_kriging[i].sigma_y_tilde;
	    double rho = points_for_kriging[i].rho;
	    double t = points_for_kriging[i].t_tilde;

	    double beta =
	      ( std::pow(x-x_0,2)*std::pow(sigma,2) +
		std::pow(y-y_0,2) -
		2*rho*(x-x_0)*(y-y_0)*sigma )/(2.0*std::pow(sigma,2) * (1-rho*rho));
	    double alpha = 4.0;

	    double mode = beta/(alpha+1);
	    current_modes[ii] = mode;
	    printf("%g, ", mode);
	  }
	  printf("\n");
	  std::vector<double>::iterator result_min =
	    std::min_element(current_modes.begin(), current_modes.end());

	  if (points_for_kriging[i].t_tilde >= *result_min) {
	    likelihood = solver.numerical_likelihood(&raw_input.vector,
						     dx_likelihood);
	  } else {
	    likelihood = std::numeric_limits<double>::quiet_NaN();
	  }
	  // likelihood = solver.likelihood_small_t_type_41_truncated(&raw_input.vector,
	  // 							   small_t,
	  // 							   dx_likelihood_for_small_t);

	} else {
	  BivariateSolver solver = BivariateSolver(private_bases,
						   1.0,
						   points_for_kriging[i].sigma_y_tilde,
						   rho,
						   -1.0,
						   -points_for_kriging[i].x_0_tilde,
						   0.0,
						   0.0,
						   points_for_kriging[i].y_0_tilde,
						   1.0,
						   points_for_kriging[i].t_tilde,
						   dx);

	  raw_input_array[0] = -points_for_kriging[i].x_t_tilde;
	  raw_input_array[1] = points_for_kriging[i].y_t_tilde;
	  
	  std::vector<BivariateImageWithTime> small_positions =
	    solver.small_t_image_positions_1_3(false);
	  std::vector<double> current_modes (small_positions.size());
	  for (unsigned ii=0; ii<small_positions.size(); ++ii) {
	    const BivariateImageWithTime& image = small_positions[ii];
	    double x = gsl_vector_get(&raw_input.vector, 0);
	    double y = gsl_vector_get(&raw_input.vector, 1);
	    double x_0 = gsl_vector_get(image.get_position(), 0);
	    double y_0 = gsl_vector_get(image.get_position(), 1);
		  
	    double sigma = points_for_kriging[i].sigma_y_tilde;
	    double rho = points_for_kriging[i].rho;
	    double t = points_for_kriging[i].t_tilde;

	    double beta =
	      ( std::pow(x-x_0,2)*std::pow(sigma,2) +
		std::pow(y-y_0,2) -
		2*rho*(x-x_0)*(y-y_0)*sigma )/(2.0*std::pow(sigma,2) * (1-rho*rho));
	    double alpha = 4.0;

	    double mode = beta/(alpha+1);
	    current_modes[ii] = mode;
	    printf("%g, ", mode);
	  }
	  printf("\n");
	  std::vector<double>::iterator result_min =
	    std::min_element(current_modes.begin(), current_modes.end());

	  if (points_for_kriging[i].t_tilde >= *result_min) {
	    likelihood = solver.numerical_likelihood(&raw_input.vector,
						     dx_likelihood);
	  } else {
	    likelihood = std::numeric_limits<double>::quiet_NaN();
	  }
	  // likelihood = solver.likelihood_small_t_type_41_truncated(&raw_input.vector,
	  // 							   small_t,
	  // 							   dx_likelihood_for_small_t);

	}


	points_for_kriging[i].log_likelihood = likelihood;
	printf("Thread %d with address %p produces likelihood %g\n",
	       omp_get_thread_num(),
	       private_bases,
	       likelihood);
      }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    std::string output_file_name = file_prefix +
      "-number-points-" + argv[1] +
      "-rho_basis-" + argv[2] +
      "-sigma_x_basis-" + argv[3] +
      "-sigma_y_basis-" + argv[4] +
      "-dx_likelihood-" + argv[5] +
      ".csv";

    std::ofstream output_file;
    output_file.open(output_file_name);

    for (i=0; i<N; ++i) {
      output_file << points_for_kriging[i];
    }
    output_file.close();

    // parameters_nominal params = parameters_nominal();
    // GaussianInterpolator GP_prior = GaussianInterpolator(points_for_integration,
    // 							 points_for_kriging,
    // 							 params);
    // // for (unsigned i=0; i<points_for_kriging.size(); ++i) {
    // //   for (unsigned j=0; j<points_for_kriging.size(); ++j) {
    // // 	std::cout << gsl_matrix_get(GP_prior.Cinv, i,j) << " ";
    // //   }
    // //   std::cout << std::endl;
    // // }

    // double integral = 0;
    // for (unsigned i=0; i<points_for_integration.size(); ++i) {
    //   double add = GP_prior(points_for_integration[i]);
    //   integral = integral +
    // 	add;
    //   if (add < 0) {
    // 	std::cout << points_for_integration[i] << std::endl;
    //   }
    // }
    // std::cout << "Integral = " << integral << std::endl;

    // // MultivariateNormal mvtnorm = MultivariateNormal();
    // // std::cout << mvtnorm.dmvnorm(N,y,mu,C) << std::endl;

    // nlopt::opt opt(nlopt::LN_NELDERMEAD, 32);
    // //    std::vector<double> lb =

    // // std::vector<double> x = params.as_vector();
    // // std::cout << optimization_wrapper(x, NULL, &points_for_kriging) << std::endl;

    gsl_rng_free(r_ptr_local);


    return 0;
}
