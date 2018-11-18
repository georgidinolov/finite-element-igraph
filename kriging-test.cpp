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
  double dx = 1.0/450.0;
  double dx_likelihood_for_small_t = 1e-5;

  static int counter = 0;
  static BivariateGaussianKernelBasis* private_bases;
  static gsl_rng* r_ptr_threadprivate;

#pragma omp threadprivate(private_bases, counter, r_ptr_threadprivate)
  omp_set_dynamic(0);
  omp_set_num_threads(30);
  
  printf("## starting basis_positive\n");
  BivariateGaussianKernelBasis basis_positive =
    BivariateGaussianKernelBasis(dx,
  				 rho_basis,
  				 sigma_x_basis,
  				 sigma_y_basis,
  				 1.0,
  				 1.0);
  printf("## ending basis_positive\n");

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
  std::vector<likelihood_point> points_for_kriging_duplicate (3*N);
  std::vector<likelihood_point> points_for_kriging_analytic (N);
  std::vector<likelihood_point> points_for_kriging_small_t (N);
  std::vector<double> upper_bound (N);
  std::vector<double> lower_bound (N);
  std::vector<double> eigenvalues_first (N);
  std::vector<double> eigenvalues_last (N);
  std::ifstream input_file(input_file_name);

  double log_t_min = log(1e-4);
  double log_t_max = log(5);
  double delta_log_t = (log_t_max-log_t_min)/N;
  double log_tt = log_t_min - delta_log_t;
  std::vector<double> log_ts(N);
  std::generate(log_ts.begin(),
		log_ts.end(),
		[&] ()->double {log_tt = log_tt + delta_log_t; return log_tt; });

  if (input_file.is_open()) {
    for (i=0; i<N; ++i) {
      // input_file >> points_for_kriging[i];
      BrownianMotion bm_point = BrownianMotion();
      input_file >> bm_point;

      points_for_kriging[i] = bm_point;
      points_for_kriging[i].log_likelihood = 1.0;
      // points_for_kriging[i].rho = 0.0;
      // points_for_kriging[i].t_tilde = exp(log_ts[i]);
      // points_for_kriging[i].sigma_y_tilde = 0.50;

      points_for_kriging_analytic[i] = points_for_kriging[i];
      points_for_kriging_small_t[i] = points_for_kriging[i];
    }
  }

  std::vector<likelihood_point> points_for_integration (1);

  auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(none) private(i) shared(points_for_kriging_duplicate, points_for_kriging, points_for_kriging_analytic, points_for_kriging_small_t, upper_bound, lower_bound, N, seed_init, r_ptr_local, dx_likelihood_for_small_t, eigenvalues_last, eigenvalues_first) firstprivate(dx_likelihood, dx)
    {
#pragma omp for
      for (i=0; i<N; ++i) {
	double raw_input_array [2] = {points_for_kriging[i].x_t_tilde,
				      points_for_kriging[i].y_t_tilde};
	gsl_vector_view raw_input = gsl_vector_view_array(raw_input_array,2);
	
	double rho = points_for_kriging[i].rho;
	double sigma_y_tilde = points_for_kriging[i].sigma_y_tilde;
	double log_likelihood = 0.0;
	double log_likelihood_analytic = 0.0;
	double log_likelihood_small_t = 0.0;

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

	  eigenvalues_first[i] = gsl_vector_get(solver.get_evals(), 1);
	  eigenvalues_last[i] = gsl_vector_get(solver.get_evals(), solver.get_evals()->size - 1);
	  
	  // HERE I'M TESTING THE ASYMPTOTIC MATCHING CONDITION FOR A SINGLE EIGENPAIR
	  std::vector<BivariateImageWithTime> small_positions =
	    solver.small_t_image_positions_type_41_symmetric(false);
	  double small_t = 0.00365; // small_positions[0].get_t();
	  double current_t = points_for_kriging[i].t_tilde;
	  std::vector<double> current_modes (small_positions.size());

	  double log_CC = -1.0*(log(2.0)+
				2*log(sigma_y_tilde) + 
				log(small_t)+log(1-rho*rho));
	  double cov_matrix_arr [4];
	  gsl_matrix_view cov_matrix_view = gsl_matrix_view_array(cov_matrix_arr, 2,2);
	  gsl_matrix* cov_matrix = &cov_matrix_view.matrix;

	  gsl_matrix_set(cov_matrix, 0,0,
			 1.0*1.0*small_t);
	  gsl_matrix_set(cov_matrix, 0,1,
			 rho*1*sigma_y_tilde*small_t);
	  gsl_matrix_set(cov_matrix, 1,0,
			 rho*1*sigma_y_tilde*small_t);
	  gsl_matrix_set(cov_matrix, 1,1,
			 sigma_y_tilde*sigma_y_tilde*small_t);
	  
	  double out = 0;
	  MultivariateNormal mvtnorm = MultivariateNormal();
	  std::vector<int> terms_signs = std::vector<int> (small_positions.size());
	  std::vector<double> log_terms = std::vector<double> (small_positions.size());

	  double x = gsl_vector_get(&raw_input.vector, 0);
	  double y = gsl_vector_get(&raw_input.vector, 1);
	  std::vector<double> log_A_coefs (4);
	  std::vector<double> betas (4);
	  std::vector<double> likelihoods (4);
	  std::vector<double> dPdaxs = solver.dPdax(&raw_input.vector,
						    dx_likelihood_for_small_t);
	  std::vector<double> dPdbxs = solver.dPdbx(&raw_input.vector,
						    dx_likelihood_for_small_t);
	  std::vector<double> dPdays = solver.dPday(&raw_input.vector,
						    dx_likelihood_for_small_t);
	  std::vector<double> dPdbys = solver.dPdby(&raw_input.vector,
						    dx_likelihood_for_small_t);

	  for (unsigned ii=0; ii<small_positions.size(); ++ii) {
	    const BivariateImageWithTime& differentiable_image = small_positions[ii];

	    double x0 = gsl_vector_get(differentiable_image.get_position(), 0);
	    double y0 = gsl_vector_get(differentiable_image.get_position(), 1);

	    log_A_coefs[ii] = 
	      - log(2*M_PI*sigma_y_tilde*std::sqrt(1-rho*rho))
	      - 4.0*log(2*sigma_y_tilde*sigma_y_tilde*(1-rho*rho))
	      + log(std::abs(dPdaxs[ii]))
	      + log(std::abs(dPdbxs[ii])) 
	      + log(std::abs(dPdays[ii])) 
	      + log(std::abs(dPdbys[ii]));

	    betas[ii] = 
	      1.0/(2.0*sigma_y_tilde*sigma_y_tilde*(1-rho*rho))*
	      ( std::pow((x-x0)*sigma_y_tilde, 2.0) +
	  	std::pow((y-y0), 2.0) -
	  	2*rho*(x-x0)*(y-y0)*sigma_y_tilde );

	    log_terms[ii] = log_A_coefs[ii] +
	      +log(std::abs(betas[ii] - 5*small_t))
	      -betas[ii]/small_t
	      -7*log(small_t);

	    terms_signs[ii] = 1;
	    if (std::signbit(differentiable_image.get_mult_factor()*
	    		     dPdaxs[ii]*dPdbxs[ii]*dPdays[ii]*dPdbys[ii]*
			     (betas[ii]-5*small_t)))
	      {
	    	terms_signs[ii] = -1;
	      }
	    
	  }

	  std::vector<double>::iterator result = std::max_element(log_terms.begin(),
								  log_terms.end());

	  out = 0;
	  for (unsigned ii=0; ii<small_positions.size(); ++ii) {
	    out = out +
	      terms_signs[ii]*std::exp(log_terms[ii]-*result);
	  }
	  out = std::log(out) + *result;

	  result = std::min_element(betas.begin(),
				    betas.end());

	  eigenvalues_last[i] = *result;

	  double rho_current = rho;
	  // while (small_t < 1e-4) {
	  //   printf("small_t too small at %g with rho=%g\n", small_t, rho_current);
	  //   rho_current = rho_current * 0.90;
	  //   solver.set_diffusion_parameters_and_data(1.0,
	  // 					     points_for_kriging[i].sigma_y_tilde,
	  // 					     rho_current,
	  // 					     points_for_kriging[i].t_tilde,
	  // 					     0.0,
	  // 					     points_for_kriging[i].x_0_tilde,
	  // 					     1.0,
	  // 					     0.0,
	  // 					     points_for_kriging[i].y_0_tilde,
	  // 					     1.0);
	  //   small_positions = solver.small_t_image_positions_type_41_symmetric(true);
	  //   small_t = small_positions[0].get_t();
	  //   printf("small_t  now at %g with rho=%g\n", small_t, rho_current);
	  // }
	  points_for_kriging_duplicate[i*3 + 0] = points_for_kriging[i];
	  points_for_kriging_duplicate[i*3 + 0].rho = rho_current;
	  points_for_kriging_duplicate[i*3 + 0].t_tilde = small_t;
	  points_for_kriging_duplicate[i*3 + 0].log_likelihood = 
	    solver.likelihood_small_t_41_truncated_symmetric(&raw_input.vector,
							     small_t,
							     dx_likelihood_for_small_t);

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
	  std::vector<double>::iterator result_max =
	    std::max_element(current_modes.begin(), current_modes.end());

	  upper_bound[i] = *result_max;
	  lower_bound[i] = *result_min;

	  solver.set_diffusion_parameters_and_data(1.0,
						   points_for_kriging[i].sigma_y_tilde,
						   rho,
						   *result_min,
						   0.0,
						   points_for_kriging[i].x_0_tilde,
						   1.0,
						   0.0,
						   points_for_kriging[i].y_0_tilde,
						   1.0);
	  points_for_kriging_duplicate[i*3 + 1] = points_for_kriging[i];
	  points_for_kriging_duplicate[i*3 + 1].t_tilde = *result_min;
	  points_for_kriging_duplicate[i*3 + 1].log_likelihood = 
	    log(solver.numerical_likelihood_first_order(&raw_input.vector,
							dx_likelihood));
	  
	  solver.set_diffusion_parameters_and_data(1.0,
						   points_for_kriging[i].sigma_y_tilde,
						   rho,
						   *result_max,
						   0.0,
						   points_for_kriging[i].x_0_tilde,
						   1.0,
						   0.0,
						   points_for_kriging[i].y_0_tilde,
						   1.0);
	  points_for_kriging_duplicate[i*3 + 2] = points_for_kriging[i];
	  points_for_kriging_duplicate[i*3 + 2].t_tilde = *result_max;
	  points_for_kriging_duplicate[i*3 + 2].log_likelihood = 
	    log(solver.numerical_likelihood_first_order(&raw_input.vector,
							dx_likelihood));

	  solver.set_diffusion_parameters_and_data(1.0,
						   points_for_kriging[i].sigma_y_tilde,
						   rho,
						   points_for_kriging[i].t_tilde,
						   0.0,
						   points_for_kriging[i].x_0_tilde,
						   1.0,
						   0.0,
						   points_for_kriging[i].y_0_tilde,
						   1.0);
	  log_likelihood_analytic = solver.analytic_likelihood(&raw_input.vector,
							       1000);
	  log_likelihood_small_t = 
	    solver.likelihood_small_t_41_truncated_symmetric(&raw_input.vector,
							     points_for_kriging[i].t_tilde,
							     dx_likelihood_for_small_t);
	  log_likelihood = 
	    log(solver.numerical_likelihood_first_order(&raw_input.vector,
							dx_likelihood));
	  
	} else {
	  BivariateSolver solver = BivariateSolver(private_bases,
						   1.0,
						   points_for_kriging[i].sigma_y_tilde,
						   -rho,
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

	  eigenvalues_first[i] = gsl_vector_get(solver.get_evals(), 0);
	  eigenvalues_last[i] = gsl_vector_get(solver.get_evals(), solver.get_evals()->size - 1);
	  
	  std::vector<BivariateImageWithTime> small_positions =
	    solver.small_t_image_positions_type_41_symmetric(false);
	  double small_t = small_positions[0].get_t();
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
	  std::vector<double>::iterator result_max =
	    std::min_element(current_modes.begin(), current_modes.end());

	  upper_bound[i] = *result_max;
	  lower_bound[i] = *result_min;

	  points_for_kriging_duplicate[i*3 + 0] = points_for_kriging[i];
	  points_for_kriging_duplicate[i*3 + 0].t_tilde = small_t;
	  points_for_kriging_duplicate[i*3 + 0].log_likelihood = 
	    solver.
	    numerical_likelihood_first_order_small_t_ax_bx(&raw_input.vector,
							   dx_likelihood_for_small_t);

	  solver.set_diffusion_parameters_and_data(1.0,
						   points_for_kriging[i].sigma_y_tilde,
						   -rho,
						   *result_min,
						   -1.0,
						   -points_for_kriging[i].x_0_tilde,
						   0.0,
						   0.0,
						   points_for_kriging[i].y_0_tilde,
						   1.0);
	  points_for_kriging_duplicate[i*3 + 1] = points_for_kriging[i];
	  points_for_kriging_duplicate[i*3 + 1].t_tilde = *result_min;
	  points_for_kriging_duplicate[i*3 + 1].log_likelihood = 
	    log(solver.numerical_likelihood(&raw_input.vector,
					    dx_likelihood));
	  
	  solver.set_diffusion_parameters_and_data(1.0,
						   points_for_kriging[i].sigma_y_tilde,
						   rho,
						   *result_max,
						   -1.0,
						   -points_for_kriging[i].x_0_tilde,
						   -.0,
						   0.0,
						   points_for_kriging[i].y_0_tilde,
						   1.0);
	  points_for_kriging_duplicate[i*3 + 2] = points_for_kriging[i];
	  points_for_kriging_duplicate[i*3 + 2].t_tilde = *result_min;
	  points_for_kriging_duplicate[i*3 + 2].log_likelihood = 
	    log(solver.numerical_likelihood_first_order(&raw_input.vector,
							dx_likelihood));

	  solver.set_diffusion_parameters_and_data(1.0,
						   points_for_kriging[i].sigma_y_tilde,
						   rho,
						   points_for_kriging[i].t_tilde,
						   -1.0,
						   -points_for_kriging[i].x_0_tilde,
						   0.0,
						   0.0,
						   points_for_kriging[i].y_0_tilde,
						   1.0);

	  log_likelihood = log(solver.numerical_likelihood_first_order(&raw_input.vector,
								       dx_likelihood));

	  log_likelihood_analytic = solver.analytic_likelihood(&raw_input.vector,
							       1000);

	  log_likelihood_small_t = 
	    solver.likelihood_small_t_41_truncated_symmetric(&raw_input.vector,
							     points_for_kriging[i].t_tilde,
							     dx_likelihood_for_small_t);
	}

	points_for_kriging[i].log_likelihood = log_likelihood;
	points_for_kriging_analytic[i].log_likelihood = log_likelihood_analytic;
	points_for_kriging_small_t[i].log_likelihood = log_likelihood_small_t;
	printf("Thread %d with address %p produces log_likelihood %g\n",
	       omp_get_thread_num(),
	       private_bases,
	       log_likelihood);
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
    std::string output_file_name_analytic = file_prefix +
      "-analytic-" +
      "-number-points-" + argv[1] +
      "-rho_basis-" + argv[2] +
      "-sigma_x_basis-" + argv[3] +
      "-sigma_y_basis-" + argv[4] +
      "-dx_likelihood-" + argv[5] +
      ".csv";
    std::string output_file_name_small_t = file_prefix +
      "-small_t-" +
      "-number-points-" + argv[1] +
      "-rho_basis-" + argv[2] +
      "-sigma_x_basis-" + argv[3] +
      "-sigma_y_basis-" + argv[4] +
      "-dx_likelihood-" + argv[5] +
      ".csv";

    std::string output_file_name_upper_bound = file_prefix +
      "-upper_bound-" +
      "-number-points-" + argv[1] +
      "-rho_basis-" + argv[2] +
      "-sigma_x_basis-" + argv[3] +
      "-sigma_y_basis-" + argv[4] +
      "-dx_likelihood-" + argv[5] +
      ".csv";

    std::string output_file_name_lower_bound = file_prefix +
      "-lower_bound-" +
      "-number-points-" + argv[1] +
      "-rho_basis-" + argv[2] +
      "-sigma_x_basis-" + argv[3] +
      "-sigma_y_basis-" + argv[4] +
      "-dx_likelihood-" + argv[5] +
      ".csv";

    std::string output_file_name_eigenvalues_first = file_prefix +
      "-eigenvalue_first-" +
      "-number-points-" + argv[1] +
      "-rho_basis-" + argv[2] +
      "-sigma_x_basis-" + argv[3] +
      "-sigma_y_basis-" + argv[4] +
      "-dx_likelihood-" + argv[5] +
      ".csv";

    std::string output_file_name_eigenvalues_last = file_prefix +
      "-eigenvalue_last-" +
      "-number-points-" + argv[1] +
      "-rho_basis-" + argv[2] +
      "-sigma_x_basis-" + argv[3] +
      "-sigma_y_basis-" + argv[4] +
      "-dx_likelihood-" + argv[5] +
      ".csv";

    std::string output_file_name_duplicate = file_prefix +
      "-duplicate-" +
      "-number-points-" + argv[1] +
      "-rho_basis-" + argv[2] +
      "-sigma_x_basis-" + argv[3] +
      "-sigma_y_basis-" + argv[4] +
      "-dx_likelihood-" + argv[5] +
      ".csv";


    std::ofstream output_file;
    std::ofstream output_file_analytic;
    std::ofstream output_file_small_t;
    std::ofstream output_file_upper_bound;
    std::ofstream output_file_lower_bound;
    std::ofstream output_file_eigenvalue_first;
    std::ofstream output_file_eigenvalue_last;
    std::ofstream output_file_duplicate;
    
    output_file.open(output_file_name);
    output_file_analytic.open(output_file_name_analytic);
    output_file_small_t.open(output_file_name_small_t);
    output_file_upper_bound.open(output_file_name_upper_bound);
    output_file_lower_bound.open(output_file_name_lower_bound);
    output_file_eigenvalue_first.open(output_file_name_eigenvalues_first);
    output_file_eigenvalue_last.open(output_file_name_eigenvalues_last);
    output_file_duplicate.open(output_file_name_duplicate);

    for (i=0; i<N; ++i) {
      output_file << points_for_kriging[i];
      output_file_analytic << points_for_kriging_analytic[i];
      output_file_small_t << points_for_kriging_small_t[i];
      output_file_upper_bound << "upper_bound=" << upper_bound[i]
			      << ";\n";
      output_file_lower_bound << "lower_bound=" << lower_bound[i]
			      << ";\n";
      output_file_eigenvalue_first << "eigenvalue_first=" << eigenvalues_first[i]
				   << ";\n";
      output_file_eigenvalue_last << "eigenvalue_last=" << eigenvalues_last[i]
				   << ";\n";
    }

    for (i=0; i<3*N; ++i) {
      output_file_duplicate << points_for_kriging_duplicate[i];
    }

    output_file.close();
    output_file_analytic.close();
    output_file_small_t.close();
    output_file_eigenvalue_first.close();
    output_file_eigenvalue_last.close();
    output_file_duplicate.close();

    gsl_rng_free(r_ptr_local);
    return 0;
}
