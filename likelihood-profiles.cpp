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

// Program produces numerical and small-t likelihoods for points
// supplied.
int main(int argc, char *argv[]) {
  if (argc < 8 || argc > 8) {
    printf("You must provide input\n");
    printf("The input is: \n\nnumber data points;\nrho_basis;\nsigma_x;\nsigma_y;\ndx_likelihood;\nfile prefix;\ndata_file for points in BM format;\n");
    exit(0);
  }

  unsigned N = std::stoi(argv[1]);
  double rho_basis = std::stod(argv[2]);
  double sigma_x_basis = std::stod(argv[3]);
  double sigma_y_basis = std::stod(argv[4]);
  double dx_likelihood = std::stod(argv[5]);
  std::string file_prefix = argv[6];
  std::string input_file_name = argv[7];
  double dx = 1.0/250.0;
  double dx_likelihood_for_small_t = 1e-5;

  static int counter = 0;
  static BivariateGaussianKernelBasis* private_bases;
  static gsl_rng* r_ptr_threadprivate;

#pragma omp threadprivate(private_bases, counter, r_ptr_threadprivate)
  omp_set_dynamic(0);
  omp_set_num_threads(8);
  
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
  std::vector<likelihood_point> points_for_kriging_small_t (N);

  unsigned m = 100; // how many time points to evaluate 
  unsigned m_eig = 20; // how many eigenvalues
  std::vector<std::vector<double>> lls (N, std::vector<double> (m));
  std::vector<std::vector<double>> lls_small_t (N, std::vector<double> (m));
  std::vector<std::vector<double>> lls_analytic (N, std::vector<double> (m));
  std::vector<std::vector<double>> lls_matched (N, std::vector<double> (m));
  std::vector<std::vector<double>> lls_ansatz (N, std::vector<double> (m));
  std::vector<std::vector<double>> tts (N, std::vector<double> (m));
  
  std::vector<std::vector<double>> eigenvalues (N, std::vector<double> (m_eig));
  std::vector<double> small_ts (N);

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
      BrownianMotion bm_point = BrownianMotion();
      input_file >> bm_point;

      points_for_kriging[i] = bm_point;
      // flipping x-axis if correlation param is negative, as the
      // basis we're using have a positive rho
      if (std::signbit(bm_point.get_rho())) {
	points_for_kriging[i].x_0_tilde = 1 - points_for_kriging[i].x_0_tilde;
	points_for_kriging[i].x_t_tilde = 1 - points_for_kriging[i].x_t_tilde;
	points_for_kriging[i].rho = -1.0*points_for_kriging[i].rho;
      }

      points_for_kriging[i].t_tilde = points_for_kriging[i].t_tilde;
      points_for_kriging[i].log_likelihood = 1.0;

      points_for_kriging_small_t[i] = points_for_kriging[i];
    }
  }

  std::vector<likelihood_point> points_for_integration (1);

  // auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(none) private(i) shared(points_for_kriging, points_for_kriging_small_t, N, seed_init, r_ptr_local, dx_likelihood_for_small_t, m, lls, lls_small_t, lls_analytic, lls_matched, lls_ansatz, eigenvalues, tts, small_ts, m_eig) firstprivate(dx_likelihood, dx)
    {
#pragma omp for
      for (i=0; i<N; ++i) {
	double raw_input_array [2] = {points_for_kriging[i].x_t_tilde,
				      points_for_kriging[i].y_t_tilde};
	gsl_vector_view raw_input = gsl_vector_view_array(raw_input_array,2);
	double log_likelihood = 0.0;
	// for some configurations, the small-time solution doesn't
	// work. Iterate over rho until it does. Always works for
	// rho=0
	double rho_for_small_t = points_for_kriging[i].rho; 
	double sigma_y_for_small_t = points_for_kriging[i].sigma_y_tilde;
	
	BivariateSolver solver = BivariateSolver(private_bases,
						 1.0,
						 points_for_kriging[i].sigma_y_tilde,
						 points_for_kriging[i].rho,
						 0.0,
						 points_for_kriging[i].x_0_tilde,
						 1.0,
						 0.0,
						 points_for_kriging[i].y_0_tilde,
						 1.0,
						 points_for_kriging[i].t_tilde,
						 dx);

	for (unsigned ii=0; ii<m_eig; ++ii) {
	  eigenvalues[i][ii] = gsl_vector_get(solver.get_evals(), ii);
	}

	std::vector<BivariateImageWithTime> small_positions =
	  solver.small_t_image_positions_type_41_symmetric(false);
	
	double small_t = small_positions[0].get_t();
	// for some configurations, the small-time solution doesn't
	// work, in which case small-t is negative. Shrink rho
	// sufficiently s.t. the small-t solution works.  It's
	// always guaranteed to work for sufficiently small rho.
	while (std::signbit(small_t)) {
	  rho_for_small_t = rho_for_small_t * 0.95;
	  printf("Trying sigma_y_for_small_t=%f\n", sigma_y_for_small_t);
	  printf("Trying rho_for_small_t=%f\n", rho_for_small_t);
	  solver.set_diffusion_parameters_and_data_small_t(1.0,
							   sigma_y_for_small_t,
							   rho_for_small_t,
							   points_for_kriging[i].t_tilde,
							   0.0,
							   points_for_kriging[i].x_0_tilde,
							   1.0,
							   0.0,
							   points_for_kriging[i].y_0_tilde,
							   1.0);
	  small_positions = solver.small_t_image_positions_type_41_symmetric(false);
	  small_t = small_positions[0].get_t();
	}

	small_ts[i] = small_t;

	raw_input_array[0] = points_for_kriging[i].x_t_tilde;
	raw_input_array[1] = points_for_kriging[i].y_t_tilde;
	
	// MATCHING CONSTANTS START
	std::vector<double> betas (4);
	for (unsigned ii=0; ii<4; ++ii) {
	  const BivariateImageWithTime& differentiable_image = small_positions[ii];
	  double x0 = gsl_vector_get(differentiable_image.get_position(), 0);
	  double y0 = gsl_vector_get(differentiable_image.get_position(), 1);
	  double x = gsl_vector_get(&raw_input.vector, 0);
	  double y = gsl_vector_get(&raw_input.vector, 1);

	  betas[ii] = 
	    1.0/(2.0*sigma_y_for_small_t*sigma_y_for_small_t*(1-rho_for_small_t*rho_for_small_t))*
	    ( std::pow((x-x0)*sigma_y_for_small_t, 2.0) +
	      std::pow((y-y0), 2.0) -
	      2*rho_for_small_t*(x-x0)*(y-y0)*sigma_y_for_small_t );
	  printf("sigma_tilde = %f, rho = %f, x0 = %f, y0 = %f, beta = %f\n", 
		 sigma_y_for_small_t, 
		 rho_for_small_t,
		 x0, y0, betas[ii]);
	}
	std::vector<double>::iterator result = std::max_element(betas.begin(),
								betas.end());
	double max_beta = *result;
	double lambda = eigenvalues[i][0];
	double t_tilde_1 = 0.5;
	double t_tilde_2 = 2.0;
	double dt = 0.1;
	solver.set_diffusion_parameters_and_data_small_t(1.0,
							 points_for_kriging[i].sigma_y_tilde,
							 points_for_kriging[i].rho,
							 t_tilde_1,
							 0.0,
							 points_for_kriging[i].x_0_tilde,
							 1.0,
							 0.0,
							 points_for_kriging[i].y_0_tilde,
							 1.0);
	double f1 = log(solver.numerical_likelihood(&raw_input.vector,
						     dx_likelihood));
	solver.set_diffusion_parameters_and_data_small_t(1.0,
							 points_for_kriging[i].sigma_y_tilde,
							 points_for_kriging[i].rho,
							 t_tilde_2,
							 0.0,
							 points_for_kriging[i].x_0_tilde,
							 1.0,
							 0.0,
							 points_for_kriging[i].y_0_tilde,
							 1.0);
	double f2 = log(solver.numerical_likelihood(&raw_input.vector,
						       dx_likelihood));
	double alpha = -(f1-f2 + lambda*(t_tilde_2 - t_tilde_1))/log(t_tilde_2/t_tilde_1);
	double big_C = f2 - alpha*log(t_tilde_2) - lambda*t_tilde_2;
	double t_tilde_star = -alpha/lambda;
	double gamma = max_beta/t_tilde_star;
	solver.set_diffusion_parameters_and_data_small_t(1.0,
							 points_for_kriging[i].sigma_y_tilde,
							 points_for_kriging[i].rho,
							 t_tilde_star,
							 0.0,
							 points_for_kriging[i].x_0_tilde,
							 1.0,
							 0.0,
							 points_for_kriging[i].y_0_tilde,
							 1.0);
	double fstar = log(solver.numerical_likelihood(&raw_input.vector,
						       dx_likelihood));
	double kappa = fstar + 
	  gamma*log(t_tilde_star) + max_beta/t_tilde_star;
	
	printf("beta = %f, alpha = %f, eigenvalues = %f, C = %f\n", 
	       max_beta, alpha, eigenvalues[i][0], big_C);
	// MATCH CONSTANTS END
	
	// filling in vector of times to evaluate
	double t_lower = 0.005; //small_t;
	double t_upper = 4.0; // points_for_kriging[i].t_tilde*1.5;
	dt = (t_upper - t_lower)/(m-1);
	std::vector<double> ts(m);
	unsigned nn=0;
	std::generate(ts.begin(), ts.end(), [&] () mutable { double out = t_lower + dt*nn; nn++; return out; });
	tts[i] = ts;
	
	for (unsigned j=0; j<m; ++j) {
	  solver.set_diffusion_parameters_and_data_small_t(1.0,
							   points_for_kriging[i].sigma_y_tilde,
							   points_for_kriging[i].rho,
							   ts[j],
							   0.0,
							   points_for_kriging[i].x_0_tilde,
							   1.0,
							   0.0,
							   points_for_kriging[i].y_0_tilde,
							   1.0);
	  lls[i][j] = log(solver.numerical_likelihood(&raw_input.vector,
						      dx_likelihood));
	  
	  solver.set_diffusion_parameters_and_data_small_t(1.0,
							   sigma_y_for_small_t,
							   rho_for_small_t,
							   ts[j],
							   0.0,
							   points_for_kriging[i].x_0_tilde,
							   1.0,
							   0.0,
							   points_for_kriging[i].y_0_tilde,
							   1.0);
	  
	  lls_small_t[i][j] = solver.
	    likelihood_small_t_41_truncated_symmetric(&raw_input.vector,
						      ts[j],
						      dx_likelihood_for_small_t);
	  
	  lls_analytic[i][j] = solver.analytic_likelihood(&raw_input.vector, 1000);

	  lls_matched[i][j] = kappa - gamma*log(ts[j]) - max_beta/ts[j];

	  lls_ansatz[i][j] = big_C + alpha*log(ts[j]) + lambda*ts[j];
	  
	  printf("%f %f\n", lls[i][j], lls_small_t[i][j]);
	}
	
	  log_likelihood = solver.
	    likelihood_small_t_41_truncated_symmetric(&raw_input.vector,
						      points_for_kriging[i].t_tilde,
						      dx_likelihood_for_small_t);
	  
	  points_for_kriging_small_t[i].log_likelihood = log_likelihood;
	  
	  points_for_kriging[i].log_likelihood = 
	    log(solver.numerical_likelihood(&raw_input.vector,
					    dx_likelihood));
	  
	printf("Thread %d with address %p produces log_likelihood %g\n",
	       omp_get_thread_num(),
	       private_bases,
	       log_likelihood);
      }
    }
    // auto t2 = std::chrono::high_resolution_clock::now();


    std::string output_file_name = file_prefix +
      "-number-points-" + argv[1] +
      "-rho_basis-" + argv[2] +
      "-sigma_x_basis-" + argv[3] +
      "-sigma_y_basis-" + argv[4] +
      "-dx_likelihood-" + argv[5] +
      ".csv";

    std::ofstream output_file;
    output_file.open(output_file_name);

    output_file << "nan = NA;\n";
    output_file << "inf = Inf;\n";
    output_file << "eigenvalues = vector(mode=\"list\", length=" << N << ");\n";
    output_file << "lls = vector(mode=\"list\", length=" << N << ");\n";
    output_file << "lls_small_t = vector(mode=\"list\", length=" << N << ");\n";
    output_file << "lls_analytic = vector(mode=\"list\", length=" << N << ");\n";
    output_file << "lls_matched = vector(mode=\"list\", length=" << N << ");\n";
    output_file << "lls_ansatz = vector(mode=\"list\", length=" << N << ");\n";
    output_file << "small_ts = rep(NA, length=" << N << ");\n";
    output_file << "ts = vector(mode=\"list\", length=" << N << ");\n";

    for (i=0; i<N; ++i) {
      output_file << points_for_kriging[i];


      output_file << "eigenvalues[[" << i+1 << "]] = c(";
      for (unsigned j=0; j<eigenvalues[i].size(); ++j) {
	if (j<eigenvalues[i].size()-1) {
	  output_file << eigenvalues[i][j] << ",";
	} else {
	  output_file << eigenvalues[i][j];
	}
      }
      output_file << ");\n";

      
      output_file << "lls[[" << i+1 << "]] = c(";
      for (unsigned j=0; j<lls[i].size(); ++j) {
	if (j<lls[i].size()-1) {
	  output_file << lls[i][j] << ",";
	} else {
	  output_file << lls[i][j];
	}
      }
      output_file << ");\n";


      output_file << "lls_small_t[[" << i+1 << "]] = c(";
      for (unsigned j=0; j<lls_small_t[i].size(); ++j) {
	if (j<lls_small_t[i].size()-1) {
	  output_file << lls_small_t[i][j] << ",";
	} else {
	  output_file << lls_small_t[i][j];
	}
      }
      output_file << ");\n";
      

      output_file << "lls_analytic[[" << i+1 << "]] = c(";
      for (unsigned j=0; j<lls_analytic[i].size(); ++j) {
	if (j<lls_analytic[i].size()-1) {
	  output_file << lls_analytic[i][j] << ",";
	} else {
	  output_file << lls_analytic[i][j];
	}
      }
      output_file << ");\n";


      output_file << "lls_matched[[" << i+1 << "]] = c(";
      for (unsigned j=0; j<lls_matched[i].size(); ++j) {
	if (j<lls_matched[i].size()-1) {
	  output_file << lls_matched[i][j] << ",";
	} else {
	  output_file << lls_matched[i][j];
	}
      }
      output_file << ");\n";

      output_file << "lls_ansatz[[" << i+1 << "]] = c(";
      for (unsigned j=0; j<lls_ansatz[i].size(); ++j) {
	if (j<lls_ansatz[i].size()-1) {
	  output_file << lls_ansatz[i][j] << ",";
	} else {
	  output_file << lls_ansatz[i][j];
	}
      }
      output_file << ");\n";


      output_file << "ts[[" << i+1 << "]] = c(";
      for (unsigned j=0; j<tts[i].size(); ++j) {
	if (j<tts[i].size()-1) {
	  output_file << tts[i][j] << ",";
	} else {
	  output_file << tts[i][j];
	}
      }
      output_file << ");\n";

      output_file << "t_tilde_" << i << "=" << points_for_kriging[i].t_tilde << ";\n";
      output_file << "small_ts[" << i+1 << "]=" << small_ts[i] << ";\n";
    }

    output_file << "## par(mar=c(2,2,2,2), mfrow=c(" << N << ",1));\n";
    for (i=0; i<N; ++i) {
      output_file << "## plot(ts[[" << i+1 << "]], lls[[" << i+1 << "]], ylim=c(-30,10));\n";
      output_file << "## lines(ts[[" << i+1 << "]], lls_small_t[[" << i+1 << "]]);\n";
      output_file << "## lines(ts[[" << i+1 << "]], lls_analytic[[" << i+1 << "]], col=2);\n";
      output_file << "## abline(v = t_tilde_" << i << ");\n\n";
    }

    output_file.close();

    gsl_rng_free(r_ptr_local);
    return 0;
}
