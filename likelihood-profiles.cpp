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

gsl_vector* find_weights(const std::vector<double>& ys,
			 const std::vector<double>& t_tildes,
			 const std::vector<double>& alphas,
			 const std::vector<double>& lambdas) {

  // This is the design matrix X with dimensions m =
  // size(t_tildes) x (n = size(lambdas) x o = size(alphas))
  unsigned oo = alphas.size();
  unsigned nn = lambdas.size();
  unsigned mm = ys.size();

  double y_array [mm];
  double X_array [mm*nn*oo];

  // filling out the design matrix
  gsl_matrix_view X_view = gsl_matrix_view_array(X_array, mm, nn*oo);
  gsl_vector_view y_view = gsl_vector_view_array(y_array, mm);

  for (unsigned ii=0; ii<mm; ++ii) {
    gsl_vector_set(&y_view.vector, ii, ys[ii]);

    for (unsigned jj=0; jj<nn; ++jj) {

      for (unsigned kk=0; kk<oo; ++kk) {
	gsl_matrix_set(&X_view.matrix, ii, jj,
		       exp(lambdas[jj]*t_tildes[ii]) *
		       std::pow(t_tildes[ii], alphas[kk]));
      }
    }
  }
  const gsl_matrix* X = &X_view.matrix;

  double work_array [(nn*oo)*(nn*oo)];
  gsl_matrix_view work_view = gsl_matrix_view_array(work_array, nn*oo, nn*oo);

  // X^T * X
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, X, 0.0, &work_view.matrix);

  // b = X^T y
  double b_array [nn*oo];
  gsl_vector_view b_view = gsl_vector_view_array(b_array, nn*oo);
  gsl_blas_dgemv(CblasTrans, 1.0, X, &y_view.vector, 0.0, &b_view.vector);

  // b = (X^T * X) weights
  gsl_vector* weights = gsl_vector_alloc(nn*oo);
  gsl_permutation* p = gsl_permutation_alloc(nn*oo);
  int s = 0;

  gsl_linalg_LU_decomp(&work_view.matrix, p, &s);
  gsl_linalg_LU_solve(&work_view.matrix, p, &b_view.vector, weights);

  gsl_permutation_free(p);

  return (weights);
}

double find_max(const gsl_vector* weights, 
		const std::vector<double>& lambdas, 
		const std::vector<double>& alphas,
		const std::vector<double>& t_tildes,
		double small_t) {
  // Find max according to the approximating solution start
  double t_min = 0.0;
  double t_max = 0.0;
  double w1 = gsl_vector_get(weights, 0);
  double w2 = gsl_vector_get(weights, 1);
  if ((w1 > 0.0) & (w2 > 0.0)) {
    t_min = small_t;
    t_max = t_tildes[t_tildes.size()-1];
  } else if ((w1 > 0.0) & (w2 < 0.0)) {
    t_min = -1.0*log(std::abs(w1/w2))/std::abs(lambdas[1]-lambdas[0]);
    t_max = t_tildes[t_tildes.size()-1];
  } else if ((w1 < 0.0) & (w2 > 0.0)) {
    t_min = small_t;
    t_max = -1.0*log(std::abs(w1/w2))/std::abs(lambdas[1]-lambdas[0]);
  } else {
    t_min = small_t;
    t_max = t_tildes[t_tildes.size()-1];
  }
  
  std::vector<double> ts(1000);
  unsigned nn = 0;
  double dt = (t_max - t_min)/999;
  std::generate(ts.begin(), ts.end(), [&] () mutable { double out = t_min + dt*nn; nn++; return out; });
  std::sort(ts.begin(), ts.end(), [&lambdas, &alphas, &w1, &w2] (double t1, double t2)->bool {
      double Delta = (lambdas[1]-lambdas[0]);
      double d1 = std::abs(lambdas[0] + alphas[0]/t1 + Delta*w2*exp(Delta*t1)/(w1 + w2*exp(Delta*t1)));
      double d2 = std::abs(lambdas[0] + alphas[0]/t2 + Delta*w2*exp(Delta*t2)/(w1 + w2*exp(Delta*t2)));
      return (d1 < d2); });
  double out = (ts[0] + ts[1])/2.0; // this is the maximum point for the approximate solution
  return out;
}

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
  double dx = 1.0/300.0;
  double dx_likelihood_for_small_t = 1e-5;

  static int counter = 0;
  static BivariateGaussianKernelBasis* private_bases;
  static gsl_rng* r_ptr_threadprivate;

#pragma omp threadprivate(private_bases, counter, r_ptr_threadprivate)
  omp_set_dynamic(0);
  omp_set_num_threads(16);

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

  unsigned m = 50; // how many time points to evaluate
  unsigned m_eig = 20; // how many eigenvalues
  std::vector<std::vector<double>> lls (N, std::vector<double> (m));
  std::vector<std::vector<double>> lls_small_t (N, std::vector<double> (m));
  std::vector<std::vector<double>> lls_analytic (N, std::vector<double> (m));
  std::vector<std::vector<double>> lls_LS (N, std::vector<double> (m));
  std::vector<std::vector<double>> lls_matched (N, std::vector<double> (m));
  std::vector<std::vector<double>> lls_ansatz (N, std::vector<double> (m));
  std::vector<std::vector<double>> tts (N, std::vector<double> (m));
  std::vector<std::vector<double>> function_weights (N, std::vector<double> (0));
  std::vector<std::vector<double>> eigenvalues (N, std::vector<double> (m_eig));
  std::vector<double> gammas (N);
  std::vector<double> betas (N);
  std::vector<double> log_omegas (N);
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
#pragma omp parallel default(none) private(i) shared(points_for_kriging, points_for_kriging_small_t, N, seed_init, r_ptr_local, dx_likelihood_for_small_t, m, lls, lls_small_t, lls_analytic, lls_LS, lls_matched, lls_ansatz, eigenvalues, tts, small_ts, m_eig, function_weights, gammas, betas, log_omegas) firstprivate(dx_likelihood, dx)
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
	  // sigma_y_for_small_t = sigma_y_for_small_t * 0.95;
	  // printf("Trying sigma_y_for_small_t=%f\n", sigma_y_for_small_t);
	  // printf("Trying rho_for_small_t=%f\n", rho_for_small_t);
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
	unsigned number_big_t_points = 3;

	// std::vector<double>::const_iterator first_lambda = eigenvalues[i].begin();
	// std::vector<double>::const_iterator last_lambda = eigenvalues[i].begin() + 1;
	// std::vector<double> lambdas (first_lambda, last_lambda);
	std::vector<double> lambdas = std::vector<double> {eigenvalues[i][0],
							   eigenvalues[i][1]};

	std::vector<double> t_tildes_small = std::vector<double> {small_t};
	std::vector<double> alphas (0);
	alphas.push_back(4.0);

	std::vector<double> log_ys_small (t_tildes_small.size());
	std::vector<double> ys_small (t_tildes_small.size());

	std::generate(log_ys_small.begin(), log_ys_small.end(),
		      [n = 0,
		       &solver,
		       &raw_input,
		       &t_tildes_small,
		       &dx_likelihood_for_small_t] () mutable { double out = solver.
										 likelihood_small_t_41_truncated_symmetric(&raw_input.vector,
															   t_tildes_small[n],
															   dx_likelihood_for_small_t);
			n++;
			return out;});
	std::generate(ys_small.begin(), ys_small.end(),
		      [n = 0,
		       &log_ys_small] () mutable { double out = exp(log_ys_small[n]);
			n++;
			return out;});

	std::vector<double> t_tildes (0); // = t_tildes_small;
	std::vector<double> ys (0); // = ys_small;
	std::vector<double> log_ys (0); // = log_ys_small;

	double t_tilde_2 = 0.30;
	while (number_big_t_points > 0) {
	  solver.set_diffusion_parameters_and_data(1.0,
						   points_for_kriging[i].sigma_y_tilde,
						   points_for_kriging[i].rho,
						   t_tilde_2,
						   0.0,
						   points_for_kriging[i].x_0_tilde,
						   1.0,
						   0.0,
						   points_for_kriging[i].y_0_tilde,
						   1.0);
	  double y2 = solver.numerical_likelihood(&raw_input.vector,
						  dx_likelihood);

	  if (std::isnan(y2)) {
	    y2 = 0.0;
	  }

	  while (std::isnan(std::log(y2))) {
	    if (t_tilde_2 <= 4) {
	      t_tilde_2 = t_tilde_2 + 0.50;
	    } else {
	      t_tilde_2 = t_tilde_2 + 20.0;
	    }

	    solver.set_diffusion_parameters_and_data(1.0,
						     points_for_kriging[i].sigma_y_tilde,
						     points_for_kriging[i].rho,
						     t_tilde_2,
						     0.0,
						     points_for_kriging[i].x_0_tilde,
						     1.0,
						     0.0,
						     points_for_kriging[i].y_0_tilde,
						     1.0);
	    y2 = solver.numerical_likelihood(&raw_input.vector,
					     dx_likelihood);
	  }

	  t_tildes.push_back(t_tilde_2);
	  ys.push_back(y2);
	  log_ys.push_back(log(y2));

	  t_tilde_2 = t_tilde_2 + 0.50;
	  number_big_t_points--;
	}

	gsl_vector* weights = find_weights(ys, t_tildes, alphas, lambdas);
	double t_max = find_max(weights, lambdas, alphas, t_tildes, small_t);
	solver.set_diffusion_parameters_and_data(1.0,
						 points_for_kriging[i].sigma_y_tilde,
						 points_for_kriging[i].rho,
						 t_max,
						 0.0,
						 points_for_kriging[i].x_0_tilde,
						 1.0,
						 0.0,
						 points_for_kriging[i].y_0_tilde,
						 1.0);
	double galerkin_like = solver.numerical_likelihood(&raw_input.vector,
							       dx_likelihood);

	double approx_log_like = alphas[0]*log(t_max) + lambdas[0]*t_max + 
	  log(gsl_vector_get(weights, 0) + gsl_vector_get(weights, 1)*exp((lambdas[1]-lambdas[0]) * t_max));
	
	if (!std::isnan(log(galerkin_like)) &
	    (log(galerkin_like) > approx_log_like) ) {

	  ys[1] = galerkin_like;
	  t_tildes[1] = t_max;

	  gsl_vector_free(weights);
	  weights = find_weights(ys, t_tildes, alphas, lambdas);
	  t_max = find_max(weights, lambdas, alphas, t_tildes, small_t);
	}

	for (unsigned ii=0; ii<weights->size; ++ii) {
	  function_weights[i].push_back(gsl_vector_get(weights, ii));
	}

	// log f_matched (t_max) 
	// first solve for gamma, beta at t_max (right-hand side of matching)
	//(-1/t_max    1/t_max^2)(gamma) = (first deriv )
	//( 1/t_max^2 -1/t_max^3)(beta )   (second deriv)
	
	double w1 = gsl_vector_get(weights, 0);
	double w2 = gsl_vector_get(weights, 1);
	
	double Delta = lambdas[1]-lambdas[0];
	double function_val = alphas[0]*log(t_max) + lambdas[0]*t_max + log(w1 + w2*exp(Delta*t_max));
	double first_deriv =  lambdas[0] + alphas[0]/t_max + Delta*w2*exp(Delta*t_max)/(w1 + w2*exp(Delta*t_max));
	double second_deriv = -alphas[0]/(t_max*t_max) + 
	  Delta*Delta*w2*exp(Delta*t_max)/(w1 + w2*exp(Delta*t_max)) -
	  Delta*Delta*w2*w2*exp(2*Delta*t_max)/std::pow(w1 + w2*exp(Delta*t_max), 2.0);
	double deriv_matrix_array [4] = {-1.0/t_max, 1.0/(t_max*t_max), 1.0/(t_max*t_max), -2.0/(t_max*t_max*t_max)};
	gsl_matrix_view deriv_matrix_view = gsl_matrix_view_array(deriv_matrix_array, 2,2);
	
	gsl_vector* gamma_beta_t_max = gsl_vector_alloc(2);
	gsl_vector* b_vector = gsl_vector_alloc(2);
	gsl_vector_set(b_vector, 0, first_deriv);
	gsl_vector_set(b_vector, 1, second_deriv);
	gsl_permutation* p = gsl_permutation_alloc(2);
	int s = 0;

	gsl_linalg_LU_decomp(&deriv_matrix_view.matrix, p, &s);
	gsl_linalg_LU_solve(&deriv_matrix_view.matrix, p, b_vector, gamma_beta_t_max);

	gsl_permutation_free(p);
	gsl_vector_free(b_vector);

	double gamma_t_max = gsl_vector_get(gamma_beta_t_max, 0);
	double beta_t_max = gsl_vector_get(gamma_beta_t_max, 1);
	double log_omega_t_max = function_val + gamma_t_max*log(t_max) + beta_t_max/t_max;

	gammas[i] = gamma_t_max;
	betas[i] = beta_t_max;
	log_omegas[i] = log_omega_t_max;

	gsl_vector_free(gamma_beta_t_max);

	printf("first derivatives = %f, log_omega_t_max = %f; gamma_t_max = %f; beta_t_max = %f; t_max = %f\n",
	       first_deriv, log_omega_t_max, gamma_t_max, beta_t_max, t_max);

	// starting work on LHS matching
	double log_omega_t_small = -1.0*log(M_PI*std::sqrt(2)) - 
	  4.5*(log(2.0) +
	       2.0*log(points_for_kriging[i].sigma_y_tilde) +
	       log(1-points_for_kriging[i].rho*points_for_kriging[i].rho));

	std::vector<double> dPdaxs = solver.dPdax(&raw_input.vector, dx_likelihood);
	std::vector<double> dPdbxs = solver.dPdbx(&raw_input.vector, dx_likelihood);
	std::vector<double> dPdays = solver.dPday(&raw_input.vector, dx_likelihood);
	std::vector<double> dPdbys = solver.dPdby(&raw_input.vector, dx_likelihood);

	std::vector<double> betas_t_small = std::vector<double> (4);
	unsigned nn = 0;
	std::generate(betas_t_small.begin(), 
		      betas_t_small.end(), 
		      [&] () mutable { 
			double beta_t_small = 1.0/(2.0*
						   points_for_kriging[i].sigma_y_tilde*
						   points_for_kriging[i].sigma_y_tilde*
						   (1.0 - points_for_kriging[i].rho*points_for_kriging[i].rho))*
			  dPdaxs[nn]*dPdbxs[nn]*dPdays[nn]*dPdbys[nn];
			nn++;
			return beta_t_small;
		      });

	std::sort(betas_t_small.begin(), betas_t_small.end());
	printf("betas_t_small[0] = %f, betas_t_small[3] = %f\n", 
	       betas_t_small[0],
	       betas_t_small[3]);

	double beta_t_small = betas_t_small[3];
	double gamma_t_small = 4.5;
	// MATCH CONSTANTS END
	
	// filling in vector of times to evaluate
	double t_lower = 0.005; //small_t;
	double t_upper = 4.0; // points_for_kriging[i].t_tilde*1.5;
	double dt = (t_upper - t_lower)/(m-1);
	std::vector<double> ts = std::vector<double> (m);
	nn = 0;
	std::generate(ts.begin(), ts.end(), [&] () mutable { double out = t_lower + dt*nn; nn++; return out; });
	tts[i] = ts;

	for (unsigned j=0; j<m; ++j) {
	  solver.set_diffusion_parameters_and_data(1.0,
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

	  double LS_sol = 0.0;
	  for (unsigned jj=0; jj<lambdas.size(); ++jj) {
	    for (unsigned kk=0; kk<alphas.size(); ++kk) {
	      LS_sol = LS_sol +
		gsl_vector_get(weights, jj)*
		exp(lambdas[jj]*ts[j]) *
		std::pow(ts[j], alphas[kk]);

	    }
	  }
	  lls_LS[i][j] = std::log(LS_sol);

	  double k = 100;
	  double omega = exp(log_omega_t_small)*exp(-k*(ts[j]-small_t)) +
	    exp(log_omega_t_max)*(1 - exp(-k*(ts[j]-small_t)));
	  double gamma = gamma_t_small*exp(-k*(ts[j]-small_t)) +
	    gamma_t_max*(1 - exp(-k*(ts[j]-small_t)));
	  double beta = beta_t_small*exp(-k*(ts[j]-small_t)) +
	    beta_t_max*(1 - exp(-k*(ts[j]-small_t)));
	  
	  lls_matched[i][j] = std::log(omega) - gamma*log(ts[j]) - beta/ts[j];

	  // printf("%f %f\n", lls[i][j], lls_small_t[i][j]);
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

	gsl_vector_free(weights);
      }
    }
    // auto t2 = std::chrono::high_resolution_clock::now();


    std::string output_file_name = file_prefix +
      "-number-points-" + argv[1] +
      "-rho_basis-" + argv[2] +
      "-sigma_x_basis-" + argv[3] +
      "-sigma_y_basis-" + argv[4] +
      "-dx_likelihood-" + argv[5] +
      ".R";

    std::ofstream output_file;
    output_file.open(output_file_name);

    output_file << "nan = NA;\n";
    output_file << "inf = Inf;\n";
    output_file << "eigenvalues = vector(mode=\"list\", length=" << N << ");\n";
    output_file << "lls = vector(mode=\"list\", length=" << N << ");\n";
    output_file << "lls_small_t = vector(mode=\"list\", length=" << N << ");\n";
    output_file << "lls_analytic = vector(mode=\"list\", length=" << N << ");\n";
    output_file << "lls_matched = vector(mode=\"list\", length=" << N << ");\n";
    output_file << "lls_LS = vector(mode=\"list\", length=" << N << ");\n";
    output_file << "lls_ansatz = vector(mode=\"list\", length=" << N << ");\n";
    output_file << "small_ts = rep(NA, length=" << N << ");\n";
    output_file << "ts = vector(mode=\"list\", length=" << N << ");\n";
    output_file << "weights = vector(mode=\"list\", length=" << N << ");\n";

    output_file << "gammas = c(";
    for (unsigned j=0; j<gammas.size(); ++j) {
      if (j<gammas.size()-1) {
	output_file << gammas[j] << ",";
      } else {
	output_file << gammas[j];
      }
    }
    output_file << ");\n";


    output_file << "betas = c(";
    for (unsigned j=0; j<betas.size(); ++j) {
      if (j<betas.size()-1) {
	output_file << betas[j] << ",";
      } else {
	output_file << betas[j];
      }
    }
    output_file << ");\n";


    output_file << "log_omegas = c(";
    for (unsigned j=0; j<log_omegas.size(); ++j) {
      if (j<log_omegas.size()-1) {
	output_file << log_omegas[j] << ",";
      } else {
	output_file << log_omegas[j];
      }
    }
    output_file << ");\n";

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


      output_file << "lls_LS[[" << i+1 << "]] = c(";
      for (unsigned j=0; j<lls_LS[i].size(); ++j) {
	if (j<lls_LS[i].size()-1) {
	  output_file << lls_LS[i][j] << ",";
	} else {
	  output_file << lls_LS[i][j];
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


      output_file << "weights[[" << i+1 << "]] = c(";
      for (unsigned j=0; j<function_weights[i].size(); ++j) {
	if (j<function_weights[i].size()-1) {
	  output_file << function_weights[i][j] << ",";
	} else {
	  output_file << function_weights[i][j];
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

    output_file << "par(mar=c(2,2,2,2), mfrow=c(4,4));\n";
    for (i=0; i<N; ++i) {
      output_file << "plot(ts[[" << i+1 << "]], lls[[" << i+1 << "]], ylim=c(-30,10));\n";
      output_file << "lines(ts[[" << i+1 << "]], lls_small_t[[" << i+1 << "]]);\n";
      output_file << "lines(ts[[" << i+1 << "]], lls_LS[[" << i+1 << "]], col=2);\n";
      output_file << "lines(ts[[" << i+1 << "]], lls_matched[[" << i+1 << "]], col=3);\n";
      output_file << "abline(v = t_tilde_" << i << ");\n\n";
    }

    output_file.close();

    gsl_rng_free(r_ptr_local);
    return 0;
}
