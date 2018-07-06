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
    printf("The input is: \n\nrho_basis;\nsigma_x;\nsigma_y;\ndx_likelihood small t;\nfile for points;\nnumber data points;\ndx_likelihood for FEM\n");
    exit(0);
  }

  unsigned N = std::stoi(argv[6]);
  double rho_basis = std::stod(argv[1]);
  double sigma_x_basis = std::stod(argv[2]);
  double sigma_y_basis = std::stod(argv[3]);
  double dx_likelihood_for_small_t = std::stod(argv[4]);
  std::string input_file_name = argv[5];
  double dx_likelihood_for_FEM = std::stod(argv[7]);
  double dx = 1.0/1000.0;

  static int counter = 0;
  static BivariateGaussianKernelBasis* private_bases;
  static gsl_rng* r_ptr_threadprivate;

#pragma omp threadprivate(private_bases, counter, r_ptr_threadprivate)
  omp_set_dynamic(0);
  omp_set_num_threads(30);

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

  // READING IN DATA
  std::vector<double> differences_in_approximate_modes (N);
  std::vector<double> modes (N);
  std::vector<double> modes_analytic (N);
  std::vector<double> true_value_at_modes (N);
  std::vector<double> FE_value_at_modes (N);
  std::vector<double> small_t_value_at_modes (N);
  std::vector<double> dx_likelihoods_for_FEM (N);

  std::vector<likelihood_point> points_for_kriging (N);
  std::ifstream input_file(input_file_name);
  for (unsigned i=0; i<N; ++i) {
    likelihood_point current_lp = likelihood_point();
    input_file >> current_lp;
    std::cout << current_lp;
    // current_lp.sigma_y_tilde = 0.5;
    points_for_kriging[i] = current_lp;
  }
  std::vector<likelihood_point> points_for_integration (1);

  auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(none) private(i) shared(points_for_kriging, N, seed_init, r_ptr_local, differences_in_approximate_modes, true_value_at_modes, FE_value_at_modes, small_t_value_at_modes, modes, modes_analytic, dx_likelihoods_for_FEM) firstprivate(dx_likelihood_for_FEM, dx_likelihood_for_small_t, dx)
    {
#pragma omp for
      for (i=0; i<N; ++i) {
	double raw_input_array [2] = {points_for_kriging[i].x_t_tilde,
				      points_for_kriging[i].y_t_tilde};
	gsl_vector_view raw_input = gsl_vector_view_array(raw_input_array,2);

	long unsigned seed = seed_init + i;
	double likelihood = 0.0;
	double rho = points_for_kriging[i].rho;
	double x [2] = {points_for_kriging[i].x_t_tilde,
			points_for_kriging[i].y_t_tilde};

	gsl_vector_view gsl_x = gsl_vector_view_array(x, 2);

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

	  x[0] = points_for_kriging[i].x_t_tilde;
	  x[1] = points_for_kriging[i].y_t_tilde;

	  std::vector<BivariateImageWithTime> all_images =
	    solver.small_t_image_positions_type_41_all(false);

	  printf("## small t = %g\n", all_images[0].get_t());
	  double small_t = all_images[0].get_t();

	  double out_small_t =
	    solver.likelihood_small_t_type_41_truncated(&raw_input.vector,
							small_t,
							dx_likelihood_for_small_t);

	  printf("## out_small_t = %g\n", out_small_t);
	  printf("## out_analytic = %g\n",
		 solver.analytic_likelihood(&raw_input.vector, 1000));

	  // BY COLUMN FOR R CONSUMPTION
	  unsigned I = 1;
	  double t_max = 1.00;
	  double t_min = 1.00;

	  unsigned J = 1;
	  double sigma_max = points_for_kriging[i].sigma_y_tilde*1.0;
	  double sigma_min = points_for_kriging[i].sigma_y_tilde*1.0;

	  unsigned K = 1;
	  double rho_max = 0.0;
	  double rho_min = 0.0;

	  std::vector<double> ts(I);
	  std::vector<double> sigmas(J);
	  std::vector<double> rhos(K);

	  double delta_t = (t_max-t_min)/I;
	  double delta_sigma = (sigma_max-sigma_min)/J;
	  double delta_rho = (rho_max-rho_min)/K;

	  double tt = t_min - delta_t;
	  double ss = sigma_min - delta_sigma;
	  double rr = rho_min - delta_rho;

	  std::generate(ts.begin(),
			ts.end(),
			[&] ()->double {tt = tt + delta_t; return tt; });
	  std::generate(sigmas.begin(),
			sigmas.end(),
			[&] ()->double {ss = ss + delta_sigma; return ss; });
	  std::generate(rhos.begin(),
			rhos.end(),
			[&] ()->double {rr = rr + delta_rho; return rr; });

	  std::vector<std::vector<std::vector<double>>> small_t_likelihood_matrix =
	    std::vector<std::vector<std::vector<double>>> (I, std::vector<std::vector<double>>(J, std::vector<double> (K, 0.0)));
	  std::vector<std::vector<std::vector<double>>> analytic_likelihood_matrix = 
	    std::vector<std::vector<std::vector<double>>> (I, std::vector<std::vector<double>>(J, std::vector<double> (K, 0.0)));

	  for (unsigned ii=0; ii<I; ++ii) {
	    for (unsigned jj=0; jj<J; ++jj) {
	      for (unsigned kk=0; kk<K; ++kk) {
		printf("rho=%g, sigma=%g\n ", rhos[kk], sigmas[jj]);

		solver.set_diffusion_parameters_and_data(1.0,
							 sigmas[jj],
							 rhos[kk],
							 ts[ii],
							 0.0,
							 points_for_kriging[i].x_0_tilde,
							 1.0,
							 0.0,
							 points_for_kriging[i].y_0_tilde,
							 1.0);

		std::vector<BivariateImageWithTime> small_positions =
		  solver.small_t_image_positions_1_3(false);
		double out_small_t = std::numeric_limits<double>::quiet_NaN();

		std::vector<double> lb (1);
		lb[0] = 0.0001;

		std::vector<double> ub (1);
		ub[0] = HUGE_VAL;

		nlopt::opt opt(nlopt::LN_NELDERMEAD, 1);
		opt.set_lower_bounds(lb);
		opt.set_upper_bounds(ub);
		opt.set_ftol_rel(0.0001);
		double max_f = 0.0;
		std::vector<likelihood_point> current_lps {points_for_kriging[i]};
		std::vector<double> optimal_params (1);
		optimal_params[0] = 1.0;

		solver.set_x_t_2(gsl_vector_get(&raw_input.vector,0));
		solver.set_y_t_2(gsl_vector_get(&raw_input.vector,1));

		
		opt.set_max_objective(BivariateSolver::wrapper,
				      &solver);
		
		
		if (ii==0) {
		  opt.optimize(optimal_params, max_f);
		  modes_analytic[i] = optimal_params[0];
		  printf("ANALYTIC MODE = %.32g\n",
			 optimal_params[0]);
		}
	      
		if (!std::signbit(small_positions[0].get_t())) {
		  out_small_t =
		    solver.numerical_likelihood_first_order_small_t_ax_bx(&raw_input.vector,
									  dx_likelihood_for_small_t);
		  
		  printf("POSITIVE t=%g, sigma=%g, rho=%g, out_small_t=%g, max_admissible_time=%g\n",
			 ts[ii],
			 sigmas[jj],
			 rhos[kk],
			 out_small_t,
			 small_positions[0].get_t());
		  printf("modes = ");
		  std::vector<double> current_modes (small_positions.size());
		  for (unsigned ii=0; ii<small_positions.size(); ++ii) {
		    const BivariateImageWithTime& image = small_positions[ii];
		    double x = gsl_vector_get(&raw_input.vector, 0);
		    double y = gsl_vector_get(&raw_input.vector, 1);
		    double x_0 = gsl_vector_get(image.get_position(), 0);
		    double y_0 = gsl_vector_get(image.get_position(), 1);
		  
		    double sigma = sigmas[jj];
		    double rho = rhos[kk];
		    double t = ts[ii];

		    double beta =
		      ( std::pow(x-x_0,2)*std::pow(sigma,2) +
			std::pow(y-y_0,2) -
			2*rho*(x-x_0)*(y-y_0) )/(2.0*std::pow(sigma,2) * (1-rho*rho));
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
		  printf("min mode = %g\n", *result_min);

		  if (ii==0) {
		    modes[i] = (*result_min);
		    differences_in_approximate_modes[i] =
		      modes_analytic[i] - modes[i];

		    solver.set_diffusion_parameters_and_data(1.0,
							     sigmas[jj],
							     rhos[kk],
							     modes[i],
							     0.0,
							     points_for_kriging[i].x_0_tilde,
							     1.0,
							     0.0,
							     points_for_kriging[i].y_0_tilde,
							     1.0);

		    double true_at_mode = solver.analytic_likelihood(&raw_input.vector, 1000);
		    true_value_at_modes[i] = true_at_mode;
		    // O(solution)/100 = epsilon/ (dx^4)
		    // ==> dx^4 = epsilon/ (O(solution)/100) = 100*epsilon/O(solution)
		    // ==> 4*log(dx) = log(100) + log(epsilon) - log(O(solution))
		    // ==> log(dx) = ( log(100) + log(epsilon) - log(O(solution)) ) / 4
		    // ==> dx = exp( (log(100) + log(epsilon) - log(O(solution))) / 4 )
		    // epsilon is assumed 1e-12
		    double epsilon = 1e-12;
		    std::vector<double> dx_proposals = std::vector<double> {exp( (log(100) + log(epsilon) - true_at_mode) / 4 ),
									    dx_likelihood_for_FEM};
		    std::vector<double>::iterator dx_proposal_result = std::min_element(dx_proposals.begin(),
											dx_proposals.end());
		    dx_likelihoods_for_FEM[i] = *dx_proposal_result;
		    double FE_at_mode = solver.numerical_likelihood_second_order(&raw_input.vector,
										 dx_likelihood_for_FEM);
		    FE_value_at_modes[i] = FE_at_mode;

		    small_t_value_at_modes[i] =
		      solver.numerical_likelihood_first_order_small_t_ax_bx(&raw_input.vector,
									  dx_likelihood_for_small_t);
		  }
		}
	      
		small_t_likelihood_matrix[ii][jj][kk] = out_small_t;

		analytic_likelihood_matrix[ii][jj][kk] = 
		  solver.analytic_likelihood(&raw_input.vector, 1000);

	      }
	    }
	  }
	  
	  printf("t = c(");
	  for (std::vector<double>::iterator it=ts.begin(); it != ts.end(); ++it) {
	    if (it != ts.end()-1) {
	      printf("%g, ", *it);
	    } else {
	      printf("%g);\n", *it);
	    }
	  }

	  printf("rho = c(");
	  for (std::vector<double>::iterator it=rhos.begin(); it != rhos.end(); ++it) {
	    if (it != rhos.end()-1) {
	      printf("%g, ", *it);
	    } else {
	      printf("%g);\n", *it);
	    }
	  }

	  printf("sigma = c(");
	  for (std::vector<double>::iterator it=sigmas.begin(); it != sigmas.end(); ++it) {
	    if (it != sigmas.end()-1) {
	      printf("%g, ", *it);
	    } else {
	      printf("%g);\n", *it);
	    }
	  }
	  
	  printf("small.t.sol = c(");
	  for (unsigned ii=0; ii<I; ++ii) {
	    for (unsigned jj=0; jj<J; ++jj) {
	      for (unsigned kk=0; kk<K; ++kk) {
		if ( (ii==I-1) && (jj==J-1) && (kk==K-1) ) {
		  printf("%g);\n", small_t_likelihood_matrix[ii][jj][kk]);
		} else if ( (kk==K-1) ) {
		  printf("%g,\n", small_t_likelihood_matrix[ii][jj][kk]);
		} else {
		  printf("%g,", small_t_likelihood_matrix[ii][jj][kk]);
		}
	      }
	    }
	  }
	  	  
	  printf("analytic.sol = c(");
	  for (unsigned ii=0; ii<I; ++ii) {
	    for (unsigned jj=0; jj<J; ++jj) {
	      for (unsigned kk=0; kk<K; ++kk) {
		if ( (ii==I-1) && (jj==J-1) && (kk==K-1) ) {
		  printf("%g);\n", analytic_likelihood_matrix[ii][jj][kk]);
		} else if ( (kk==K-1) ) {
		  printf("%g,\n", analytic_likelihood_matrix[ii][jj][kk]);
		} else {
		  printf("%g,", analytic_likelihood_matrix[ii][jj][kk]);
		}
	      }
	    }
	  }

	  printf("z=matrix(small.t.sol,nrow=length(t),ncol=length(sigma), byrow=TRUE);\n");
	  printf("z.anal=matrix(analytic.sol,nrow=length(t),ncol=length(sigma), byrow=TRUE);\n");
	  printf("filled.contour(x=t, y=sigma, z, xlab=\"t\", ylab=\"sigma\");\n");
	  
	  printf("pdf(\"./src/kernel-expansion/documentation/chapter-3/chapter-3-figure-validation-1-analytic.pdf\", 6,6)\n");
	  printf("## filled.contour(x=sigma, y=t, z=matrix(analytic.sol,%i,%i), xlab=\"sigma\", ylab=\"t\");\n", J, I);
	  printf("dev.off();\n");

	  printf("pdf(\"./src/kernel-expansion/documentation/chapter-3/chapter-3-figure-validation-1-small-t-symmetric.pdf\", 6,6)\n");
	  printf("## filled.contour(x=sigma, y=t, z=matrix(small.t.sol,%i,%i), xlab=\"sigma\", ylab=\"t\");\n", J, I);
	  printf("dev.off();\n");

	}

	points_for_kriging[i].log_likelihood = log(likelihood);
	printf("Thread %d with address %p produces likelihood %g\n",
	       omp_get_thread_num(),
	       private_bases,
	       likelihood);
      }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    std::ofstream output_file;
    output_file.open("./src/kernel-expansion/documentation/chapter-3/chapter-3-figure-validation-modes.R");

    output_file << "modes.analytic = c(";
    for (unsigned i=0; i<N-1; ++i) {
      output_file << modes_analytic[i] << ",\n";
    }
    output_file << modes_analytic[N-1] << ");\n";

    output_file << "modes = c(";
    for (unsigned i=0; i<N-1; ++i) {
      output_file << modes[i] << ",\n";
    }
    output_file << modes[N-1] << ");\n";

    output_file << "differences = c(";
    for (unsigned i=0; i<differences_in_approximate_modes.size()-1; ++i) {
      output_file << differences_in_approximate_modes[i] << ",\n";
    }
    output_file << differences_in_approximate_modes[differences_in_approximate_modes.size()-1] << ");" 
	      << "\n";

    output_file << "sigma_ys = c(";
    for (unsigned i=0; i<points_for_kriging.size()-1; ++i) {
      output_file << points_for_kriging[i].sigma_y_tilde << ",\n";
    }
    output_file << points_for_kriging[points_for_kriging.size()-1].sigma_y_tilde << ");" 
	      << "\n";

    output_file << "true_value_at_modes = c(";
    for (unsigned i=0; i<true_value_at_modes.size()-1; ++i) {
      output_file << true_value_at_modes[i] << ",\n";
    }
    output_file << true_value_at_modes[true_value_at_modes.size()-1] << ");" 
	      << "\n";

    output_file << "FE_value_at_modes = c(";
    for (unsigned i=0; i<FE_value_at_modes.size()-1; ++i) {
      output_file << FE_value_at_modes[i] << ",\n";
    }
    output_file << FE_value_at_modes[FE_value_at_modes.size()-1] << ");" 
	      << "\n";

    output_file << "small_t_value_at_modes = c(";
    for (unsigned i=0; i<small_t_value_at_modes.size()-1; ++i) {
      output_file << small_t_value_at_modes[i] << ",\n";
    }
    output_file << small_t_value_at_modes[small_t_value_at_modes.size()-1] << ");" 
		<< "\n";

    output_file << "dx_likelihoods_for_FEM = c(";
    for (unsigned i=0; i<dx_likelihoods_for_FEM.size()-1; ++i) {
      output_file << dx_likelihoods_for_FEM[i] << ",\n";
    }
    output_file << dx_likelihoods_for_FEM[dx_likelihoods_for_FEM.size()-1] << ");" 
		<< "\n";

    output_file << "pdf(\"./src/kernel-expansion/documentation/chapter-3/chapter-3-figure-validation-modes-scatterplot.pdf\");\n";
    output_file << "plot(sigma_ys, differences, xlab=expression(tilde(sigma)));\n";
    output_file << "dev.off();\n";

    output_file << "pdf(\"./src/kernel-expansion/documentation/chapter-3/chapter-3-figure-validation-modes-histogram.pdf\");\n";
    output_file << "hist(differences, xlab=\"differences\", prob=1, main=\"\", ylab=\"\");\n";
    output_file << "dev.off();\n";

    output_file << "pdf(\"./src/kernel-expansion/documentation/chapter-3/chapter-3-figure-validation-modes-scatterplot-2.pdf\");\n";
    output_file << "plot(modes.analytic, differences, xlab=\"true modes\", ylab=\"differences\");\n";
    output_file << "dev.off();\n";

    output_file << "pdf(\"./src/kernel-expansion/documentation/chapter-3/chapter-3-figure-validation-modes-histogram-2.pdf\");\n";
    output_file << "hist(true_value_at_modes, xlab=\"true log likelihood at approximate mode\", prob=1, main=\"\", ylab=\"\");\n";
    output_file << "dev.off();\n";

    output_file << "pdf(\"./src/kernel-expansion/documentation/chapter-3/chapter-3-figure-validation-modes-histogram-3.pdf\");\n";
    output_file << "hist(abs(FE_value_at_modes-exp(true_value_at_modes))/exp(true_value_at_modes), xlab=\"relative error at approximate modes\", prob=1, main=\"\", ylab=\"\");\n";
    output_file << "dev.off();\n";

    output_file.close();
    gsl_rng_free(r_ptr_local);
    return 0;
}
