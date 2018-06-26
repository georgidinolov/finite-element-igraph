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
    printf("The input is: \n\nrho_basis;\nsigma_x;\nsigma_y;\ndx_likelihood;\n");
    exit(0);
  }

  unsigned N = 1;
  double rho_basis = std::stod(argv[1]);
  double sigma_x_basis = std::stod(argv[2]);
  double sigma_y_basis = std::stod(argv[3]);
  double dx_likelihood = std::stod(argv[4]);
  double dx = 1.0/600.0;

  static int counter = 0;
  static BivariateGaussianKernelBasis* private_bases;
  static gsl_rng* r_ptr_threadprivate;

#pragma omp threadprivate(private_bases, counter, r_ptr_threadprivate)
  omp_set_dynamic(0);
  omp_set_num_threads(1);

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

  std::vector<likelihood_point> points_for_kriging (1);
  points_for_kriging[0].x_0_tilde=1.00e-01;
  points_for_kriging[0].y_0_tilde=1.00e-01;
  points_for_kriging[0].x_t_tilde=5.00e-01;
  points_for_kriging[0].y_t_tilde=5.00e-01;
  points_for_kriging[0].sigma_y_tilde=0.5;
  points_for_kriging[0].t_tilde=0.0581497;//0.0581497;
  points_for_kriging[0].rho=0.4;
  points_for_kriging[0].log_likelihood=-4.32267567159303e+00;
  points_for_kriging[0].FLIPPED=0;

  // points_for_kriging[1].x_0_tilde=6.50160070618248e-02;
  // points_for_kriging[1].y_0_tilde=7.61119804763752e-01;
  // points_for_kriging[1].x_t_tilde=1.85135596642142e-01;
  // points_for_kriging[1].y_t_tilde=3.19775518059827e-02;
  // points_for_kriging[1].sigma_y_tilde=2.24954509905047e-01;
  // points_for_kriging[1].t_tilde=9.06089050024474e+00;
  // points_for_kriging[1].rho=8.65650669386276e-01;
  // points_for_kriging[1].log_likelihood=-4.32267567159303e+00;
  // points_for_kriging[1].FLIPPED=0;

  std::vector<likelihood_point> points_for_integration (1);

  auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(none) private(i) shared(points_for_kriging, N, seed_init, r_ptr_local) firstprivate(dx_likelihood, dx)
    {
#pragma omp for
      for (i=0; i<N; ++i) {
	double raw_input_array [2] = {points_for_kriging[i].x_t_tilde,points_for_kriging[i].y_t_tilde};
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
							dx_likelihood);

	  printf("## out_small_t = %g\n", out_small_t);
	  printf("## out_analytic = %g\n",
		 solver.analytic_likelihood(&raw_input.vector, 1000));

	  // BY COLUMN FOR R CONSUMPTION
	  unsigned I = 50;
	  double t_max = 10.00;
	  double t_min = 0.01;

	  unsigned J = 1;
	  double sigma_max = points_for_kriging[i].sigma_y_tilde;
	  double sigma_min = points_for_kriging[i].sigma_y_tilde;

	  unsigned K = 1;
	  double rho_max = 0.0;
	  double rho_min = 0.0;

	  std::vector<double> ts(I);
	  std::vector<double> sigmas(J);
	  std::vector<double> rhos(K);

	  double delta_log_t = (log(t_max)-log(t_min))/I;
	  double delta_sigma = (sigma_max-sigma_min)/J;
	  double delta_rho = (rho_max-rho_min)/K;

	  double tt = exp(log(t_min) - delta_log_t);
	  double ss = sigma_min - delta_sigma;
	  double rr = rho_min - delta_rho;

	  std::generate(ts.begin(),
			ts.end(),
			[&] ()->double {tt = exp(log(tt) + delta_log_t); return tt; });
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
	  std::vector<std::vector<double>> modes =
	    std::vector<std::vector<double>> (J, std::vector<double> (K, 0));
	  std::vector<std::vector<double>> small_t_bounds =
	    std::vector<std::vector<double>> (J, std::vector<double> (K, 0));

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
	      
		if (!std::signbit(small_positions[0].get_t())) {
		  out_small_t =
		    solver.numerical_likelihood_first_order_small_t_ax_bx(&raw_input.vector,
									  dx_likelihood);
		  
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
		  std::vector<double>::iterator result =
		    std::min_element(current_modes.begin(), current_modes.end());
		  printf("min mode = %g\n", *result);

		  modes[jj][kk] = *result;
		  small_t_bounds[jj][kk] = small_positions[0].get_t();
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

	  printf("modes = c(");
	  for (unsigned jj=0; jj<J; ++jj) {
	    for (unsigned kk=0; kk<K; ++kk) {
	      if ( (jj==J-1) && (kk==K-1) ) {
		printf("%g);\n", modes[jj][kk]);
	      } else if ( (kk==K-1) ) {
		printf("%g,\n", modes[jj][kk]);
	      } else {
		printf("%g,", modes[jj][kk]);
	      }
	    }
	  }

	  printf("small.t.bounds = c(");
	  for (unsigned jj=0; jj<J; ++jj) {
	    for (unsigned kk=0; kk<K; ++kk) {
	      if ( (jj==J-1) && (kk==K-1) ) {
		printf("%g);\n", small_t_bounds[jj][kk]);
	      } else if ( (kk==K-1) ) {
		printf("%g,\n", small_t_bounds[jj][kk]);
	      } else {
		printf("%g,", small_t_bounds[jj][kk]);
	      }
	    }
	  }
	  
	  printf("z=matrix(small.t.sol,nrow=length(t),ncol=length(sigma), byrow=TRUE);\n");
	  printf("z.anal=matrix(analytic.sol,nrow=length(t),ncol=length(sigma), byrow=TRUE);\n");
	  printf("plot(log(t), z, type=\"l\", lwd=2); lines(log(t), z.anal, col=\"red\", lty=2);\n");
	  printf("abline(v=log(small.t.bounds));\n");
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

    gsl_rng_free(r_ptr_local);
    return 0;
}
