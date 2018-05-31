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
  if (argc < 6 || argc > 6) {
    printf("You must provide input\n");
    printf("The input is: \n\nnumber data points;\nrho_basis;\nsigma_x;\nsigma_y;\ndx_likelihood;\n");
    exit(0);
  }

  unsigned N = std::stoi(argv[1]);
  double rho_basis = std::stod(argv[2]);
  double sigma_x_basis = std::stod(argv[3]);
  double sigma_y_basis = std::stod(argv[4]);
  double dx_likelihood = std::stod(argv[5]);
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
  points_for_kriging[0].y_0_tilde=3.00e-01;
  points_for_kriging[0].x_t_tilde=3.00e-01;
  points_for_kriging[0].y_t_tilde=5.00e-01;
  points_for_kriging[0].sigma_y_tilde=0.5;
  points_for_kriging[0].t_tilde=0.0581497;//0.0581497;
  points_for_kriging[0].rho=0.0;
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
	    solver.small_t_image_positions_type_41_all(true);

	  printf("## small t = %g\n", all_images[0].get_t());
	  double small_t = all_images[0].get_t();

	  gsl_matrix* cov_matrix = gsl_matrix_alloc(2,2);
	  gsl_matrix_set(cov_matrix, 0,0,
	  		 solver.get_sigma_x()*solver.get_sigma_x()*small_t);
	  gsl_matrix_set(cov_matrix, 0,1,
	  		 solver.get_rho()*solver.get_sigma_x()*solver.get_sigma_y()*small_t);
	  gsl_matrix_set(cov_matrix, 1,0,
	  		 solver.get_rho()*solver.get_sigma_x()*solver.get_sigma_y()*small_t);
	  gsl_matrix_set(cov_matrix, 1,1,
	  		 solver.get_sigma_y()*solver.get_sigma_y()*small_t);

	  MultivariateNormal mvtnorm = MultivariateNormal();
	  double out_small_t = 0;

	  for (const BivariateImageWithTime& image : all_images) {
	    out_small_t = out_small_t +
	      image.get_mult_factor()*mvtnorm.dmvnorm(2,
						      &raw_input.vector,
						      image.get_position(),
						      cov_matrix);
	  }
	  printf("## out_small_t = %g\n", out_small_t);
	  printf("## out_analytic = %g\n",
		 solver.analytic_solution(&raw_input.vector));

	  // BY COLUMN FOR R CONSUMPTION
	  unsigned M = 30;
	  double mmax = 0.999;
	  double mmin = 0.000;

	  std::vector<double> xs(M);
	  std::vector<double> ys(M);
	  double xx = 0.0;
	  double delta = mmax/M;
	  double yy = 0.0;
	  std::generate(xs.begin(),
			xs.end(),
			[&] ()->double {xx = xx + delta; return xx; });
	  std::generate(ys.begin(),
			ys.end(),
			[&] ()->double {yy = yy + delta; return yy; });

	  std::vector<std::vector<double>> small_t_solution_matrix =
	    std::vector<std::vector<double>> (M, std::vector<double>(M,0.0));
	  std::vector<std::vector<double>> analytic_solution_matrix =
	    std::vector<std::vector<double>> (M, std::vector<double>(M,0.0));

	  double max_diff = 0.0;
	  for (unsigned jj=0; jj<M; ++jj) {
	    for (unsigned ii=0; ii<M; ++ii) {
	      gsl_vector_set(&raw_input.vector,0,xs[jj]);
	      gsl_vector_set(&raw_input.vector,1,ys[ii]);

	      double out_small_t = 0;
	      for (const BivariateImageWithTime& image : all_images) {
		out_small_t = out_small_t +
		  image.get_mult_factor()*mvtnorm.dmvnorm(2,
							  &raw_input.vector,
							  image.get_position(),
							  cov_matrix);
	      }

	      small_t_solution_matrix[jj][ii] = out_small_t;

	      analytic_solution_matrix[jj][ii] =
		solver.analytic_solution(&raw_input.vector);

	      double rel_diff =
		std::abs(analytic_solution_matrix[jj][ii]-
			 out_small_t)/
		analytic_solution_matrix[jj][ii];

	      if (rel_diff > max_diff) {
		max_diff = rel_diff;
	      }
	    }
	  }
	  
	  printf("small.t.sol = c(");
	  for (std::vector<std::vector<double>>::iterator it=small_t_solution_matrix.begin();
	       it != small_t_solution_matrix.end(); ++it) {

	    if (it != small_t_solution_matrix.end()-1) {
	      for (double sol : *it) {
		printf("%g, ", sol);
	      }
	      printf("\n");

	    } else {
	      for (std::vector<double>::iterator y_it=(*it).begin(); y_it != (*it).end(); ++y_it) {
		if (y_it != (*it).end()-1) {
		  printf("%g, ", *y_it);
		} else {
		  printf("%g);\n", *y_it);
		}
	      }
	    }
	  }

	  printf("analytic.sol = c(");
	  for (std::vector<std::vector<double>>::iterator it=analytic_solution_matrix.begin();
	       it != analytic_solution_matrix.end(); ++it) {

	    if (it != analytic_solution_matrix.end()-1) {
	      for (double sol : *it) {
		printf("%g, ", sol);
	      }
	      printf("\n");

	    } else {
	      for (std::vector<double>::iterator y_it=(*it).begin(); y_it != (*it).end(); ++y_it) {
		if (y_it != (*it).end()-1) {
		  printf("%g, ", *y_it);
		} else {
		  printf("%g);\n", *y_it);
		}
	      }
	    }
	  }

	  printf("pdf(\"./src/kernel-expansion/documentation/chapter-3/chapter-3-figure-illustration-rel-error.pdf\", 6,6)\n");
	  printf("filled.contour(z=matrix(abs(small.t.sol-analytic.sol)/analytic.sol,%i,%i));\n", M, M);
	  printf("dev.off();\n");

	  printf("\n MAX DIFF = %g;\n", max_diff);

	  gsl_matrix_free(cov_matrix);
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
