#include "2DBrownianMotionPath.hpp"
#include <algorithm>
#include "BivariateSolver.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <iostream>
#include <limits>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <vector>

struct likelihood_point {
  double x_0_tilde;
  double y_0_tilde;
  //
  double x_t_tilde;
  double y_t_tilde;
  //
  double sigma_y_tilde;
  double t_tilde;
  double rho;
  //
  double likelihood;

  friend std::ostream& operator<<(std::ostream& os,
				  const likelihood_point& point)
  {
    os << "x_0_tilde = " << point.x_0_tilde << ";\n";
    os << "y_0_tilde = " << point.y_0_tilde << ";\n";
    //
    os << "x_t_tilde = " << point.x_t_tilde << ";\n";
    os << "y_t_tilde = " << point.y_t_tilde << ";\n";
    //
    os << "sigma_y_tilde = " << point.sigma_y_tilde << ";\n";
    os << "t_tilde = " << point.t_tilde << ";\n";
    os << "rho = " << point.rho << ";\n";
    //
    os << "likelihood = " << point.likelihood << ";";

    return os;
  }
};

int main(int argc, char *argv[]) {
  if (argc < 7 || argc > 7) {
    printf("You must provide input\n");
    printf("The input is: \n\nnumber data points;\nrho;\nrho_basis;\nsigma_x;\nsigma_y;\ndx_likelihood;\n");
    exit(0);
  }

  unsigned N = std::stoi(argv[1]);
  double rho = std::stod(argv[2]);
  double rho_basis = std::stod(argv[3]);
  double sigma_x_basis = std::stod(argv[4]);
  double sigma_y_basis = std::stod(argv[5]);
  double dx_likelihood = std::stod(argv[6]);
  double dx = 1.0/500.0;

  static int counter = 0;
  static BivariateGaussianKernelBasis* private_bases;

#pragma omp threadprivate(private_bases, counter)
  omp_set_dynamic(0);
  omp_set_num_threads(2);

  BivariateGaussianKernelBasis basis_positive =
    BivariateGaussianKernelBasis(dx,
				 rho_basis,
				 sigma_x_basis,
				 sigma_y_basis,
				 1.0,
				 1.0);
  int tid=0;
  unsigned i=0;
#pragma omp parallel default(none) private(tid, i) shared(basis_positive)
  {
    tid = omp_get_thread_num();

    private_bases = new BivariateGaussianKernelBasis();
    (*private_bases) = basis_positive;

    printf("Thread %d: counter %d\n", tid, counter);
  }

  long unsigned seed_init = 10;

  gsl_rng * r_ptr_local;
  const gsl_rng_type * Type;
  gsl_rng_env_setup();
  Type = gsl_rng_default;
  r_ptr_local = gsl_rng_alloc(Type);
  gsl_rng_set(r_ptr_local, seed_init + N);

  std::vector<likelihood_point> points_for_kriging (N);

  auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(none) private(i) shared(points_for_kriging, N, seed_init) firstprivate(dx_likelihood, dx)
    {
#pragma omp for
      for (i=0; i<N; ++i) {
	long unsigned seed = seed_init + i;

	double log_sigma_x = 1 + gsl_ran_gaussian(r_ptr_local, 1.0);
	double log_sigma_y = 1 + gsl_ran_gaussian(r_ptr_local, 1.0);

	BrownianMotion BM = BrownianMotion(seed,
					   10e6,
					   rho,
					   exp(log_sigma_x),
					   exp(log_sigma_y),
					   0.0,
					   0.0,
					   1.0);

        double Lx = BM.get_b() - BM.get_a();
	double Ly = BM.get_d() - BM.get_c();

	double x_T = BM.get_x_T() - BM.get_a();
	double x_0 = BM.get_x_0() - BM.get_a();

	double y_T = BM.get_y_T() - BM.get_c();
	double y_0 = BM.get_y_0() - BM.get_c();
	double t = BM.get_t();
	// STEP 1
	double tau_x = exp(log_sigma_x)/Lx;
	double tau_y = exp(log_sigma_y)/Ly;

	double x_T_tilde = x_T/Lx;
	double y_T_tilde = y_T/Ly;
	double x_0_tilde = x_0/Lx;
	double y_0_tilde = y_0/Ly;
	double sigma_y_tilde = tau_y/tau_x;
	double t_tilde = std::pow(tau_x/tau_y, 2);
	if (tau_x < tau_y) {
	  x_T_tilde = y_T/Ly;
	  y_T_tilde = x_T/Lx;
	  //
	  x_0_tilde = y_0/Ly;
	  y_0_tilde = x_0/Lx;
	  //
	  sigma_y_tilde = tau_x/tau_y;
	  //
	  t_tilde = std::pow(tau_y/tau_x, 2);
	}

	points_for_kriging[i].x_0_tilde = x_0_tilde;
	points_for_kriging[i].y_0_tilde = y_0_tilde;
	//
	points_for_kriging[i].x_t_tilde = x_T_tilde;
	points_for_kriging[i].y_t_tilde = y_T_tilde;
	//
	points_for_kriging[i].sigma_y_tilde = sigma_y_tilde;
	points_for_kriging[i].t_tilde = t_tilde;
	points_for_kriging[i].rho = rho;

	double likelihood = 0.0;
	double x [2] = {x_T_tilde, y_T_tilde};
	gsl_vector_view gsl_x = gsl_vector_view_array(x, 2);

	if (!std::signbit(rho)) {
	  BivariateSolver solver = BivariateSolver(private_bases,
						   1.0,
						   sigma_y_tilde,
						   rho,
						   0.0,
						   x_0_tilde,
						   1.0,
						   0.0,
						   y_0_tilde,
						   1.0,
						   t_tilde,
						   dx);
	  
	  x[0] = x_T_tilde;
	  x[1] = y_T_tilde;
	  
	  likelihood = solver.numerical_likelihood_extended(&gsl_x.vector,
							    dx_likelihood);
	} else {
	  BivariateSolver solver = BivariateSolver(private_bases,
						   1.0,
						   sigma_y_tilde,
						   rho,
						   -1.0,
						   -x_0_tilde,
						   0.0,
						   0.0,
						   y_0_tilde,
						   1.0,
						   t_tilde,
						   dx);

	  x[0] = -x_T_tilde;
	  x[1] = y_T_tilde;

	  likelihood = solver.numerical_likelihood_extended(&gsl_x.vector,
							    dx_likelihood);
	}

	points_for_kriging[i].likelihood = likelihood;

	printf("Thread %d with address %p produces likelihood %f\n",
	       omp_get_thread_num(),
	       private_bases,
	       likelihood);
      }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    for (i=0; i<N; ++i) {
      std::cout << points_for_kriging[i] << std::endl;
    }

    gsl_rng_free(r_ptr_local);
    return 0;
}
