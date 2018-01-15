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
#include <sstream>
#include <limits>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <vector>

struct parameters_nominal {
  double sigma_2;
  //
  double phi;
  //
  double nu;
  double tau_2;
  std::vector<double> lower_triag_mat_as_vec = std::vector<double> (1.0, 28);

  parameters_nominal()
    : sigma_2(1.0),
      phi(1.0),
      nu(2.0),
      tau_2(1.0)
  {
    lower_triag_mat_as_vec = std::vector<double> (1.0, 28);
  };

  parameters_nominal& operator=(const parameters_nominal& rhs)
  {
    if (this==&rhs) {
      return *this;
    } else {
      sigma_2 = rhs.sigma_2;
      phi = rhs.phi;
      nu = rhs.nu;
      tau_2 = rhs.tau_2;
      lower_triag_mat_as_vec = rhs.lower_triag_mat_as_vec;

      return *this;
    }
  }

  parameters_nominal(const parameters_nominal& rhs)
  {
    sigma_2 = rhs.sigma_2;
    phi = rhs.phi;
    nu = rhs.nu;
    tau_2 = rhs.tau_2;
    lower_triag_mat_as_vec = rhs.lower_triag_mat_as_vec;
  }

  std::vector<double> as_vector() const
  {
    std::vector<double> out = std::vector<double> (32);
    out[0] = sigma_2;
    out[1] = phi;
    out[2] = nu;
    out[3] = tau_2;

    for (unsigned i=4; i<32; ++i) {
      out[i] = lower_triag_mat_as_vec[i-4];
    }

    return out;
  }

  gsl_matrix* lower_triag_matrix() const
  {
    gsl_matrix* out = gsl_matrix_calloc(7,7);
    unsigned counter = 0;

    for (unsigned i=0; i<7; ++i) {
      for (unsigned j=0; j<i; ++j) {
	gsl_matrix_set(out, i, j,
		       lower_triag_mat_as_vec[counter]);
	counter = counter + 1;
      }
    }

    return out;
  }
    
};

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
    os << "x_0_tilde=" << point.x_0_tilde << ";\n";
    os << "y_0_tilde=" << point.y_0_tilde << ";\n";
    //
    os << "x_t_tilde=" << point.x_t_tilde << ";\n";
    os << "y_t_tilde=" << point.y_t_tilde << ";\n";
    //
    os << "sigma_y_tilde=" << point.sigma_y_tilde << ";\n";
    os << "t_tilde=" << point.t_tilde << ";\n";
    os << "rho=" << point.rho << ";\n";
    //
    os << "likelihood=" << point.likelihood << ";\n";

    return os;
  }

  friend std::istream& operator>>(std::istream& is,
				  likelihood_point& point)
  {
    std::string name;
    std::string value;
    // x_0_tilde
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.x_0_tilde = std::stod(value);

    // y_0_tilde
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.y_0_tilde = std::stod(value);

    // x_t_tilde
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.x_t_tilde = std::stod(value);

    // y_t_tilde
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.y_t_tilde = std::stod(value);

    // sigma_y_tilde
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.sigma_y_tilde = std::stod(value);

    // t_tilde
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.t_tilde = std::stod(value);

    // rho
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.rho = std::stod(value);

    // likelihood
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.likelihood = std::stod(value);

    return is;
  }
};

double likelihood_point_distance(const likelihood_point& lp1,
				 const likelihood_point& lp2,
				 const parameters_nominal& params)
{
  double out = 0;
  return out;
};

int main(int argc, char *argv[]) {
  if (argc < 8 || argc > 8) {
    printf("You must provide input\n");
    printf("The input is: \n\nnumber data points;\nrho;\nrho_basis;\nsigma_x;\nsigma_y;\ndx_likelihood;\nfile_prefix;\n");
    exit(0);
  }

  unsigned N = std::stoi(argv[1]);
  double rho = std::stod(argv[2]);
  double rho_basis = std::stod(argv[3]);
  double sigma_x_basis = std::stod(argv[4]);
  double sigma_y_basis = std::stod(argv[5]);
  double dx_likelihood = std::stod(argv[6]);
  std::string file_prefix = argv[7];
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
	  // likelihood = solver.analytic_likelihood(&gsl_x.vector, 100);
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
	  // likelihood = solver.analytic_likelihood(&gsl_x.vector, 100);
	}

	points_for_kriging[i].likelihood = likelihood;

	printf("Thread %d with address %p produces likelihood %f\n",
	       omp_get_thread_num(),
	       private_bases,
	       likelihood);
	std::cout << std::pow(Lx*Ly, 3) << std::endl;
      }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    std::string output_file_name = file_prefix +
      "-number-points-" + argv[1] +
      "-rho-" + argv[2] +
      "-rho_basis-" + argv[3] + 
      "-sigma_x_basis-" + argv[4] +
      "-sigma_y_basis-" + argv[5] +
      "-dx_likelihood-" + argv[6] +
      ".csv";

    std::ofstream output_file;
    output_file.open(output_file_name);
    
    for (i=0; i<N; ++i) {
      output_file << points_for_kriging[i];
    }
    output_file.close();

    std::ifstream data_ (output_file_name);
    likelihood_point lp;

    for (unsigned i=0; i<N; ++i) {
      data_ >> lp;
      std::cout << lp;
    }

    gsl_rng_free(r_ptr_local);
    return 0;
}
