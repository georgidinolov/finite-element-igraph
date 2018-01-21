#include "2DBrownianMotionPath.hpp"
#include <algorithm>
#include "BivariateSolver.hpp"
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


struct parameters_nominal {
  double sigma_2;
  //
  double phi;
  //
  int nu;
  double tau_2;
  std::vector<double> lower_triag_mat_as_vec = std::vector<double> (1.0, 28);

  parameters_nominal()
    : sigma_2(1.0),
      phi(1.0),
      nu(2),
      tau_2(1.0)
  {
    lower_triag_mat_as_vec = std::vector<double> (1.0, 28);
  };

  parameters_nominal(const std::vector<double> &x)
    : sigma_2(x[0]),
      phi(x[1]),
      nu(x[2]),
      tau_2(x[3]),
      lower_triag_mat_as_vec(std::vector<double>(1.0, 28))
  {
    for (unsigned i=0; i<28; ++i) {
      lower_triag_mat_as_vec[i] = x[i+4];
    }
  }

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

  likelihood_point()
    : x_0_tilde(0.5),
      y_0_tilde(0.5),
      x_t_tilde(0.5),
      y_t_tilde(0.5),
      sigma_y_tilde(1.0),
      t_tilde(1.0),
      rho(0.0),
      likelihood(1.0)
  {}

  likelihood_point& operator=(const likelihood_point& rhs)
  {
    if (this==&rhs) {
      return *this;
    } else {
      x_0_tilde = rhs.x_0_tilde;
      y_0_tilde = rhs.y_0_tilde;
      //
      x_t_tilde = rhs.x_t_tilde;
      y_t_tilde = rhs.y_t_tilde;
      //
      sigma_y_tilde = rhs.sigma_y_tilde;
      t_tilde = rhs.t_tilde;
      rho = rhs.rho;
      //
      likelihood = rhs.likelihood;

      return *this;
    }
  }

  likelihood_point& operator=(const BrownianMotion& BM)
  {
    double Lx = BM.get_b() - BM.get_a();
    double Ly = BM.get_d() - BM.get_c();

    double x_T = BM.get_x_T() - BM.get_a();
    double x_0 = BM.get_x_0() - BM.get_a();

    double y_T = BM.get_y_T() - BM.get_c();
    double y_0 = BM.get_y_0() - BM.get_c();
    double t = BM.get_t();
    // STEP 1
    double tau_x = BM.get_sigma_x()/Lx;
    double tau_y = BM.get_sigma_y()/Ly;

    x_t_tilde = x_T/Lx;
    y_t_tilde = y_T/Ly;
    x_0_tilde = x_0/Lx;
    y_0_tilde = y_0/Ly;
    sigma_y_tilde = tau_y/tau_x;
    t_tilde = std::pow(tau_x, 2);
	
    if (tau_x < tau_y) {
      x_t_tilde = y_T/Ly;
      y_t_tilde = x_T/Lx;
      //
      x_0_tilde = y_0/Ly;
      y_0_tilde = x_0/Lx;
      //
      sigma_y_tilde = tau_x/tau_y;
      //
      t_tilde = std::pow(tau_y, 2);
    }

    rho = BM.get_rho();
    likelihood = 0.0;
    
    return *this;
  }

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

  gsl_vector* as_gsl_vector() const
  {
    gsl_vector* out = gsl_vector_alloc(7);
    gsl_vector_set(out, 0, x_0_tilde);
    gsl_vector_set(out, 1, y_0_tilde);
    //
    gsl_vector_set(out, 2, x_t_tilde);
    gsl_vector_set(out, 3, y_t_tilde);
    //
    gsl_vector_set(out, 4, sigma_y_tilde);
    gsl_vector_set(out, 5, t_tilde);
    gsl_vector_set(out, 6, rho);

    return out;
  };
};

double likelihood_point_distance(const likelihood_point& lp1,
				 const likelihood_point& lp2,
				 const parameters_nominal& params)
{
  gsl_vector* lp1_vec = lp1.as_gsl_vector();
  gsl_vector* lp2_vec = lp2.as_gsl_vector();
  gsl_vector_sub(lp1_vec, lp2_vec);

  gsl_matrix* L = params.lower_triag_matrix();
  gsl_blas_dtrmv(CblasLower, CblasTrans, CblasNonUnit, L, lp1_vec);

  gsl_vector* squared_scaled_diff = gsl_vector_alloc(7);
  gsl_vector_memcpy(squared_scaled_diff, lp1_vec);
  gsl_vector_mul(squared_scaled_diff, lp1_vec);

  double out = 0;
  for (unsigned i=0; i<7; ++i) {
    out = out +
      gsl_vector_get(squared_scaled_diff, i);
  }

  out = std::sqrt(out);
  
  gsl_vector_free(lp1_vec);
  gsl_vector_free(lp2_vec);
  gsl_matrix_free(L);
  gsl_vector_free(squared_scaled_diff);
  return out;
};

double covariance(const likelihood_point& lp1,
		  const likelihood_point& lp2,
		  const parameters_nominal& params) {

  double dist = likelihood_point_distance(lp1,lp2,params);
  double out = params.tau_2 * std::pow(dist, params.nu) /
    (std::pow(2, params.nu-1) * gsl_sf_gamma(params.nu)) *
    gsl_sf_bessel_Kn(params.nu, dist);

  return out;
}

gsl_matrix* covariance_matrix(const std::vector<likelihood_point>& lps,
			      const parameters_nominal& params) {
  unsigned dimension = lps.size();
  gsl_matrix* out = gsl_matrix_alloc(dimension,dimension);

  for (unsigned i=0; i<dimension; ++i) {
    for (unsigned j=i; j<dimension; ++j) {
      double cov = 0;
      if (i==j) {
	cov = params.sigma_2;
	gsl_matrix_set(out, i,j, cov);
      } else {
	cov = covariance(lps[i], lps[j], params);
	gsl_matrix_set(out, i,j, cov);
	gsl_matrix_set(out, j,i, cov);
      }
    }
  }

  return out;
}

struct GaussianInterpolator {
  std::vector<likelihood_point> points_for_integration;
  std::vector<likelihood_point> points_for_interpolation;
  parameters_nominal parameters;
  gsl_matrix* C;
  gsl_matrix* Cinv;
  gsl_vector* y;
  gsl_vector* mean;
  gsl_vector* difference;
  gsl_vector* c;

  GaussianInterpolator()
    : points_for_integration(std::vector<likelihood_point> (1)),
      points_for_interpolation(std::vector<likelihood_point> (1)),
      parameters(parameters_nominal()),
      C(NULL),
      Cinv(NULL),
      y(NULL),
      mean(NULL),
      difference(NULL),
      c(NULL)
  {
    C = covariance_matrix(points_for_interpolation,
			  parameters);
    Cinv = gsl_matrix_alloc(points_for_interpolation.size(),
			    points_for_interpolation.size());
    gsl_matrix_memcpy(Cinv, C);
    gsl_linalg_cholesky_decomp(Cinv);
    gsl_linalg_cholesky_invert(Cinv);

    y = gsl_vector_alloc(points_for_interpolation.size());
    mean = gsl_vector_alloc(points_for_interpolation.size());
    difference = gsl_vector_alloc(points_for_interpolation.size());
    c = gsl_vector_alloc(points_for_interpolation.size());

    for (unsigned i=0; i<points_for_interpolation.size(); ++i) {
      gsl_vector_set(y, i, points_for_interpolation[i].likelihood);
      gsl_vector_set(mean, i, parameters.phi);
      gsl_vector_set(difference, i,
		     points_for_interpolation[i].likelihood-parameters.phi);
		     
    }
  }

  GaussianInterpolator(const std::vector<likelihood_point>& points_for_integration_in,
		       const std::vector<likelihood_point>& points_for_interpolation_in,
		       const parameters_nominal& params)
    : points_for_integration(points_for_integration_in),
      points_for_interpolation(points_for_interpolation_in),
      parameters(params),
      C(NULL),
      Cinv(NULL),
      y(NULL),
      mean(NULL),
      difference(NULL),
      c(NULL)
  {
    C = covariance_matrix(points_for_interpolation,
			  parameters);
    Cinv = gsl_matrix_alloc(points_for_interpolation.size(),
			    points_for_interpolation.size());
    gsl_matrix_memcpy(Cinv, C);
    gsl_linalg_cholesky_decomp(Cinv);
    gsl_linalg_cholesky_invert(Cinv);

    y = gsl_vector_alloc(points_for_interpolation.size());
    mean = gsl_vector_alloc(points_for_interpolation.size());
    c = gsl_vector_alloc(points_for_interpolation.size());
    difference = gsl_vector_alloc(points_for_interpolation.size());
    
    for (unsigned i=0; i<points_for_interpolation.size(); ++i) {
      gsl_vector_set(y, i, points_for_interpolation[i].likelihood);
      gsl_vector_set(mean, i, parameters.phi);
      gsl_vector_set(difference, i,
		     points_for_interpolation[i].likelihood - parameters.phi);
    }
  }

  GaussianInterpolator& operator=(const GaussianInterpolator& rhs)
  {
    if (this==&rhs) {
      return *this;
    } else {
      points_for_integration = rhs.points_for_integration;
      points_for_interpolation = rhs.points_for_interpolation;
      parameters = parameters;
      set_linalg_members();

      return *this;
    }
  }

  GaussianInterpolator(const GaussianInterpolator& rhs)
  {
    points_for_integration = rhs.points_for_integration;
    points_for_interpolation = rhs.points_for_interpolation;
    parameters = parameters;
    set_linalg_members();
  }

  ~GaussianInterpolator() {
    gsl_matrix_free(C);
    gsl_matrix_free(Cinv);
    gsl_vector_free(y);
    gsl_vector_free(mean);
    gsl_vector_free(difference);
    gsl_vector_free(c);
  }

  void set_linalg_members() {
    gsl_matrix_free(C);
    gsl_matrix_free(Cinv);
    gsl_vector_free(y);
    gsl_vector_free(mean);
    gsl_vector_free(difference);
    gsl_vector_free(c);

    C = covariance_matrix(points_for_interpolation,
			  parameters);
    Cinv = gsl_matrix_alloc(points_for_interpolation.size(),
			    points_for_interpolation.size());
    gsl_matrix_memcpy(Cinv, C);
    gsl_linalg_cholesky_decomp(Cinv);
    gsl_linalg_cholesky_invert(Cinv);

    y = gsl_vector_alloc(points_for_interpolation.size());
    mean = gsl_vector_alloc(points_for_interpolation.size());
    c = gsl_vector_alloc(points_for_interpolation.size());
    difference = gsl_vector_alloc(points_for_interpolation.size());

    for (unsigned i=0; i<points_for_interpolation.size(); ++i) {
      gsl_vector_set(y, i, points_for_interpolation[i].likelihood);
      gsl_vector_set(mean, i, parameters.phi);
      gsl_vector_set(difference, i,
		     points_for_interpolation[i].likelihood-parameters.phi);
    }
  }

  void add_point_for_integration(const likelihood_point& new_point) {
    points_for_integration.push_back(new_point);
  }

  void add_point_for_interpolation(const likelihood_point& new_point) {
    points_for_interpolation.push_back(new_point);
    set_linalg_members();
  }

  double operator()(const likelihood_point& x) {
    double out = 0;
    double work [points_for_interpolation.size()];
    gsl_vector_view work_view = gsl_vector_view_array(work,
						      points_for_interpolation.size());
    
    
    for (unsigned i=0; i<points_for_interpolation.size(); ++i) {
      gsl_vector_set(c, i, covariance(points_for_interpolation[i],
  				      x,
  				      parameters));
    }

    // y = alpha*op(A)*x + beta*y
    gsl_blas_dgemv(CblasTrans, // op(.) = ^T
		   1.0, // alpha = 1
		   Cinv, // A
		   c, // x
		   0.0, // beta = 0
		   &work_view.vector); // this is the output
    
    gsl_blas_ddot(&work_view.vector, difference, &out);

    out = out + parameters.phi;

    return out;
  }
};

double optimization_wrapper(const std::vector<double> &x,
			     std::vector<double> &grad,
			     void * data)
{
  std::vector<likelihood_point> * y_ptr=
    reinterpret_cast<std::vector<likelihood_point>*>(data);
  parameters_nominal params = parameters_nominal(x);

  unsigned N = (*y_ptr).size();
  gsl_vector* y = gsl_vector_alloc (N);
  gsl_vector* mu = gsl_vector_alloc(N);
  gsl_matrix* C = covariance_matrix(*y_ptr, params);

  for (unsigned i=0; i<N; ++i) {
    gsl_vector_set(y, i, (*y_ptr)[i].likelihood);
    gsl_vector_set(mu, i, params.phi);
  }

  MultivariateNormal mvtnorm = MultivariateNormal();
  double out = mvtnorm.dmvnorm(N,y,mu,C);

  gsl_vector_free(y);
  gsl_vector_free(mu);
  gsl_matrix_free(C);

  return out;
}

int main(int argc, char *argv[]) {
  if (argc < 5 || argc > 5) {
    printf("You must provide input\n");
    printf("The input is: \n\nnumber data points;\nrho;\ninput file for likelihoods;\nnumber_points_for_integration;\n");
    exit(0);
  }

  unsigned N = std::stoi(argv[1]);
  double rho = std::stod(argv[2]);
  std::string input_file_name = argv[3];
  int M = std::stoi(argv[4]);
  
  static int counter = 0;
  static gsl_rng* r_ptr_threadprivate;

#pragma omp threadprivate(counter, r_ptr_threadprivate)
  omp_set_dynamic(0);
  omp_set_num_threads(3);

  long unsigned seed_init = 10;
  gsl_rng * r_ptr_local;
  const gsl_rng_type * Type;
  gsl_rng_env_setup();
  Type = gsl_rng_default;
  r_ptr_local = gsl_rng_alloc(Type);
  gsl_rng_set(r_ptr_local, seed_init + N);
  
  int tid=0;
  unsigned i=0;
#pragma omp parallel default(none) private(tid, i) shared(r_ptr_local)
  {
    tid = omp_get_thread_num();

    r_ptr_threadprivate = gsl_rng_clone(r_ptr_local);
    gsl_rng_set(r_ptr_threadprivate, tid);

    printf("Thread %d: counter %d\n", tid, counter);
  }

  std::vector<likelihood_point> points_for_kriging (N);
  std::vector<likelihood_point> points_for_integration (M);

  auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(none) private(i) shared(points_for_integration, N, M, seed_init, rho)
  {
#pragma omp for
    for (i=0; i<M; ++i) {
      long unsigned seed = seed_init + N + i;

      double log_sigma_x = 1 + gsl_ran_gaussian(r_ptr_threadprivate, 1.0);
      double log_sigma_y = 1 + gsl_ran_gaussian(r_ptr_threadprivate, 1.0);

      BrownianMotion BM = BrownianMotion(seed,
					 10e6,
					 rho,
					 exp(log_sigma_x),
					 exp(log_sigma_y),
					 0.0,
					 0.0,
					 1.0);

      points_for_integration[i] = BM;
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();

  std::cout << "Generating " << M << " points for integration took "
	    << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << " seconds." << std::endl;

  std::ifstream input_file(input_file_name);

  if (input_file.is_open()) {
    for (i=0; i<N; ++i) {
      input_file >> points_for_kriging[i];
      points_for_kriging[i].likelihood = log(points_for_kriging[i].likelihood);
    }
  }

  parameters_nominal params = parameters_nominal();
  GaussianInterpolator GP_prior = GaussianInterpolator(points_for_integration,
						       points_for_kriging,
						       params);
  
  double integral = 0;
  for (unsigned i=0; i<points_for_integration.size(); ++i) {
    double add = GP_prior(points_for_integration[i]);
    integral = integral +
      1.0/M * add;
    if (add < 0) {
      std::cout << points_for_integration[i] << std::endl;
    }
  }
  std::cout << "Integral = " << integral << std::endl;
  
  return 0;
}
