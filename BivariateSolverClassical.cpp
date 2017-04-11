#include <algorithm>
#include "BasisElementTypes.hpp"
#include <chrono>
#include <gsl/gsl_blas.h>
#include <iostream>

BivariateSolverClassical::BivariateSolverClassical()
  : sigma_x_(1.0),
    sigma_y_(1.0),
    rho_(0.0),
    x_0_(0.5),
    y_0_(0.5),
    mvtnorm_(MultivariateNormal()),
    xi_eta_input_(gsl_vector_alloc(2)),
    initial_condition_xi_eta_(gsl_vector_alloc(2)),
    Rotation_matrix_(gsl_matrix_alloc(2,2)),
    tt_(0),
    Variance_(gsl_matrix_alloc(2,2)),
    initial_condition_xi_eta_reflected_(gsl_vector_alloc(2)),
    function_grid_(gsl_matrix_alloc(1,1))
{}

BivariateSolverClassical::BivariateSolverClassical(double sigma_x,
						   double sigma_y,
						   double rho,
						   double x_0,
						   double y_0)
  : sigma_x_(sigma_x),
    sigma_y_(sigma_y),
    rho_(rho),
    x_0_(x_0),
    y_0_(y_0),
    mvtnorm_(MultivariateNormal()),
    xi_eta_input_(gsl_vector_alloc(2)),
    initial_condition_xi_eta_(gsl_vector_alloc(2)),
    Rotation_matrix_(gsl_matrix_alloc(2,2)),
    tt_(0),
    Variance_(gsl_matrix_alloc(2,2)),
    initial_condition_xi_eta_reflected_(gsl_vector_alloc(2)),
    function_grid_(gsl_matrix_alloc(1,1))
{
  if (x_0_ < 0.0 || x_0_ > 1.0 || y_0_ < 0.0 || y_0_ > 1.0) {
    std::cout << "ERROR: IC out of range" << std::endl;
  }
  std::cout << "sigma_x_ = " << sigma_x_ << "\n"
	    << "sigma_y_ = " << sigma_y_ << "\n"
	    << "rho_ = " << rho_ << std::endl;
  
  double cc = std::sin(M_PI/4.0);

  gsl_matrix_set(Rotation_matrix_, 0, 0, cc / (sigma_x_*std::sqrt(1.0-rho_)));
  gsl_matrix_set(Rotation_matrix_, 1, 0, cc / (sigma_x_*std::sqrt(1.0+rho_)));
  gsl_matrix_set(Rotation_matrix_, 0, 1, -1.0*cc / (sigma_y_*std::sqrt(1-rho_)));
  gsl_matrix_set(Rotation_matrix_, 1, 1, cc / (sigma_y_*std::sqrt(1+rho_)));

  gsl_vector *initial_condition = gsl_vector_alloc(2);
  gsl_vector_set(initial_condition, 0, x_0_);
  gsl_vector_set(initial_condition, 1, y_0_);

  // rotating the initial condition
  gsl_blas_dgemv(CblasNoTrans, 1.0,
		 Rotation_matrix_, initial_condition, 0.0,
		 initial_condition_xi_eta_);
  double xi_ic = gsl_vector_get(initial_condition_xi_eta_, 0);
  double eta_ic = gsl_vector_get(initial_condition_xi_eta_, 1);

  // gsl_vector *slopes = gsl_vector_alloc(4);
  // gsl_vector_set(slopes, 0, std::sqrt(1-rho_)/std::sqrt(1+rho_));
  // gsl_vector_set(slopes, 1, std::sqrt(1-rho_)/std::sqrt(1+rho_));
  // gsl_vector_set(slopes, 2, -1.0*std::sqrt(1-rho_)/std::sqrt(1+rho_));
  // gsl_vector_set(slopes, 3, -1.0*std::sqrt(1-rho_)/std::sqrt(1+rho_));

  // BORDER 1
  double ss_1 = std::atan(-1.0*std::sqrt(1.0+rho_)/
			  std::sqrt(1-rho_));
  double C_1 = (-1.0*gsl_vector_get(initial_condition_xi_eta_,1)
		+ xi_ic*
		std::sqrt(1.0-rho_)/std::sqrt(1.0+rho_))/
    (std::sin(ss_1) - std::cos(ss_1)*std::sqrt(1.0-rho_)/std::sqrt(1.0+rho_));

  // BORDER 2
  double ss_2 = M_PI + ss_1;
  double C_2 = (-1.0*eta_ic +
		xi_ic*std::sqrt(1-rho_)/std::sqrt(1+rho_) +
		 1.0/(sigma_y_*std::sqrt(1.0+rho_)*cc))/
	      (std::sin(ss_2) - std::cos(ss_2)*std::sqrt(1-rho_)/
	       std::sqrt(1+rho_));

  // BORDER 4
  double ss_4 = atan(sqrt(1.0+rho_)/sqrt(1.0-rho_));
  double C_4 = (-1.0*eta_ic - xi_ic*sqrt(1.0-rho_)/sqrt(1.0+rho_) +
		1/(sigma_x_*sqrt(1.0+rho_)*cc))/
        (sin(ss_4) + cos(ss_4)*sqrt(1.0-rho_)/sqrt(1.0+rho_));

  // BORDER 3
  double ss_3 = M_PI + ss_4;
  double C_3 = (-1.0*eta_ic - xi_ic*sqrt(1.0-rho_)/sqrt(1.0+rho_))/
    (sin(ss_3) + cos(ss_3)*sqrt(1.0-rho_)/sqrt(1.0+rho_));


  std::vector<double> Cs = std::vector<double> {C_1,
						C_2,
						C_3,
						C_4};

  std::vector<double> ss_s = std::vector<double> {ss_1,
						  ss_2,
						  ss_3,
						  ss_4};
  
  std::vector<unsigned> Cs_indeces (Cs.size());
  unsigned n = 0;
  std::generate(Cs_indeces.begin(), Cs_indeces.end(), [&n]{ return n++; });

  std::sort(Cs_indeces.begin(), Cs_indeces.end(),
  	    [&Cs] (unsigned i1, unsigned i2) -> bool
  	    {
  	      return Cs[i1] < Cs[i2];
  	    });
  tt_ = std::pow(Cs[Cs_indeces[1]]/4.0, 2.0);

  for (int i=0; i<2; ++i) {
    for (int j=0; j<2; ++j) {
      if (i==j) {
  	gsl_matrix_set(Variance_, i, i, tt_);
      } else {
  	gsl_matrix_set(Variance_, i, j, 0.0);
      }
    }
  }
  
  double xi_ic_reflected = 2.0*Cs[Cs_indeces[0]]*cos(ss_s[Cs_indeces[0]])
    + xi_ic;
  double eta_ic_reflected = 2.0*Cs[Cs_indeces[0]]*sin(ss_s[Cs_indeces[0]])
    + eta_ic;

  gsl_vector_set(initial_condition_xi_eta_reflected_, 0, xi_ic_reflected);
  gsl_vector_set(initial_condition_xi_eta_reflected_, 1, eta_ic_reflected);

  gsl_vector_free(initial_condition);
  // gsl_vector_free(slopes);
}

BivariateSolverClassical::~BivariateSolverClassical()
{
  // freeing vectors
  gsl_vector_free(xi_eta_input_);
  gsl_vector_free(initial_condition_xi_eta_);
  gsl_vector_free(initial_condition_xi_eta_reflected_);

  // freeing matrices
  gsl_matrix_free(Rotation_matrix_);
  gsl_matrix_free(Variance_);
  gsl_matrix_free(function_grid_);
}

double BivariateSolverClassical::
operator()(const gsl_vector* input) const
{
  gsl_blas_dgemv(CblasNoTrans, 1.0,
		 Rotation_matrix_, input, 0.0,
		 xi_eta_input_);

  double out = (mvtnorm_.dmvnorm(2,
				 xi_eta_input_,
				 initial_condition_xi_eta_,
				 Variance_) -
		mvtnorm_.dmvnorm(2,
				 xi_eta_input_,
				 initial_condition_xi_eta_reflected_,
				 Variance_)) /
		(sigma_x_*sigma_y_*sqrt(1-rho_)*sqrt(1+rho_));
  return out;
}

double BivariateSolverClassical::
operator()(const gsl_vector* input, double tt) const
{
  gsl_blas_dgemv(CblasNoTrans, 1.0,
		 Rotation_matrix_, input, 0.0,
		 xi_eta_input_);

  gsl_matrix* Variance = gsl_matrix_alloc(2,2);
  gsl_matrix_set_all(Variance, 0.0);
  
  for (int i=0; i<2; ++i) {
    gsl_matrix_set(Variance, i, i, tt);
  }
  
  double out = (mvtnorm_.dmvnorm(2,
				 xi_eta_input_,
				 initial_condition_xi_eta_,
				 Variance) -
		mvtnorm_.dmvnorm(2,
				 xi_eta_input_,
				 initial_condition_xi_eta_reflected_,
				 Variance)) /
		(sigma_x_*sigma_y_*sqrt(1-rho_)*sqrt(1+rho_));
  
  gsl_matrix_free(Variance);
  return out;
}


double BivariateSolverClassical::norm() const
{
  return 0.0;
}

double BivariateSolverClassical::first_derivative(const gsl_vector* input,
						  long int coord_index) const
{
  return 0.0;
}

double BivariateSolverClassical::get_t() const
{
  return tt_;
}

const gsl_matrix* BivariateSolverClassical::get_function_grid() const
{
  return function_grid_;
}

void BivariateSolverClassical::set_function_grid(double dx)
{
  gsl_matrix_free(function_grid_);
  function_grid_ = gsl_matrix_alloc(1/dx + 1, 1/dx + 1);

  auto t1 = std::chrono::high_resolution_clock::now();
  double out = 0;
  gsl_vector * input = gsl_vector_alloc(2);
  double x = 0;
  double y = 0;

  for (int i=0; i<1/dx + 1; ++i) {
    x = i*dx;
    gsl_vector_set(input, 0, x);

    for (int j=0; j<1/dx + 1; ++j) {
      y = j*dx;
      gsl_vector_set(input, 1, y);

      out = (*this)(input);
      gsl_matrix_set(function_grid_, i, j, out);
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "duration in Bivariate Classical Solver = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
    	    << " milliseconds\n";

  gsl_vector_free(input);
}
