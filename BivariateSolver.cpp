#include <algorithm>
#include "BivariateSolver.hpp"
#include <gsl/gsl_blas.h>
#include <iostream>

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
    mvtnorm_(MultivariateNormal())
{
  if (x_0_ < 0.0 || x_0_ > 1.0 || y_0_ < 0.0 || y_0_ > 1.0) {
    std::cout << "ERROR: IC out of range" << std::endl;
  }
  xi_eta_input_ = gsl_vector_alloc(2);
  
  double cc = std::sin(M_PI/4.0);

  Rotation_matrix_ = gsl_matrix_alloc(2,2);
  gsl_matrix_set(Rotation_matrix_, 0, 0, cc / (sigma_x_*std::sqrt(1.0-rho_)));
  gsl_matrix_set(Rotation_matrix_, 1, 0, cc / (sigma_x_*std::sqrt(1.0+rho_)));
  gsl_matrix_set(Rotation_matrix_, 0, 1, -1.0*cc / (sigma_y_*std::sqrt(1-rho_)));
  gsl_matrix_set(Rotation_matrix_, 1, 1, cc / (sigma_y_*std::sqrt(1+rho_)));

  gsl_vector *initial_condition = gsl_vector_alloc(2);
  gsl_vector_set(initial_condition, 0, x_0_);
  gsl_vector_set(initial_condition, 1, y_0_);

  // rotating the initial condition
  initial_condition_xi_eta_ = gsl_vector_alloc(2);
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
  double tt = std::pow(Cs[Cs_indeces[1]]/4.0, 2.0);

  Variance_ = gsl_matrix_alloc(2, 2);
  for (int i=0; i<2; ++i) {
    for (int j=0; j<2; ++j) {
      if (i==j) {
  	gsl_matrix_set(Variance_, i, i, tt);
      } else {
  	gsl_matrix_set(Variance_, i, j, 0.0);
      }
    }
  }
  
  double Max = mvtnorm_.dmvnorm(2,
  				initial_condition_xi_eta_,
  				initial_condition_xi_eta_,
  				Variance_) /
    (sigma_x_*sigma_y_*sqrt(1.0-rho_)*sqrt(1.0+rho_));

  double xi_ic_reflected = 2.0*Cs[Cs_indeces[0]]*cos(ss_s[Cs_indeces[0]])
    + xi_ic;
  double eta_ic_reflected = 2.0*Cs[Cs_indeces[0]]*sin(ss_s[Cs_indeces[0]])
    + eta_ic;

  initial_condition_xi_eta_reflected_ = gsl_vector_alloc(2);
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
}

double BivariateSolverClassical::
operator()(const igraph_vector_t& input) const
{
  gsl_vector *gsl_input = gsl_vector_alloc(2);
  gsl_vector_set(gsl_input, 0, igraph_vector_e(&input, 0));
  gsl_vector_set(gsl_input, 1, igraph_vector_e(&input, 1));
  
  gsl_blas_dgemv(CblasNoTrans, 1.0,
		 Rotation_matrix_, gsl_input, 0.0,
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

  gsl_vector_free(gsl_input);
  return out;
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


double BivariateSolverClassical::norm() const
{
  return 0.0;
}

double BivariateSolverClassical::first_derivative(const igraph_vector_t& input,
						  long int coord_index) const
{
  return 0.0;
}
