#include <algorithm>
#include "BivariateSolver.hpp"
#include <cmath>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <iostream>
#include <limits>
#include <string>

BivariateSolver::BivariateSolver()
  : a_(0),
    b_(1),
    c_(0),
    d_(1),
    sigma_x_(1.0),
    sigma_y_(1.0),
    rho_(0.0),
    x_0_(0.5),
    y_0_(0.5),
    mvtnorm_(MultivariateNormal()),
    basis_(NULL),
    small_t_solution_(new BivariateSolverClassical()),
    t_(1),
    dx_(0.01),
    IC_coefs_(gsl_vector_alloc(1)),
    mass_matrix_(gsl_matrix_alloc(1,1)),
    stiffness_matrix_(gsl_matrix_alloc(1,1)),
    eval_(gsl_vector_alloc(1)),
    evec_(gsl_matrix_alloc(1,1)),
    solution_coefs_(gsl_vector_alloc(1))
{}

BivariateSolver::BivariateSolver(BivariateBasis* basis)
  : a_(0),
    b_(1),
    c_(0),
    d_(1),
    sigma_x_(1.0),
    sigma_y_(1.0),
    rho_(0.0),
    x_0_(0.5),
    y_0_(0.5),
    mvtnorm_(MultivariateNormal()),
    basis_(basis),
    small_t_solution_(NULL),
    t_(1),
    dx_(basis->get_dx()),
    IC_coefs_(gsl_vector_alloc(basis_->get_orthonormal_elements().size())),
    mass_matrix_(gsl_matrix_alloc(basis_->get_orthonormal_elements().size(),
				  basis_->get_orthonormal_elements().size())),
    stiffness_matrix_(gsl_matrix_alloc(basis_->get_orthonormal_elements().size(),
				       basis_->get_orthonormal_elements().size())),
    eval_(gsl_vector_alloc(basis_->get_orthonormal_elements().size())),
    evec_(gsl_matrix_alloc(basis_->get_orthonormal_elements().size(),
			   basis_->get_orthonormal_elements().size())),
    solution_coefs_(gsl_vector_alloc(basis->get_orthonormal_elements().size()))
{
  // printf("basis address = %p\n", basis_);
  set_scaled_data();

  small_t_solution_ = new BivariateSolverClassical(sigma_x_2_,
  						   sigma_y_2_,
  						   rho_,
  						   x_0_2_,
  						   y_0_2_);
  small_t_solution_->set_function_grid(dx_);

  set_mass_and_stiffness_matrices();
  set_eval_and_evec();
  set_IC_coefs();
  set_solution_coefs();
}

BivariateSolver::BivariateSolver(BivariateBasis* basis,
				 double sigma_x,
				 double sigma_y,
				 double rho,
				 double a,
				 double x_0,
				 double b,
				 double c,
				 double y_0,
				 double d,
				 double t,
				 double dx)
  : a_(a),
    b_(b),
    c_(c),
    d_(d),
    sigma_x_(sigma_x),
    sigma_y_(sigma_y),
    rho_(rho),
    x_0_(x_0),
    y_0_(y_0),
    mvtnorm_(MultivariateNormal()),
    basis_(basis),
    small_t_solution_(NULL),
    t_(t),
    dx_(dx),
    IC_coefs_(gsl_vector_alloc(basis->get_orthonormal_elements().size())),
    mass_matrix_(gsl_matrix_alloc(basis_->get_orthonormal_elements().size(),
				  basis_->get_orthonormal_elements().size())),
    stiffness_matrix_(gsl_matrix_alloc(basis_->get_orthonormal_elements().size(),
				       basis_->get_orthonormal_elements().size())),
    eval_(gsl_vector_alloc(basis_->get_orthonormal_elements().size())),
    evec_(gsl_matrix_alloc(basis_->get_orthonormal_elements().size(),
			   basis_->get_orthonormal_elements().size())),
    solution_coefs_(gsl_vector_alloc(basis->get_orthonormal_elements().size()))
{
  if (x_0 < a || x_0 > b || y_0 < c || y_0 > d) {
    std::cout << "ERROR: IC out of range" << std::endl;
  }
  // std::cout << "before the first-ever scaling, we have\n";
  // std::cout << "sigma_x_ = " << sigma_x_ << "\n";
  // std::cout << "sigma_y_ = " << sigma_y_ << "\n";
  // std::cout << "t_ = " << t_ << std::endl;
  set_scaled_data();

  small_t_solution_ = new BivariateSolverClassical(sigma_x_2_,
						   sigma_y_2_,
						   rho_,
						   x_0_2_,
						   y_0_2_),
  small_t_solution_->set_function_grid(dx_);

  set_mass_and_stiffness_matrices();
  set_eval_and_evec();
  set_IC_coefs();
  set_solution_coefs();
}

BivariateSolver::~BivariateSolver()
{
  delete small_t_solution_;

  // freeing vectors
  gsl_vector_free(solution_coefs_);

  // freeing matrices
  gsl_matrix_free(mass_matrix_);
  gsl_matrix_free(stiffness_matrix_);
}

// BivariateSolver::BivariateSolver(const BivariateSolver& solver)
//   :

BivariateSolver& BivariateSolver::
operator=(const BivariateSolver& rhs)
{
  a_ = rhs.a_;
  b_ = rhs.b_;
  c_ = rhs.c_;
  d_ = rhs.d_;
  sigma_x_ = rhs.sigma_x_;
  sigma_y_ = rhs.sigma_y_;
  rho_ = rhs.rho_;
  x_0_ = rhs.x_0_;
  y_0_ = rhs.y_0_;
  mvtnorm_ = MultivariateNormal();
  basis_ = rhs.basis_;

  t_ = rhs.t_;
  dx_ = rhs.dx_;

  gsl_vector_free(IC_coefs_);
  IC_coefs_ = gsl_vector_alloc(basis_->get_orthonormal_elements().size());
  gsl_vector_memcpy(IC_coefs_, rhs.IC_coefs_);

  gsl_matrix_free(mass_matrix_);
  mass_matrix_ = gsl_matrix_alloc(basis_->get_orthonormal_elements().size(),
				  basis_->get_orthonormal_elements().size());
  gsl_matrix_memcpy(mass_matrix_, rhs.mass_matrix_);

  gsl_matrix_free(stiffness_matrix_);
  stiffness_matrix_ = gsl_matrix_alloc(basis_->get_orthonormal_elements().size(),
				       basis_->get_orthonormal_elements().size());
  gsl_matrix_memcpy(stiffness_matrix_, rhs.stiffness_matrix_);

  gsl_vector_free(eval_);
  eval_ = gsl_vector_alloc(basis_->get_orthonormal_elements().size());
  gsl_vector_memcpy(eval_, rhs.eval_);

  gsl_matrix_free(evec_);
  evec_ = gsl_matrix_alloc(basis_->get_orthonormal_elements().size(),
			   basis_->get_orthonormal_elements().size());
  gsl_matrix_memcpy(evec_, rhs.evec_);

  gsl_vector_free(solution_coefs_);
  solution_coefs_ = gsl_vector_alloc(basis_->get_orthonormal_elements().size());
  gsl_vector_memcpy(solution_coefs_, rhs.solution_coefs_);

  set_scaled_data();
  delete small_t_solution_;
  small_t_solution_ = new BivariateSolverClassical(sigma_x_2_,
						   sigma_y_2_,
						   rho_,
						   x_0_2_,
						   y_0_2_),
  small_t_solution_->set_function_grid(dx_);

  set_mass_and_stiffness_matrices();
  set_eval_and_evec();
  set_IC_coefs();
  set_solution_coefs();

  return *this;
}

void BivariateSolver::set_diffusion_parameters(double sigma_x,
					       double sigma_y,
					       double rho)
{
  sigma_x_ = sigma_x;
  sigma_y_ = sigma_y;
  rho_ = rho;
  set_scaled_data();

  delete small_t_solution_;
  small_t_solution_ = new BivariateSolverClassical(sigma_x_2_,
						   sigma_y_2_,
						   rho_,
						   x_0_2_,
						   y_0_2_);
  small_t_solution_->set_function_grid(dx_);

  set_mass_and_stiffness_matrices();
  set_eval_and_evec();
  set_IC_coefs();
  set_solution_coefs();
}

gsl_vector* BivariateSolver::scale_input(const gsl_vector* input) const
{
  double x = gsl_vector_get(input,0);
  double y = gsl_vector_get(input,1);

  // STEP 1
  double x_1 = x - a_;
  double y_1 = y - c_;

  double a_1 = a_ - a_;
  double b_1 = b_ - a_;
  double c_1 = c_ - c_;
  double d_1 = d_ - c_;

  // STEP 2
  double Lx_2 = b_1 - a_1;
  double x_2 =  x_1 / Lx_2;

  double Ly_2 = d_1 - c_1;
  double y_2 =  y_1 / Ly_2;

  // STEP 3
  if (flipped_xy_flag_) {
    double x_3 = x_2;
    double y_3 = y_2;
    x_2 = y_3;
    y_2 = x_3;
  }

  gsl_vector* scaled_input = gsl_vector_alloc(2);
  gsl_vector_set(scaled_input, 0, x_2);
  gsl_vector_set(scaled_input, 1, y_2);

  return scaled_input;
}

void BivariateSolver::set_data(double a,
			       double x_0,
			       double b,
			       double c,
			       double y_0,
			       double d)
{
  a_ = a;
  b_ = b;
  c_ = c;
  d_ = d;
  x_0_ = x_0;
  y_0_ = y_0;
  set_scaled_data();

  // std::cout << "In set_data" << std::endl;
  // print_diffusion_params();

  delete small_t_solution_;
  small_t_solution_ = new BivariateSolverClassical(sigma_x_2_,
						   sigma_y_2_,
						   rho_,
						   x_0_2_,
						   y_0_2_);
  small_t_solution_->set_function_grid(dx_);

  set_mass_and_stiffness_matrices();
  set_eval_and_evec();
  set_IC_coefs();
  set_solution_coefs();
}

void BivariateSolver::set_diffusion_parameters_and_data(double sigma_x,
							double sigma_y,
							double rho,
							double t,
							double a,
							double x_0,
							double b,
							double c,
							double y_0,
							double d)
{
  t_ = t;
  a_ = a;
  b_ = b;
  c_ = c;
  d_ = d;
  x_0_ = x_0;
  y_0_ = y_0;
  sigma_x_ = sigma_x;
  sigma_y_ = sigma_y;

  set_scaled_data();

  // std::cout << "In set_diffusion_parameters_and_data" << std::endl;
  // print_diffusion_params();

  delete small_t_solution_;
  small_t_solution_ = new BivariateSolverClassical(sigma_x_2_,
						   sigma_y_2_,
						   rho_,
						   x_0_2_,
						   y_0_2_);
  small_t_solution_->set_function_grid(dx_);

  set_mass_and_stiffness_matrices();
  set_eval_and_evec();
  set_IC_coefs();
  set_solution_coefs();
}


const gsl_vector* BivariateSolver::get_solution_coefs() const
{
  return solution_coefs_;
}

const gsl_vector* BivariateSolver::get_ic_coefs() const
{
  return IC_coefs_;
}

const gsl_vector* BivariateSolver::get_evals() const
{
  return eval_;
}

double BivariateSolver::
operator()(const gsl_vector* input) const
{
  gsl_vector* scaled_input = scale_input(input);

  double out = 0;
  if ( std::signbit(t_2_-small_t_solution_->get_t()) ) {
    out =  (*small_t_solution_)(scaled_input, t_2_);
  } else {
    int x_int = std::trunc(gsl_vector_get(scaled_input, 0)/dx_);
    int y_int = std::trunc(gsl_vector_get(scaled_input, 1)/dx_);

    if (x_int == 1/dx_) {
      x_int = 1/dx_ - 1;
    }
    if (y_int == 1/dx_) {
      y_int = 1/dx_ - 1;
    }

    double x = gsl_vector_get(scaled_input, 0);
    double y = gsl_vector_get(scaled_input, 1);

    double x_1 = x_int*dx_;
    double x_2 = (x_int+1)*dx_;
    double y_1 = y_int*dx_;
    double y_2 = (y_int+1)*dx_;

    double f_11 = 0;
    double f_12 = 0;
    double f_21 = 0;
    double f_22 = 0;
    double current_f = 0;

    for (unsigned i=0; i<basis_->get_orthonormal_elements().size(); ++i) {
      f_11 = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
    			    x_int,
    			    y_int);
      f_12 = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
    			    x_int,
    			    y_int+1);
      f_21 = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
    			    x_int+1,
    			    y_int);
      f_22 = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
    			    x_int+1,
    			    y_int+1);
      current_f = 1.0/((x_2-x_1)*(y_2-y_1)) *
    	((x_2 - x) * (f_11*(y_2-y) + f_12*(y-y_1)) +
    	 (x - x_1) * (f_21*(y_2-y) + f_22*(y-y_1)));

      current_f = current_f * gsl_vector_get(solution_coefs_, i);

      out = out + current_f;
    }

    // for (unsigned i=0; i<basis_->get_orthonormal_elements().size(); ++i) {
    //   current_f = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
    // 			    x_int,
    // 			    y_int);
    //   current_f = current_f * gsl_vector_get(solution_coefs_, i);
    //   out = out + current_f;
    // }
  }

  double Lx_2 = b_ - a_;
  double Ly_2 = d_ - c_;
  out = out / (Lx_2 * Ly_2);

  gsl_vector_free(scaled_input);
  return out;
}

double BivariateSolver::analytic_solution(const gsl_vector* input) const
{
  int little_n = 10;

  std::vector<double> d1x (2*little_n + 1);
  std::vector<double> d2x (2*little_n + 1);

  std::vector<double> d1y (2*little_n + 1);
  std::vector<double> d2y (2*little_n + 1);

  double sum_x = 0.0;
  double sum_y = 0.0;

  for (int i=0; i<2*little_n+1; ++i) {
    int n = i - little_n;

    d1x[i] = std::pow(gsl_vector_get(input,0) - x_0_ - 2.0*n*(b_-a_), 2) /
      (2.0 * std::pow(sigma_x_, 2) * t_);
    d2x[i] = std::pow(gsl_vector_get(input,0) + x_0_ - 2.0*a_ - 2.0*n*(b_-a_), 2) /
      (2.0 * std::pow(sigma_x_, 2) * t_);

    d1y[i] = std::pow(gsl_vector_get(input,1) - y_0_ - 2.0*n*(d_-c_), 2) /
      (2.0 * std::pow(sigma_y_, 2) * t_);
    d2y[i] = std::pow(gsl_vector_get(input,1) + y_0_ - 2.0*c_ - 2.0*n*(d_-c_), 2) /
      (2.0 * std::pow(sigma_y_, 2) * t_);

    sum_x = sum_x +
      (std::exp(-d1x[i]) - std::exp(-d2x[i]));

    sum_y = sum_y +
      (std::exp(-d1y[i]) - std::exp(-d2y[i]));

    std::cout << "d1x[" << n << "] = " << d1x[i] << " ";
  }
  std::cout << std::endl;

  double out_x = (1.0/std::sqrt(2.0*M_PI*std::pow(sigma_x_,2)*t_))*
    sum_x;

  double out_y = (1.0/std::sqrt(2.0*M_PI*std::pow(sigma_y_,2)*t_))*
    sum_y;

  double out = out_x*out_y;
  std::cout << "t = " << t_ << ";" << std::endl;
  std::cout << "ax = " << a_ << ";" << std::endl;
  std::cout << "bx = " << b_ << ";" << std::endl;
  std::cout << "x.ic = " << x_0_ << ";" << std::endl;
  std::cout << "x.fc = " << gsl_vector_get(input,0) << ";" << std::endl;
  std::cout << "sigma.2.x = " << std::pow(sigma_x_, 2) << ";" << std::endl;
  std::cout << "out_x = " << out_x << ";" << std::endl;
  return out;
}

double BivariateSolver::numerical_likelihood_extended(const gsl_vector* input,
						      double h)
{
  double t_lower_bound = 0.3;
  double sigma_y_2_lower_bound = 0.40;
  double likelihood = -1.0;
  double likelihood_upper_bound = 25.0;

  double t_2_current = t_2_;
  double sigma_y_2_current = sigma_y_2_;

  set_scaled_data();
  if (t_2_ >= t_lower_bound && sigma_y_2_ >= sigma_y_2_lower_bound) {

    // printf("t_2_ = %f, sigma_y_2_ = %f, CONDITION 0 MET\n", t_2_, sigma_y_2_);
    likelihood = numerical_likelihood(input, h);

    if ( std::signbit(likelihood) ||
	 ( (likelihood-likelihood_upper_bound) > std::numeric_limits<double>::epsilon() ) ) {

      double t_current = t_;
      double sigma_x_current = sigma_x_;
      double sigma_y_current = sigma_y_;

      if (!flipped_xy_flag_) {
	sigma_y_ = sigma_y_2_lower_bound * sigma_x_ * (d_ - c_)/(b_ - a_);
      } else {
	sigma_x_ = sigma_y_2_lower_bound * sigma_y_ * (b_ - a_)/(d_ - c_);
      }

      set_diffusion_parameters(sigma_x_,
			       sigma_y_,
			       rho_);

      likelihood = extrapolate_sigma_y_direction(likelihood_upper_bound,
						 sigma_y_2_lower_bound,
						 sigma_y_2_,
						 sigma_x_,
						 sigma_y_,
						 flipped_xy_flag_,
						 input,
						 h);

      sigma_x_ = sigma_x_current;
      sigma_y_ = sigma_y_current;
      set_diffusion_parameters(sigma_x_,
			       sigma_y_,
			       rho_);
    }

  } else if (t_2_ < t_lower_bound && sigma_y_2_ >= sigma_y_2_lower_bound) {
    // EXTRAPOLATING IN THE T-DIRECTION
    // printf("t_2_ = %f, sigma_y_2_ = %f, CONDITION 1 MET\n", t_2_, sigma_y_2_);

    likelihood = extrapolate_t_direction(likelihood_upper_bound,
					 t_lower_bound,
					 t_2_current,
					 t_,
					 flipped_xy_flag_,
			 		 input,
					 h);

  } else if (t_2_ >= t_lower_bound && sigma_y_2_ < sigma_y_2_lower_bound) {
    // printf("t_2_ = %f, sigma_y_2_ = %f, CONDITION 2 MET\n", t_2_, sigma_y_2_);

    likelihood = extrapolate_sigma_y_direction(likelihood_upper_bound,
					       sigma_y_2_lower_bound,
					       sigma_y_2_current,
					       sigma_x_,
					       sigma_y_,
					       flipped_xy_flag_,
					       input,
					       h);

  } else {
    // printf("t_2_ = %f, sigma_y_2_ = %f, CONDITION 3 MET\n", t_2_, sigma_y_2_);

    double sigma_x_current = sigma_x_;
    double sigma_y_current = sigma_y_;

    // REMEMBER TO RE-SET SIGMAS
    // This sets sigma_y_2_ to be on the computational boundary
    if (!flipped_xy_flag_) {
      sigma_y_ = (sigma_y_2_lower_bound) *
	sigma_x_ * (d_ - c_)/(b_ - a_);
    } else {
      sigma_x_ = (sigma_y_2_lower_bound) *
	sigma_y_ * (b_ - a_)/(d_ - c_);
    }
    // -------------------------- //

    // fixing sigma_y_2_ on boundary and extrapolating to the small
    // t_2_
    double f1 = extrapolate_t_direction(likelihood_upper_bound,
					t_lower_bound,
					t_2_,
					t_,
					flipped_xy_flag_,
					input,
					h);
    double x1 = sigma_y_2_;

    sigma_x_ = sigma_x_current;
    sigma_y_ = sigma_y_current;
    set_scaled_data();
    // extrapolating to the true small sigma_y_2_ using exponential
    // truncation only.
    double beta = -1.0*log(f1)*x1;
    likelihood = exp(-beta/sigma_y_2_);

  }

  return likelihood;
}

double BivariateSolver::analytic_likelihood(const gsl_vector* input,
					    int little_n)
{
  double d1x [2*little_n + 1];
  double d2x [2*little_n + 1];

  double d1y [2*little_n + 1];
  double d2y [2*little_n + 1];

  for (int i=0; i<2*little_n+1; ++i) {
    int n = i - little_n;

    d1x[i] = std::pow(gsl_vector_get(input,0) - x_0_ - 2.0*n*(b_-a_), 2) /
      (2.0 * std::pow(sigma_x_, 2) * t_);
    d2x[i] = std::pow(gsl_vector_get(input,0) + x_0_ - 2.0*a_ - 2.0*n*(b_-a_), 2) /
      (2.0 * std::pow(sigma_x_, 2) * t_);
    d1y[i] = std::pow(gsl_vector_get(input,1) - y_0_ - 2.0*n*(d_-c_), 2) /
      (2.0 * std::pow(sigma_y_, 2) * t_);
    d2y[i] = std::pow(gsl_vector_get(input,1) + y_0_ - 2.0*c_ - 2.0*n*(d_-c_), 2) /
      (2.0 * std::pow(sigma_y_, 2) * t_);

    // d1x[i] = std::pow(gsl_vector_get(scaled_input,0) - x_0_ - 2.0*n, 2) /
    //   (2.0);
    // d2x[i] = std::pow(gsl_vector_get(scaled_input,0) + x_0_ - 2.0*n, 2) /
    //   (2.0);
    // d1y[i] = std::pow(gsl_vector_get(scaled_input,1) - y_0_ - 2.0*n, 2) /
    //   (2.0 * std::pow(sigma_y_, 2));
    // d2x[i] = std::pow(gsl_vector_get(scaled_input,1) + y_0_ - 2.0*n, 2) /
    //   (2.0 * std::pow(sigma_y_, 2));
  }

  double deriv_x = 0.0;
  double deriv_y = 0.0;

  double sum_x = 0.0;
  double sum_y = 0.0;

  for (int i=0; i<2*little_n+1; ++i) {
    int n = i - little_n;

    sum_x = sum_x +
      (4.0*n*n*(2.0*d1x[i] - 1)*std::exp(-d1x[i]) - 4.0*n*(n-1)*(2.0*d2x[i] - 1)*std::exp(-d2x[i]));

    sum_y = sum_y +
      (4.0*n*n*(2.0*d1y[i] - 1)*std::exp(-d1y[i]) - 4.0*n*(n-1)*(2.0*d2y[i] - 1)*std::exp(-d2y[i]));
  }

  deriv_x = 1.0/(std::sqrt(2*M_PI)*std::pow(sigma_x_*std::sqrt(t_), 3)) * sum_x;
  deriv_y = 1.0/(std::sqrt(2*M_PI)*std::pow(sigma_y_*std::sqrt(t_), 3)) * sum_y;

  double out = deriv_x*deriv_y;
  return (out);
}

double BivariateSolver::numerical_likelihood(const gsl_vector* input,
					     double h)
{
  double x = gsl_vector_get(input, 0);
  double y = gsl_vector_get(input, 1);
  double likelihood = 0;

  double h_x = h*(b_ - a_);
  double h_y = h*(d_ - c_);

  if (x > (a_+h_x) &&
      x < (b_-h_x) &&
      y > (c_+h_y) &&
      y < (d_-h_y) &&
      //
      x_0_ > (a_+h_x) &&
      x_0_ < (b_-h_x) &&
      y_0_ > (c_+h_y) &&
      y_0_ < (d_-h_y)) {
    likelihood = numerical_likelihood_second_order(input, h);
    // printf("second order deriv \n");
  } else {
    // printf("first order deriv \n");
    likelihood = numerical_likelihood_first_order(input, h);
  }


  return likelihood;
}


double BivariateSolver::numerical_likelihood_second_order(const gsl_vector* raw_input,
							  double h)
{
  // there are 16 solutions to be computed
  double current_a = a_;
  double current_b = b_;
  double current_c = c_;
  double current_d = d_;
  double current_x_0 = x_0_;
  double current_y_0 = y_0_;
  // double Lx = b_ - a_;
  // double Ly = d_ - c_;

  // // NORMALIZING BEFORE DIFFERENTIATION START
  // double a = a_/Lx;
  // double x_0 = x_0_/Lx;
  // double b = b_/Lx;

  // double c = c_/Ly;
  // double y_0 = y_0_/Ly;
  // double d = d_/Ly;

  // set_data(a, x_0, b,
  // 	   c, y_0, d);

  // gsl_vector* input = scale_input(raw_input);
  // // NORMALIZING BEFORE DIFFERENTIATION END

  std::vector<int> a_indeces { 1,-1};
  std::vector<int> b_indeces {-1, 1};
  std::vector<int> c_indeces { 1,-1};
  std::vector<int> d_indeces {-1, 1};

  int a_power=1;
  int b_power=1;
  int c_power=1;
  int d_power=1;

  double derivative = 0;
  double h_x = h*(b_ - a_);
  double h_y = h*(d_ - c_);

  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=1; } else { a_power=0; };

    for (unsigned j=0; j<b_indeces.size(); ++j) {
      if (j==0) { b_power=1; } else { b_power=0; };

      for (unsigned k=0; k<c_indeces.size(); ++k) {
	if (k==0) { c_power=1; } else { c_power=0; };

	for (unsigned l=0; l<d_indeces.size(); ++l) {
	  if (l==0) { d_power=1; } else { d_power=0; };

	  set_data(current_a + a_indeces[i]*h_x,
		   current_x_0,
		   current_b + b_indeces[j]*h_x,
		   current_c + c_indeces[k]*h_y,
		   current_y_0,
		   current_d + d_indeces[l]*h_y);

	  double out = (*this)(raw_input);

	  derivative = derivative +
	    std::pow(-1, a_power)*
	    std::pow(-1, b_power)*
	    std::pow(-1, c_power)*
	    std::pow(-1, d_power)*out;
	}
      }
    }
  }

  set_data(current_a,
  	   current_x_0,
  	   current_b,
  	   current_c,
  	   current_y_0,
  	   current_d);

  // derivative = derivative / (16 * h^2 * h^2);
  if (std::signbit(derivative)) {
    derivative =
      -std::exp(std::log(std::abs(derivative)) - log(16) - 2*log(h_x) - 2*log(h_y));
  } else {
    derivative =
      std::exp(std::log(std::abs(derivative)) - log(16) - 2*log(h_x) - 2*log(h_y));
  }

  //  derivative = derivative/(std::pow(Lx, 3) * std::pow(Ly,3));
  //  gsl_vector_free(raw_input);
  return derivative;
}

double BivariateSolver::analytic_likelihood_second_order_small_t(const gsl_vector* raw_input,
								 double small_t,
								 double h)
{
  printf("in analytic_likelihood_second_order_small_t\n");
  gsl_matrix* Rotation_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_memcpy(Rotation_matrix, small_t_solution_->get_rotation_matrix());

  double Rotation_matrix_inv [4];
  gsl_matrix_view Rotation_matrix_inv_view =
    gsl_matrix_view_array(Rotation_matrix_inv, 2, 2);

  gsl_matrix_memcpy(&Rotation_matrix_inv_view.matrix,
		    small_t_solution_->get_rotation_matrix());
  
  int s = 0;
  gsl_permutation * p = gsl_permutation_alloc(2);
  gsl_linalg_LU_decomp(Rotation_matrix, p, &s);
  gsl_linalg_LU_invert(Rotation_matrix, p, &Rotation_matrix_inv_view.matrix);
  gsl_permutation_free(p);
  gsl_matrix_free(Rotation_matrix);
  
  gsl_vector * scaled_input = scale_input(raw_input);
  double corner_points_array [20] = {a_2_, b_2_, b_2_, a_2_, x_0_2_, gsl_vector_get(scaled_input,0), x_0_2_, x_0_2_, x_0_2_, x_0_2_,
				     c_2_, c_2_, d_2_, d_2_, y_0_2_, gsl_vector_get(scaled_input,1), y_0_2_, y_0_2_, y_0_2_, y_0_2_};

  double images_array [32];
  for (unsigned i=0; i<16; ++i) {
    images_array[i] = x_0_2_;
    images_array[i+16] = y_0_2_;
  }
  double images_transformed_array [32];

  printf("library(gridExtra);\n");
  printf("par(mar=c(2,2,0,0));\n");
  printf("plot(x=0, type=\"n\", xlim = c(-0.5, 1.5), ylim=c(-0.5, 1.5), xlab=\"x\", ylab=\"y\");\n");
  printf("lines(x=c(0,1), y=c(0,0), lwd=2, col=\"black\");\n"); // border 1
  printf("lines(x=c(1,1), y=c(0,1), lwd=2, col=\"black\");\n"); // border 2
  printf("lines(x=c(1,0), y=c(1,1), lwd=2, col=\"black\");\n"); // border 3
  printf("lines(x=c(0,0), y=c(1,0), lwd=2, col=\"black\");\n"); // border 4
  printf("points(x=%g, y=%g, lwd=3, pch=20, col=\"black\");\n",
	 gsl_vector_get(scaled_input, 0),
	 gsl_vector_get(scaled_input, 0)); // IC
  printf("samples <- rmvnorm(n=1e6, mean=c(%g,%g), sigma=diag(c(1,%g)));\n",
	 gsl_vector_get(scaled_input, 0),
	 gsl_vector_get(scaled_input, 0),
	 sigma_y_2_);
  

  gsl_matrix_view images_view = gsl_matrix_view_array(images_array, 2, 16);
  gsl_matrix_view images_transformed_view = gsl_matrix_view_array(images_transformed_array, 2, 16);
  
  gsl_matrix_view corner_points_view =
    gsl_matrix_view_array(corner_points_array, 2, 10);

  gsl_vector_view lower_left_view = gsl_matrix_column(&corner_points_view.matrix,0);
  gsl_vector_view lower_right_view = gsl_matrix_column(&corner_points_view.matrix,1);
  gsl_vector_view upper_right_view = gsl_matrix_column(&corner_points_view.matrix,2);
  gsl_vector_view upper_left_view = gsl_matrix_column(&corner_points_view.matrix,3);
  gsl_vector_view final_condition_view = gsl_matrix_column(&corner_points_view.matrix,4);
  gsl_vector_view point_view = gsl_matrix_column(&corner_points_view.matrix,5);
  gsl_vector_view initial_condition_view = gsl_matrix_column(&corner_points_view.matrix,6);
  
  double corner_points_transformed_array [20];
  gsl_matrix_view corner_points_transformed_view =
    gsl_matrix_view_array(corner_points_transformed_array, 2, 10);

  gsl_vector_view lower_left_transformed_view = gsl_matrix_column(&corner_points_transformed_view.matrix,0);
  gsl_vector_view lower_right_transformed_view = gsl_matrix_column(&corner_points_transformed_view.matrix,1);
  gsl_vector_view upper_right_transformed_view = gsl_matrix_column(&corner_points_transformed_view.matrix,2);
  gsl_vector_view upper_left_transformed_view = gsl_matrix_column(&corner_points_transformed_view.matrix,3);
  gsl_vector_view final_condition_transformed_view = gsl_matrix_column(&corner_points_transformed_view.matrix,4);
  gsl_vector_view point_transformed_view = gsl_matrix_column(&corner_points_transformed_view.matrix,5);
  gsl_vector_view initial_condition_transformed_view = gsl_matrix_column(&corner_points_transformed_view.matrix,6);
  
  // C = alpha*op(A)*op(B) + beta*C
  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
		 CblasNoTrans, //op(B) = B
		 1.0, //alpha=1
		 small_t_solution_->get_rotation_matrix(), //A
		 &corner_points_view.matrix, //B
		 0.0, //beta=0
		 &corner_points_transformed_view.matrix); //C

  // C = alpha*op(A)*op(B) + beta*C
  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
		 CblasNoTrans, //op(B) = B
		 1.0, //alpha=1
		 small_t_solution_->get_rotation_matrix(), //A
		 &images_view.matrix, //B
		 0.0, //beta=0
		 &images_transformed_view.matrix); //C
  
  printf("corner1=c(%g,%g); corner2=c(%g,%g); corner3=c(%g,%g); corner4=c(%g,%g); final.condition=c(%g,%g); initial.condition=c(%g,%g);\n",
	 gsl_vector_get(&lower_left_view.vector,0), // corner1 x
	 gsl_vector_get(&lower_left_view.vector,1), // corner1 y
	 gsl_vector_get(&lower_right_view.vector,0), // corner2 x
	 gsl_vector_get(&lower_right_view.vector,1), // corner2 y
	 gsl_vector_get(&upper_right_view.vector,0), // corner3 x
	 gsl_vector_get(&upper_right_view.vector,1), // corner3 y
	 gsl_vector_get(&upper_left_view.vector,0), // corner4 x
	 gsl_vector_get(&upper_left_view.vector,1), // corner4 y
	 gsl_vector_get(&final_condition_view.vector,0), // final condition x
	 gsl_vector_get(&final_condition_view.vector,1), // final condition y
	 gsl_vector_get(&initial_condition_view.vector,0),
	 gsl_vector_get(&initial_condition_view.vector,1));

  printf("corner1=c(%g,%g); corner2=c(%g,%g); corner3=c(%g,%g); corner4=c(%g,%g); final.condition=c(%g,%g); initial.condition=c(%g,%g);\n",
	 gsl_vector_get(&lower_left_transformed_view.vector,0), // corner1 x
	 gsl_vector_get(&lower_left_transformed_view.vector,1), // corner1 y
	 gsl_vector_get(&lower_right_transformed_view.vector,0), // corner2 x
	 gsl_vector_get(&lower_right_transformed_view.vector,1), // corner2 y
	 gsl_vector_get(&upper_right_transformed_view.vector,0), // corner3 x
	 gsl_vector_get(&upper_right_transformed_view.vector,1), // corner3 y
	 gsl_vector_get(&upper_left_transformed_view.vector,0), // corner4 x
	 gsl_vector_get(&upper_left_transformed_view.vector,1), // corner4 y
	 gsl_vector_get(&final_condition_transformed_view.vector,0), // final condition x
	 gsl_vector_get(&final_condition_transformed_view.vector,1), // final condition y
	 gsl_vector_get(&initial_condition_transformed_view.vector,0), // initial condition x
	 gsl_vector_get(&initial_condition_transformed_view.vector,1)); // initial condition y

  std::vector<std::vector<gsl_vector*>> lines {
    std::vector<gsl_vector*> {&lower_left_transformed_view.vector,
	&lower_right_transformed_view.vector}, // line 1
      std::vector<gsl_vector*> {&upper_right_transformed_view.vector,
	  &lower_right_transformed_view.vector}, // line 2
	std::vector<gsl_vector*> {&upper_left_transformed_view.vector,
	    &upper_right_transformed_view.vector}, // line 3
	  std::vector<gsl_vector*> {&upper_left_transformed_view.vector,
	      &lower_left_transformed_view.vector} // line 4
  };

  std::vector<double> distance_to_line {
    small_t_solution_->
      distance_from_point_to_axis_raw(lines[0][0],
				      lines[0][1],
				      &initial_condition_transformed_view.vector,
				      &final_condition_transformed_view.vector),
      small_t_solution_->
      distance_from_point_to_axis_raw(lines[1][0],
				      lines[1][1],
				      &initial_condition_transformed_view.vector,
				      &final_condition_transformed_view.vector),
      small_t_solution_->
      distance_from_point_to_axis_raw(lines[2][0],
				      lines[2][1],
				      &initial_condition_transformed_view.vector,
				      &final_condition_transformed_view.vector),
      small_t_solution_->
      distance_from_point_to_axis_raw(lines[3][0],
				      lines[3][1],
				      &initial_condition_transformed_view.vector,
				      &final_condition_transformed_view.vector)};
	  
  std::vector<unsigned> distance_to_line_indeces (4);
  unsigned n=0;
  std::generate(distance_to_line_indeces.begin(),
		distance_to_line_indeces.end(),
		[&n]{ return n++; });

  std::sort(distance_to_line_indeces.begin(), distance_to_line_indeces.end(),
	    [&distance_to_line] (unsigned i1, unsigned i2) -> bool
	    {
	      return distance_to_line[i1] < distance_to_line[i2];
	    });

  std::vector<gsl_vector_view> images_vector (16);
  for (unsigned i=0; i<16; ++i) {
    images_vector[i] =
      gsl_matrix_column(&images_transformed_view.matrix,i);
  }

  unsigned counter = 0;
  for (unsigned l=0; l<2; ++l) {
    for (unsigned k=0; k<2; ++k) {
      for (unsigned j=0; j<2; ++j) {
	for (unsigned i=0; i<2; ++i) {
	  gsl_vector* current_image = &images_vector[counter].vector;
	  if (i==1) {
	    small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[0]][0],
						 lines[distance_to_line_indeces[0]][1],
						 current_image);
	  }
	  if (j==1) {
	    small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[1]][0],
						 lines[distance_to_line_indeces[1]][1],
						 current_image);
	  }
	  if (k==1) {
	    small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[2]][0],
						 lines[distance_to_line_indeces[2]][1],
						 current_image);
	  }
	  if (l==1) {
	    small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[3]][0],
						 lines[distance_to_line_indeces[3]][1],
						 current_image);
	  }
	  counter = counter + 1;
	  std::cout << "image " << counter << " distances: ";


	  double d1 = small_t_solution_->
	    distance_from_point_to_axis_raw(lines[distance_to_line_indeces[0]][0],
					    lines[distance_to_line_indeces[0]][1],
					    &initial_condition_transformed_view.vector,
					    current_image);
	  double d2 = small_t_solution_->
	    distance_from_point_to_axis_raw(lines[distance_to_line_indeces[1]][0],
					    lines[distance_to_line_indeces[1]][1],
					    &initial_condition_transformed_view.vector,
					    current_image);
	  double d3 = small_t_solution_->
	    distance_from_point_to_axis_raw(lines[distance_to_line_indeces[2]][0],
					    lines[distance_to_line_indeces[2]][1],
					    &initial_condition_transformed_view.vector,
					    current_image);
	  double d4 = small_t_solution_->
	    distance_from_point_to_axis_raw(lines[distance_to_line_indeces[2]][0],
					    lines[distance_to_line_indeces[2]][1],
					    &initial_condition_transformed_view.vector,
					    current_image);
	  std::cout << d1 << " " << d2 << " " << d3 << " " << d4 << "\n";
	  
	}
      }
    }
  }

  
  gsl_vector_view image_1_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,7);
  gsl_vector_view image_2_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,8);
  gsl_vector_view image_3_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,9);
  // REFLECTION FINAL CONDITION
  small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[0]][0],
				       lines[distance_to_line_indeces[0]][1],
				       &final_condition_transformed_view.vector);
  
  small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[1]][0],
				       lines[distance_to_line_indeces[1]][1],
				       &final_condition_transformed_view.vector);
  
  small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[2]][0],
  				       lines[distance_to_line_indeces[2]][1],
  				       &final_condition_transformed_view.vector);
  
  small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[3]][0],
  				       lines[distance_to_line_indeces[3]][1],
  				       &final_condition_transformed_view.vector);
  
  // REFLECTION IMAGE 1
  small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[0]][0],
				       lines[distance_to_line_indeces[0]][1],
				       &image_1_transformed_view.vector);
  // REFLECTION IMAGE 2
  small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[1]][0],
				       lines[distance_to_line_indeces[1]][1],
				       &image_2_transformed_view.vector);
  // REFLECTION IMAGE 3
  small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[0]][0],
				       lines[distance_to_line_indeces[0]][1],
				       &image_3_transformed_view.vector);
  small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[1]][0],
				       lines[distance_to_line_indeces[1]][1],
				       &image_3_transformed_view.vector);

printf("\ntwo TIMES from image1 and IC to line 2 are: %g, %g\n",
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[1]][0],
								     lines[distance_to_line_indeces[1]][1],
								     &initial_condition_transformed_view.vector,
								     &image_1_transformed_view.vector)/5.0, 2),
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[1]][0],
								     lines[distance_to_line_indeces[1]][1],
								     &initial_condition_transformed_view.vector,
								     &initial_condition_transformed_view.vector)/5.0, 2));
 printf("two TIMES from IC and image1 to line 3 are: %g, %g\n",
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[2]][0],
								     lines[distance_to_line_indeces[2]][1],
								     &initial_condition_transformed_view.vector,
								     &image_1_transformed_view.vector)/5.0, 2),
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[2]][0],
								     lines[distance_to_line_indeces[2]][1],
								     &initial_condition_transformed_view.vector,
								     &initial_condition_transformed_view.vector)/5.0, 2));
  printf("two TIMES from IC and image1 to line 4 are: %g, %g\n\n",
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[3]][0],
								     lines[distance_to_line_indeces[3]][1],
								     &initial_condition_transformed_view.vector,
								     &image_1_transformed_view.vector)/5.0, 2),
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[3]][0],
								     lines[distance_to_line_indeces[3]][1],
								     &initial_condition_transformed_view.vector,
								     &initial_condition_transformed_view.vector)/5.0, 2));

  printf("two TIMES from IC to lines 1,2, 3 and 4 are: %g, %g, %g, %g\n\n",
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[0]][0],
								     lines[distance_to_line_indeces[0]][1],
								     &initial_condition_transformed_view.vector,
								     &initial_condition_transformed_view.vector)/5.0, 2),
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[1]][0],
								     lines[distance_to_line_indeces[1]][1],
								     &initial_condition_transformed_view.vector,
								     &initial_condition_transformed_view.vector)/5.0, 2),
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[2]][0],
								     lines[distance_to_line_indeces[2]][1],
								     &initial_condition_transformed_view.vector,
								     &initial_condition_transformed_view.vector)/5.0, 2),
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[3]][0],
								     lines[distance_to_line_indeces[3]][1],
								     &initial_condition_transformed_view.vector,
								     &initial_condition_transformed_view.vector)/5.0, 2));
  
  printf("two distances from images 2 and 3 to line 1 are: %g, %g\n",
	 small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[0]][0],
							    lines[distance_to_line_indeces[0]][1],
							    &initial_condition_transformed_view.vector,
							    &image_2_transformed_view.vector),
	 small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[0]][0],
							    lines[distance_to_line_indeces[0]][1],
							    &initial_condition_transformed_view.vector,
							    &image_3_transformed_view.vector));
  printf("two TIMES from images 2 and 3 to line 1 are: %g, %g\n",
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[0]][0],
								     lines[distance_to_line_indeces[0]][1],
								     &initial_condition_transformed_view.vector,
								     &image_2_transformed_view.vector)/5.0, 2),
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[0]][0],
								     lines[distance_to_line_indeces[0]][1],
								     &initial_condition_transformed_view.vector,
								     &image_3_transformed_view.vector)/5.0, 2));
  printf("two TIMES from images 2 and 3 to line 3 are: %g, %g\n",
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[2]][0],
								     lines[distance_to_line_indeces[2]][1],
								     &initial_condition_transformed_view.vector,
								     &image_2_transformed_view.vector)/5.0, 2),
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[2]][0],
								     lines[distance_to_line_indeces[2]][1],
								     &initial_condition_transformed_view.vector,
								     &image_3_transformed_view.vector)/5.0, 2));
  printf("two TIMES from images 2 and 3 to line 4 are: %g, %g\n",
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[3]][0],
								     lines[distance_to_line_indeces[3]][1],
								     &initial_condition_transformed_view.vector,
								     &image_2_transformed_view.vector)/5.0, 2),
	 std::pow(small_t_solution_->distance_from_point_to_axis_raw(lines[distance_to_line_indeces[3]][0],
								     lines[distance_to_line_indeces[3]][1],
								     &initial_condition_transformed_view.vector,
								     &image_3_transformed_view.vector)/5.0, 2)); 
  printf("current small time is %g, and current time is %g \n\n",
	 small_t_solution_->get_t(),
	 t_2_);

  printf("flipped_flag = %u\n", flipped_xy_flag_);
  printf("after reflection\ncorner1=c(%g,%g); corner2=c(%g,%g); corner3=c(%g,%g); corner4=c(%g,%g); final.condition=c(%g,%g); initial.condition=c(%g,%g);\n",
	 gsl_vector_get(&lower_left_transformed_view.vector,0), // corner1 x
	 gsl_vector_get(&lower_left_transformed_view.vector,1), // corner1 y
	 gsl_vector_get(&lower_right_transformed_view.vector,0), // corner2 x
	 gsl_vector_get(&lower_right_transformed_view.vector,1), // corner2 y
	 gsl_vector_get(&upper_right_transformed_view.vector,0), // corner3 x
	 gsl_vector_get(&upper_right_transformed_view.vector,1), // corner3 y
	 gsl_vector_get(&upper_left_transformed_view.vector,0), // corner4 x
	 gsl_vector_get(&upper_left_transformed_view.vector,1), // corner4 y
	 gsl_vector_get(&final_condition_transformed_view.vector,0), // final condition x
	 gsl_vector_get(&final_condition_transformed_view.vector,1), // final condition y
	 gsl_vector_get(&initial_condition_transformed_view.vector,0),
	 gsl_vector_get(&initial_condition_transformed_view.vector,1));
  printf("plot(x=1,type=\"n\",xlim=6*c(-sqrt(2),sqrt(2)),ylim=6*c(-sqrt(2),sqrt(2)));\n");
  printf("lines(c(corner1[1], corner2[1]),c(corner1[2], corner2[2]));\n");
  printf("lines(c(corner2[1], corner3[1]),c(corner2[2], corner3[2]));\n");
  printf("lines(c(corner3[1], corner4[1]),c(corner3[2], corner4[2]));\n");
  printf("lines(c(corner4[1], corner1[1]),c(corner4[2], corner1[2]));\n");
  printf("points(final.condition[1], final.condition[2],lwd=2,col=\"red\");\n");
  printf("points(initial.condition[1], initial.condition[2],lwd=2,col=\"red\");\n");
  printf("\nimage.1=c(%g,%g); image.2=c(%g,%g); image.3=c(%g,%g); image.4=c(%g,%g);\n",
  	 gsl_vector_get(&images_vector[0].vector,0),
  	 gsl_vector_get(&images_vector[0].vector,1),
  	 gsl_vector_get(&images_vector[1].vector,0),
  	 gsl_vector_get(&images_vector[1].vector,1),
  	 gsl_vector_get(&images_vector[2].vector,0),
  	 gsl_vector_get(&images_vector[2].vector,1),
  	 gsl_vector_get(&images_vector[3].vector,0),
  	 gsl_vector_get(&images_vector[3].vector,1));
  printf("points(image.1[1], image.1[2],lwd=2,col=\"green\");\n");
  printf("points(image.2[1], image.2[2],lwd=2,col=\"green\");\n");
  printf("points(image.3[1], image.3[2],lwd=2,col=\"green\");\n");
  printf("points(image.4[1], image.4[2],lwd=2,col=\"green\");\n");
  printf("\nimage.5=c(%g,%g); image.6=c(%g,%g); image.7=c(%g,%g); image.8=c(%g,%g);\n",
  	 gsl_vector_get(&images_vector[4].vector,0),
  	 gsl_vector_get(&images_vector[4].vector,1),
  	 gsl_vector_get(&images_vector[5].vector,0),
  	 gsl_vector_get(&images_vector[5].vector,1),
  	 gsl_vector_get(&images_vector[6].vector,0),
  	 gsl_vector_get(&images_vector[6].vector,1),
  	 gsl_vector_get(&images_vector[7].vector,0),
  	 gsl_vector_get(&images_vector[7].vector,1));
  printf("points(image.5[1], image.5[2],lwd=2,col=\"green\");\n");
  printf("points(image.6[1], image.6[2],lwd=2,col=\"green\");\n");
  printf("points(image.7[1], image.7[2],lwd=2,col=\"green\");\n");
  printf("points(image.8[1], image.8[2],lwd=2,col=\"green\");\n");

  printf("\nimage.9=c(%g,%g); image.10=c(%g,%g); image.11=c(%g,%g); image.12=c(%g,%g);\n",
  	 gsl_vector_get(&images_vector[8].vector,0),
  	 gsl_vector_get(&images_vector[8].vector,1),
  	 gsl_vector_get(&images_vector[9].vector,0),
  	 gsl_vector_get(&images_vector[9].vector,1),
  	 gsl_vector_get(&images_vector[10].vector,0),
  	 gsl_vector_get(&images_vector[10].vector,1),
  	 gsl_vector_get(&images_vector[11].vector,0),
  	 gsl_vector_get(&images_vector[11].vector,1));
  printf("points(image.9[1], image.9[2],lwd=2,col=\"green\");\n");
  printf("points(image.10[1], image.10[2],lwd=2,col=\"green\");\n");
  printf("points(image.11[1], image.11[2],lwd=2,col=\"green\");\n");
  printf("points(image.12[1], image.12[2],lwd=2,col=\"green\");\n");
  printf("\nimage.13=c(%g,%g); image.14=c(%g,%g); image.15=c(%g,%g); image.16=c(%g,%g);\n",
  	 gsl_vector_get(&images_vector[12].vector,0),
  	 gsl_vector_get(&images_vector[12].vector,1),
  	 gsl_vector_get(&images_vector[13].vector,0),
  	 gsl_vector_get(&images_vector[13].vector,1),
  	 gsl_vector_get(&images_vector[14].vector,0),
  	 gsl_vector_get(&images_vector[14].vector,1),
  	 gsl_vector_get(&images_vector[15].vector,0),
  	 gsl_vector_get(&images_vector[15].vector,1));
  printf("points(image.13[1], image.13[2],lwd=2,col=\"green\");\n");
  printf("points(image.14[1], image.14[2],lwd=2,col=\"green\");\n");
  printf("points(image.15[1], image.15[2],lwd=2,col=\"green\");\n");
  printf("points(image.16[1], image.16[2],lwd=2,col=\"green\");\n");
  
  // C = alpha*op(A)*op(B) + beta*C
  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
		 CblasNoTrans, //op(B) = B
		 1.0, //alpha=1
		 &Rotation_matrix_inv_view.matrix, //A
		 &corner_points_transformed_view.matrix, //B
		 0.0, //beta=0
		 &corner_points_view.matrix); //C

  printf("transforming back\ncorner1=c(%g,%g); corner2=c(%g,%g); corner3=c(%g,%g); corner4=c(%g,%g); final.condition=c(%g,%g);\n",
	 gsl_vector_get(&lower_left_view.vector,0), // corner1 x
	 gsl_vector_get(&lower_left_view.vector,1), // corner1 y
	 gsl_vector_get(&lower_right_view.vector,0), // corner2 x
	 gsl_vector_get(&lower_right_view.vector,1), // corner2 y
	 gsl_vector_get(&upper_right_view.vector,0), // corner3 x
	 gsl_vector_get(&upper_right_view.vector,1), // corner3 y
	 gsl_vector_get(&upper_left_view.vector,0), // corner4 x
	 gsl_vector_get(&upper_left_view.vector,1), // corner4 y
	 gsl_vector_get(&final_condition_view.vector,0), // final condition x
	 gsl_vector_get(&final_condition_view.vector,1)); // final condition y
  printf("rho = %g\n", rho_);
  printf("sigma = %g\n", sigma_y_2_); 
  printf("t = %g\n", t_2_); 
  gsl_vector_free(scaled_input);

  // MultivariateNormal() mvtnorm = MultivariateNormal();
  //  double log_initial_q = 
  
  // // there are 16 solutions to be computed
  // double current_a = a_;
  // double current_b = b_;
  // double current_c = c_;
  // double current_d = d_;
  // double current_x_0 = x_0_;
  // double current_y_0 = y_0_;

  // std::vector<int> a_indeces { 1,-1};
  // std::vector<int> b_indeces {-1, 1};
  // std::vector<int> c_indeces { 1,-1};
  // std::vector<int> d_indeces {-1, 1};

  // int a_power=1;
  // int b_power=1;
  // int c_power=1;
  // int d_power=1;

  double derivative = 0;
  // double h_x = h*(b_ - a_);
  // double h_y = h*(d_ - c_);

  // for (unsigned i=0; i<a_indeces.size(); ++i) {
  //   if (i==0) { a_power=1; } else { a_power=0; };

  //   for (unsigned j=0; j<b_indeces.size(); ++j) {
  //     if (j==0) { b_power=1; } else { b_power=0; };

  //   //   for (unsigned k=0; k<c_indeces.size(); ++k) {
  //   //   	if (k==0) { c_power=1; } else { c_power=0; };

  //   // 	for (unsigned l=0; l<d_indeces.size(); ++l) {
  //   // 	  if (l==0) { d_power=1; } else { d_power=0; };

  //     set_data(current_a + a_indeces[i]*h_x,
  // 	  	   current_x_0,
  // 	  	   current_b + b_indeces[j]*h_x,
  // 	  	   current_c,
  // 	  	   current_y_0,
  // 	  	   current_d);
    
  // 	  // set_data(current_a + a_indeces[i]*h_x,
  // 	  // 	   current_x_0,
  // 	  // 	   current_b + b_indeces[j]*h_x,
  // 	  // 	   current_c + c_indeces[k]*h_y,
  // 	  // 	   current_y_0,
  // 	  // 	   current_d + d_indeces[l]*h_y);

  // 	  gsl_matrix* Rotation_matrix = gsl_matrix_alloc(2,2);
  // 	  gsl_matrix_memcpy(Rotation_matrix, small_t_solution_->get_rotation_matrix());

  // 	  gsl_vector* scaled_input = scale_input(raw_input);
  // 	  double scaled_center [2] = {x_0_2_, y_0_2_};
  // 	  gsl_vector_view scaled_center_view = gsl_vector_view_array(scaled_center, 2);

    
  // 	  double corner_points_array [14] = {0, 1, 1, 0, x_0_2_, gsl_vector_get(scaled_input,0), x_0_2_,
  // 					     0, 0, 1, 1, y_0_2_, gsl_vector_get(scaled_input,1), y_0_2_};
  // 	  gsl_matrix_view corner_points_view =
  // 	    gsl_matrix_view_array(corner_points_array, 2, 7);

  // 	  double corner_points_transformed_array [14];
  // 	  gsl_matrix_view corner_points_transformed_view =
  // 	    gsl_matrix_view_array(corner_points_transformed_array, 2, 7);

  // 	  // C = alpha*op(A)*op(B) + beta*C
  // 	  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  // 			 CblasNoTrans, //op(B) = B
  // 			 1.0, //alpha=1
  // 			 Rotation_matrix, //A
  // 			 &corner_points_view.matrix, //B
  // 			 0.0, //beta=0
  // 			 &corner_points_transformed_view.matrix); //C
    
  // 	  gsl_vector_view lower_left_view = gsl_matrix_column(&corner_points_transformed_view.matrix,0);
  // 	  gsl_vector_view lower_right_view = gsl_matrix_column(&corner_points_transformed_view.matrix,1);
  // 	  gsl_vector_view upper_right_view = gsl_matrix_column(&corner_points_transformed_view.matrix,2);
  // 	  gsl_vector_view upper_left_view = gsl_matrix_column(&corner_points_transformed_view.matrix,3);
  // 	  gsl_vector_view final_condition_view = gsl_matrix_column(&corner_points_transformed_view.matrix,4);
  // 	  gsl_vector_view point_view = gsl_matrix_column(&corner_points_transformed_view.matrix,5);
  // 	  gsl_vector_view initial_condition_view = gsl_matrix_column(&corner_points_transformed_view.matrix,6);

  // 	  printf("corner1=c(%g,%g); corner2=c(%g,%g); corner3=c(%g,%g); corner4=c(%g,%g);\n",
  // 	  	 gsl_vector_get(&lower_left_view.vector,0), // corner1 x
  // 	  	 gsl_vector_get(&lower_left_view.vector,1), // corner1 y
  // 	  	 gsl_vector_get(&lower_right_view.vector,0), // corner2 x
  // 	  	 gsl_vector_get(&lower_right_view.vector,1), // corner2 y
  // 	  	 gsl_vector_get(&upper_right_view.vector,0), // corner3 x
  // 	  	 gsl_vector_get(&upper_right_view.vector,1), // corner3 y
  // 	  	 gsl_vector_get(&upper_left_view.vector,0), // corner4 x
  // 		 gsl_vector_get(&upper_left_view.vector,1)); // corner4 y


  // 	  std::vector<std::vector<gsl_vector*>> lines {
  // 	    std::vector<gsl_vector*> {&lower_left_view.vector,
  // 		&lower_right_view.vector}, // line 1
  // 	      std::vector<gsl_vector*> {&upper_right_view.vector,
  // 		  &lower_right_view.vector}, // line 2
  // 		std::vector<gsl_vector*> {&upper_left_view.vector,
  // 		    &upper_right_view.vector}, // line 3
  // 		  std::vector<gsl_vector*> {&upper_left_view.vector,
  // 		      &lower_left_view.vector} // line 4
  // 	  };

  // 	  std::vector<double> distance_to_line {
  // 	    small_t_solution_->
  // 	      distance_from_point_to_axis_raw(lines[0][0],
  // 					      lines[0][1],
  // 					      &final_condition_view.vector),
  // 	      small_t_solution_->
  // 	      distance_from_point_to_axis_raw(lines[1][0],
  // 					      lines[1][1],
  // 					      &final_condition_view.vector),
  // 	      small_t_solution_->
  // 	      distance_from_point_to_axis_raw(lines[2][0],
  // 					      lines[2][1],
  // 					      &final_condition_view.vector),
  // 	      small_t_solution_->
  // 	      distance_from_point_to_axis_raw(lines[3][0],
  // 					      lines[3][1],
  // 					      &final_condition_view.vector)};
	  
  // 	  std::vector<unsigned> distance_to_line_indeces (4);
  // 	  unsigned n=0;
  // 	  std::generate(distance_to_line_indeces.begin(),
  // 	  		distance_to_line_indeces.end(),
  // 	  		[&n]{ return n++; });

  // 	  std::sort(distance_to_line_indeces.begin(), distance_to_line_indeces.end(),
  // 		    [&distance_to_line] (unsigned i1, unsigned i2) -> bool
  // 		    {
  // 		      return distance_to_line[i1] < distance_to_line[i2];
  // 		    });

  // 	  // REFLECTION
  // 	  small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[0]][0],
  // 	  				       lines[distance_to_line_indeces[0]][1],
  // 	  				       &final_condition_view.vector);
  // 	  small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[1]][0],
  // 	  				       lines[distance_to_line_indeces[1]][1],
  // 	  				       &final_condition_view.vector);
  // 	  small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[2]][0],
  // 	  				       lines[distance_to_line_indeces[2]][1],
  // 	  				       &final_condition_view.vector);
  // 	  small_t_solution_->reflect_point_raw(lines[distance_to_line_indeces[3]][0],
  // 	  				       lines[distance_to_line_indeces[3]][1],
  // 	  				       &final_condition_view.vector);
	  
  // 	  gsl_matrix* Rotation_matrix_inv = gsl_matrix_alloc(2,2);
  // 	  gsl_matrix_memcpy(Rotation_matrix_inv, Rotation_matrix);
  // 	  int s = 0;
  // 	  gsl_permutation * p = gsl_permutation_alloc(2);
  // 	  gsl_linalg_LU_decomp(Rotation_matrix, p, &s);
  // 	  gsl_linalg_LU_invert(Rotation_matrix, p, Rotation_matrix_inv);
  // 	  gsl_permutation_free(p);

  // 	  // C = alpha*op(A)*op(B) + beta*C
  // 	  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  // 	  		 CblasNoTrans, //op(B) = B
  // 	  		 1.0, //alpha=1
  // 	  		 Rotation_matrix_inv, //A
  // 	  		 &corner_points_transformed_view.matrix, //B
  // 	  		 0.0, //beta=0
  // 	  		 &corner_points_view.matrix); //C

  // 	  gsl_vector_view lower_left_back_view = gsl_matrix_column(&corner_points_view.matrix,0);
  // 	  gsl_vector_view lower_right_back_view = gsl_matrix_column(&corner_points_view.matrix,1);
  // 	  gsl_vector_view upper_right_back_view = gsl_matrix_column(&corner_points_view.matrix,2);
  // 	  gsl_vector_view upper_left_back_view = gsl_matrix_column(&corner_points_view.matrix,3);
  // 	  gsl_vector_view final_condition_back_view = gsl_matrix_column(&corner_points_view.matrix,4);
  // 	  gsl_vector_view point_back_view = gsl_matrix_column(&corner_points_view.matrix,5);
  // 	  gsl_vector_view initial_condition_back_view = gsl_matrix_column(&corner_points_view.matrix,6);

  // 	  printf("corner1=c(%g,%g); corner2=c(%g,%g); corner3=c(%g,%g); corner4=c(%g,%g);\n",
  // 	  	 gsl_vector_get(&lower_left_back_view.vector,0), // corner1 x
  // 	  	 gsl_vector_get(&lower_left_back_view.vector,1), // corner1 y
  // 	  	 gsl_vector_get(&lower_right_back_view.vector,0), // corner2 x
  // 	  	 gsl_vector_get(&lower_right_back_view.vector,1), // corner2 y
  // 	  	 gsl_vector_get(&upper_right_back_view.vector,0), // corner3 x
  // 	  	 gsl_vector_get(&upper_right_back_view.vector,1), // corner3 y
  // 	  	 gsl_vector_get(&upper_left_back_view.vector,0), // corner4 x
  // 		 gsl_vector_get(&upper_left_back_view.vector,1)); // corner4 y
	  
  // 	  printf("final_condition = c(%g, %g)\n",
  // 		 gsl_vector_get(&final_condition_back_view.vector,0),
  // 		 gsl_vector_get(&final_condition_back_view.vector,1));
  // 	  double out = gsl_vector_get(&final_condition_back_view.vector,0);
  // 	  out = out*(b_-a_) + a_;
  // 	  printf("out = %g\n", out);
  // 	  printf("flipped_flag = %u\n", flipped_xy_flag_);

  // 	  derivative = derivative +
  // 	    std::pow(-1, a_power)*
  // 	    std::pow(-1, b_power)*
  // 	    std::pow(-1, c_power)*
  // 	    std::pow(-1, d_power)*out;

  // 	  gsl_vector_free(scaled_input);
  // 	  // gsl_matrix_free(corner_points_transformed);
  // 	  // gsl_matrix_free(Variance);
  // 	  gsl_matrix_free(Rotation_matrix);
  // 	  gsl_matrix_free(Rotation_matrix_inv);
  //   // 	}
  //   //   }
  //   }
  // }

  // set_data(current_a,
  // 	   current_x_0,
  // 	   current_b,
  // 	   current_c,
  // 	   current_y_0,
  // 	   current_d);

  // // derivative = derivative / (16 * h^2 * h^2);
  // derivative = std::exp(std::log(std::abs(derivative)) - log(4) - 2*log(h_x) - 0*log(h_y));

  // // if (std::signbit(derivative)) {
  // //   derivative =     
  // //     -std::exp(std::log(std::abs(derivative)) - log(16) - 2*log(h_x) - 2*log(h_y));
  // // } else {
  // //   derivative =
  // //     std::exp(std::log(std::abs(derivative)) - log(16) - 2*log(h_x) - 2*log(h_y));
  // // }

  // //  derivative = derivative/(std::pow(Lx, 3) * std::pow(Ly,3));
  // //  gsl_vector_free(raw_input);
  return derivative;
}

double BivariateSolver::numerical_likelihood_first_order(const gsl_vector* raw_input,
							 double h)
{
  // there are 16 solutions to be computed
  double current_a = a_;
  double current_b = b_;
  double current_c = c_;
  double current_d = d_;
  double current_x_0 = x_0_;
  double current_y_0 = y_0_;
  // double Lx = b_ - a_;
  // double Ly = d_ - c_;
  // double current_sigma_x = sigma_x_;
  // double current_sigma_y = sigma_y_;

  // // NORMALIZING BEFORE DIFFERENTIATION START
  // double a = a_/Lx;
  // double x_0 = x_0_/Lx;
  // double b = b_/Lx;

  // double c = c_/Ly;
  // double y_0 = y_0_/Ly;
  // double d = d_/Ly;

  // double sigma_x = sigma_x_/Lx;
  // double sigma_y = sigma_y_/Ly;

  // print_diffusion_params();

  // set_diffusion_parameters_and_data(sigma_x,
  // 				    sigma_y,
  // 				    rho_,
  // 				    t_,
  // 				    a,
  // 				    x_0,
  // 				    b,
  // 				    c,
  // 				    y_0,
  // 				    d);

  // gsl_vector* input = scale_input(input);
  // // NORMALIZING BEFORE DIFFERENTIATION END

  std::vector<int> a_indeces {0,-1};
  std::vector<int> b_indeces {1, 0};
  std::vector<int> c_indeces {0,-1};
  std::vector<int> d_indeces {1, 0};

  int a_power=1;
  int b_power=1;
  int c_power=1;
  int d_power=1;

  double derivative = 0;
  double h_x = h*(b_ - a_);
  double h_y = h*(d_ - c_);
  // printf("h_x = %f, h_y = %f\n", h_x, h_y);
  // printf("%f  %f  %f\n%f  %f  %f\n\n",
	 // current_a,
	 // current_x_0,
	 // current_b,
	 // current_c,
	 // current_y_0,
	 // current_d);


  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=1; } else { a_power=0; };

    for (unsigned j=0; j<b_indeces.size(); ++j) {
      if (j==0) { b_power=1; } else { b_power=0; };

      for (unsigned k=0; k<c_indeces.size(); ++k) {
	if (k==0) { c_power=1; } else { c_power=0; };

	for (unsigned l=0; l<d_indeces.size(); ++l) {
	  if (l==0) { d_power=1; } else { d_power=0; };

	  set_data(current_a + a_indeces[i]*h_x,
		   current_x_0,
		   current_b + b_indeces[j]*h_x,
		   current_c + c_indeces[k]*h_y,
		   current_y_0,
		   current_d + d_indeces[l]*h_y);

	  double out = (*this)(raw_input);
	  // double out = analytic_solution(raw_input);
	  // printf("function value = %f\n\n", out);

	  derivative = derivative +
	    std::pow(-1, a_power)*
	    std::pow(-1, b_power)*
	    std::pow(-1, c_power)*
	    std::pow(-1, d_power)*out;

	}
      }
    }
  }

  set_data(current_a,
  	   current_x_0,
  	   current_b,
  	   current_c,
  	   current_y_0,
  	   current_d);

  // derivative = derivative / (h^2 * h^2);
  if (std::signbit(derivative)) {
    derivative =
      -std::exp(std::log(std::abs(derivative)) - 2*log(h_x) - 2*log(h_y));
  } else {
    derivative =
      std::exp(std::log(std::abs(derivative)) - 2*log(h_x) - 2*log(h_y));
  }

  // derivative = derivative/(std::pow(Lx, 3) * std::pow(Ly,3));

  //  gsl_vector_free(input);
  return derivative;
}

void BivariateSolver::set_IC_coefs()
{
  unsigned K = basis_->get_orthonormal_elements().size();

  // Assigning coefficients for IC
  gsl_vector* IC_coefs_projection = gsl_vector_alloc(IC_coefs_->size);
  for (unsigned i=0; i<IC_coefs_->size; ++i) {
    gsl_vector_set(IC_coefs_projection, i,
		   basis_->project(*small_t_solution_,
				   basis_->get_orthonormal_element(i)));
  }
  int s = 0;
  gsl_permutation * p = gsl_permutation_alloc(K);
  gsl_linalg_LU_decomp(mass_matrix_, p, &s);
  gsl_linalg_LU_solve(mass_matrix_, p,
		      IC_coefs_projection,
		      IC_coefs_);

  gsl_vector_free(IC_coefs_projection);
  gsl_permutation_free(p);
}

void BivariateSolver::set_mass_and_stiffness_matrices()
{
  // Mass matrix
  gsl_matrix_memcpy(mass_matrix_, basis_->get_mass_matrix());

  // Stiffness matrix
  const gsl_matrix* system_matrix_dx_dx = basis_->get_system_matrix_dx_dx();
  const gsl_matrix* system_matrix_dy_dy = basis_->get_system_matrix_dy_dy();
  const gsl_matrix* system_matrix_dx_dy = basis_->get_system_matrix_dx_dy();
  const gsl_matrix* system_matrix_dy_dx = basis_->get_system_matrix_dy_dx();

  gsl_matrix* left = gsl_matrix_alloc(basis_->get_orthonormal_elements().size(),
				      basis_->get_orthonormal_elements().size());
  gsl_matrix* right = gsl_matrix_alloc(basis_->get_orthonormal_elements().size(),
				       basis_->get_orthonormal_elements().size());

  gsl_matrix_memcpy(left, system_matrix_dx_dx);
  gsl_matrix_scale(left, -0.5*std::pow(sigma_x_2_,2));

  gsl_matrix_memcpy(right, system_matrix_dx_dy);
  gsl_matrix_add(right, system_matrix_dy_dx);
  gsl_matrix_scale(right, -rho_*sigma_x_2_ * sigma_y_2_*0.5);

  gsl_matrix_add(left, right);

  gsl_matrix_memcpy(right, system_matrix_dy_dy);
  gsl_matrix_scale(right, -0.5*std::pow(sigma_y_2_,2));

  gsl_matrix_add(left, right);

  gsl_matrix_memcpy(stiffness_matrix_, left);

  // double in = 0;
  // for (unsigned i=0; i<basis_->get_orthonormal_elements().size(); ++i) {
  //   for (unsigned j=0; j<basis_->get_orthonormal_elements().size(); ++j) {
  //     in = -0.5*std::pow(sigma_x_,2)*gsl_matrix_get(system_matrix_dx_dx, i, j)
  // 	+ -rho_*sigma_x_*sigma_y_*0.5*(gsl_matrix_get(system_matrix_dx_dy, i, j)+
  // 				       gsl_matrix_get(system_matrix_dy_dx, i, j))
  // 	+ -0.5*std::pow(sigma_y_,2)*gsl_matrix_get(system_matrix_dy_dy, i, j);
  //     gsl_matrix_set(stiffness_matrix_, i, j, in);
  //   }
  // }

  gsl_matrix_free(left);
  gsl_matrix_free(right);
}

void BivariateSolver::set_eval_and_evec()
{
  // System matrix: The system matrix Sys is given by M^{-1} S = Sys,
  // where M is the mass matrix and S is the stiffness_matrix_
  int s = 0;
  unsigned K = basis_->get_orthonormal_elements().size();

  gsl_permutation * p = gsl_permutation_alloc(K);
  gsl_matrix* system_matrix = gsl_matrix_alloc(K,K);
  gsl_matrix* mass_matrix_inv = gsl_matrix_alloc(K,K);
  gsl_matrix* exp_system_matrix = gsl_matrix_alloc(K,K);

  gsl_linalg_LU_decomp(mass_matrix_, p, &s);
  // First we solve S = M * Sys = L (U * Sys).
  for (unsigned k=0; k<K; ++k) {
    gsl_vector_const_view col_k_stiffness_mat =
      gsl_matrix_const_column(stiffness_matrix_, k);

    gsl_vector_view col_k_system_mat = gsl_matrix_column(system_matrix, k);
    gsl_linalg_LU_solve(mass_matrix_, p,
			&col_k_stiffness_mat.vector,
			&col_k_system_mat.vector);
  }

  gsl_matrix *evec_tr = gsl_matrix_alloc(K, K);

  gsl_eigen_symmv_workspace * w =
    gsl_eigen_symmv_alloc(K);

  gsl_eigen_symmv(system_matrix, eval_, evec_, w);

  gsl_eigen_symmv_sort(eval_, evec_,
		       GSL_EIGEN_SORT_ABS_ASC);

  gsl_permutation_free(p);
  gsl_matrix_free(system_matrix);
  gsl_matrix_free(mass_matrix_inv);
  gsl_matrix_free(exp_system_matrix);
  gsl_matrix_free(evec_tr);
  gsl_eigen_symmv_free(w);
}

void BivariateSolver::set_solution_coefs()
{
  unsigned K = basis_->get_orthonormal_elements().size();
  gsl_matrix* evec = gsl_matrix_alloc(K,K);
  gsl_matrix* evec_tr = gsl_matrix_alloc(K,K);
  gsl_matrix* exp_system_matrix = gsl_matrix_alloc(K,K);

  gsl_matrix_memcpy(evec, evec_);
  gsl_matrix_transpose_memcpy(evec_tr, evec_);

  double time_diff = 0.0;
  if ( std::signbit(t_2_-small_t_solution_->get_t()) ) {
    // printf("t_2_ <= small_t_solution_->get_t()\n");
    time_diff = 0.0;
  } else {
    time_diff = t_2_-small_t_solution_->get_t();
  }

  // evec %*% diag(eval)
  for (unsigned i=0; i<K; ++i) {
    gsl_vector_view col_i = gsl_matrix_column(evec, i);
    gsl_vector_scale(&col_i.vector, std::exp(gsl_vector_get(eval_, i)*
					     (time_diff)));
  }
  // exp_system_matrix = [evec %*% diag(exp(eval*(t-t_small)))] %*% t(evec)
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,
		 evec, evec_tr,
		 0.0,
		 exp_system_matrix);
  // exp_system_matrix %*% ic_coefs_
  gsl_blas_dsymv(CblasUpper, 1.0, exp_system_matrix, IC_coefs_,
		 0.0, solution_coefs_);

  gsl_matrix_free(evec);
  gsl_matrix_free(evec_tr);
  gsl_matrix_free(exp_system_matrix);
}

void BivariateSolver::set_scaled_data()
{
  // STEP 1 SHIFTING
  double x_0_1 = x_0_ - a_;
  double b_1 = b_ - a_;
  double a_1 = a_ - a_;

  double y_0_1 = y_0_ - c_;
  double c_1 = c_ - c_;
  double d_1 = d_ - c_;

  // STEP 2 SCALING
  double Lx_2 = b_1 - a_1;
  x_0_2_ =  x_0_1 / Lx_2;
  a_2_ = a_1 / Lx_2;
  b_2_ = b_1 / Lx_2;
  sigma_x_2_ = sigma_x_ / Lx_2;

  double Ly_2 = d_1 - c_1;
  y_0_2_ =  y_0_1 / Ly_2;
  c_2_ = c_1 / Ly_2;
  d_2_ = d_1 / Ly_2;
  sigma_y_2_ = sigma_y_ / Ly_2;

  // STEP 3 RE-SCALING
  double sigma_x_3 = 1.0;
  double sigma_y_3 = sigma_y_2_ / sigma_x_2_;
  //
  double x_0_3 = x_0_2_;
  double y_0_3 = y_0_2_;
  t_2_ = t_ * (sigma_x_2_*sigma_x_2_);
  flipped_xy_flag_ = 0;

  if (sigma_x_2_ < sigma_y_2_) {
    //  printf("sigma_x_2_ < sigma_y_2_\n");
    sigma_x_3 = 1.0;
    sigma_y_3 = sigma_x_2_ / sigma_y_2_;
    //
    x_0_3 = y_0_2_;
    y_0_3 = x_0_2_;
    t_2_ = t_ * (sigma_y_2_*sigma_y_2_);
    flipped_xy_flag_ = 1;
  }

  sigma_x_2_ = sigma_x_3;
  sigma_y_2_ = sigma_y_3;
  x_0_2_ = x_0_3;
  y_0_2_ = y_0_3;

  // printf("sigma_x_ = %f, sigma_y_ = %f, sigma_x_2_^2 = %f, sigma_y_2_^2 = %f, x_0_2_ = %f, y_0_2_ = %f, t_2_ = %f\n",
  // 	 sigma_x_,
  // 	 sigma_y_,
  // 	 sigma_x_2_*sigma_x_2_,
  // 	 sigma_y_2_*sigma_y_2_,
  // 	 x_0_2_, y_0_2_, t_2_);
}

double BivariateSolver::extrapolate_t_direction(const double likelihood_upper_bound,
						const double t_lower_bound_in,
						const double t_2_current,
						const double t_current,
						const bool flipped_xy_flag,
						const gsl_vector* input,
						const double h)
{
  double likelihood = -1.0;
  double t_lower_bound = t_lower_bound_in;

  if (!flipped_xy_flag) {
    t_ = t_lower_bound * ((b_ - a_)/sigma_x_) * ((b_ - a_)/sigma_x_);
  } else {
    t_ = t_lower_bound * ((d_ - c_)/sigma_y_) * ((d_ - c_)/sigma_y_);
  }
  set_scaled_data();

  // std::cout << "In extrapolate_t_direction" << std::endl;
  // print_diffusion_params();

  double f1 = numerical_likelihood(input, h);
  double x1 = t_2_;

  while ( (std::signbit(f1) ||
	   (f1-likelihood_upper_bound > std::numeric_limits<double>::epsilon())) &&
	  t_2_ < 3.0) {
    t_lower_bound = t_lower_bound + 0.1;

    if (!flipped_xy_flag) {
      t_ = t_lower_bound * ((b_ - a_)/sigma_x_) * ((b_ - a_)/sigma_x_);
    } else {
      t_ = t_lower_bound * ((d_ - c_)/sigma_y_) * ((d_ - c_)/sigma_y_);
    }
    set_scaled_data();

    f1 = numerical_likelihood(input, h);
    x1 = t_2_;
  }


  if (!flipped_xy_flag) {
    t_ = (t_lower_bound + 0.03) * ((b_ - a_)/sigma_x_) * ((b_ - a_)/sigma_x_);
  } else {
    t_ = (t_lower_bound + 0.03) * ((d_ - c_)/sigma_y_) * ((d_ - c_)/sigma_y_);
  }
  set_scaled_data();
  double f2 = numerical_likelihood(input, h);
  double x2 = t_2_;

  if (!flipped_xy_flag) {
    t_ = (t_lower_bound + 0.06) * ((b_ - a_)/sigma_x_) * ((b_ - a_)/sigma_x_);
  } else {
    t_ = (t_lower_bound + 0.06) * ((d_ - c_)/sigma_y_) * ((d_ - c_)/sigma_y_);
  }
  set_scaled_data();
  double f3 = numerical_likelihood(input, h);
  double x3 = t_2_;

  double neg_alpha_min_1 = (log(f2/f3) - log(f1/f2)/(1/x1 - 1/x2)*(1/x2 - 1/x3)) /
		     ( log(x2/x3) - log(x1/x2)*(1/x2-1/x3)/(1/x1 - 1/x2)  );
  double alpha = -1.0*neg_alpha_min_1 + 1;
  double beta = -1.0*(log(f1/f2) + (alpha+1)*log(x1/x2))/(1/x1 - 1/x2);
  double CC = exp(log(f1) + (alpha+1)*log(x1) + beta/x1);

  if (!std::signbit(alpha) && !std::signbit(beta)) {
    if (log(likelihood_upper_bound) + (alpha+1)*(log(beta) - log(alpha+1) + 1) >= log(CC)) {

      likelihood = CC*std::pow(t_2_current, -1.0*(alpha+1.0))*
	exp(-1.0*beta/t_2_current);
    } else {
      double log_CC = log(likelihood_upper_bound) + (alpha+1)*(log(beta) - log(alpha+1) + 1);
      CC = exp(log_CC);

      likelihood = CC*std::pow(t_2_current, -1.0*(alpha+1.0))*
	exp(-1.0*beta/t_2_current);
    }
  } else {
    if ( (f1-1) < std::numeric_limits<double>::epsilon() &&
	 std::signbit(f1-1.0) ) {
      beta = -1.0*log(f1)*x1;
      likelihood = exp(-beta/t_2_current);
    } else {
      beta = -1.0*log(1.1)*x1;
      likelihood = exp(-beta/t_2_current);
    }
  }

  // RESETTING t_2_;
  t_ = t_current;
  set_scaled_data();

  // printf("function.vals = c(%f, %f, %f);\n",
	 // f1,f2,f3);
  // printf("xs = c(%f, %f, %f);\n",
	 // x1,x2,x3);
  // printf("params = c(%f,%f,%f);\n",
	 // alpha,beta,CC);
  // printf("t.2.current = %f;\n", t_2_);
  // printf("t.2.current input = %f;\n", t_2_current);
  // LAST TWO SHOULD BE THE SAME!!

  printf("\nalpha = %f; beta = %f; f1 = %f; f2 = %f; f3 = %f; x1 = %f; x2 = %f; x3 = %f; CC = %f;\n",
	 alpha, beta,
	 f1, f2, f3,
	 x1, x2, x3,
	 CC);
  return likelihood;
}


double BivariateSolver::extrapolate_sigma_y_direction(const double likelihood_upper_bound,
						      const double sigma_y_2_lower_bound_in,
						      const double sigma_y_2_current,
						      const double sigma_x_current,
						      const double sigma_y_current,
						      const bool flipped_xy_flag,
						      const gsl_vector* input,
						      const double h)
{
  double likelihood = -1.0;
  double sigma_y_2_lower_bound = sigma_y_2_lower_bound_in;

  double x1 = sigma_y_2_lower_bound;

  // THIS IS WHAT WE'RE DOING:  sigma_y_2_ = sigma_y_2_lower_bound;
  if (!flipped_xy_flag) {
    sigma_y_ = sigma_y_2_lower_bound * sigma_x_ * (d_ - c_)/(b_ - a_);
  } else {
    sigma_x_ = sigma_y_2_lower_bound * sigma_y_ * (b_ - a_)/(d_ - c_);
  }

  // printf("sigma_y_2_lower_bound = %f, x1 = %f\n",
	 // sigma_y_2_lower_bound, x1);

  set_diffusion_parameters(sigma_x_,
			   sigma_y_,
			   rho_);

  double f1 = numerical_likelihood(input, h);

  while ( (std::signbit(f1) ||
	   (f1 - likelihood_upper_bound > std::numeric_limits<double>::epsilon()))
	  && sigma_y_2_lower_bound < 0.9) {
    sigma_y_2_lower_bound = sigma_y_2_lower_bound + 0.1;

    if (!flipped_xy_flag) {
      sigma_y_ = sigma_y_2_lower_bound * sigma_x_ * (d_ - c_)/(b_ - a_);
    } else {
      sigma_x_ = sigma_y_2_lower_bound * sigma_y_ * (b_ - a_)/(d_ - c_);
    }

    set_diffusion_parameters(sigma_x_,
			     sigma_y_,
			     rho_);
    f1 = numerical_likelihood(input, h);
    x1 = sigma_y_2_;
  }

  // printf("sigma_y_2_lower_bound = %f, sigma_y_2_ = %f, x1 = %f\n",
	 // sigma_y_2_lower_bound, sigma_y_2_, x1);

  if (!flipped_xy_flag) {
    sigma_y_ = (sigma_y_2_lower_bound + 1.0/2.0 * 1.0/10.0 * (1-sigma_y_2_lower_bound)) *
      sigma_x_ * (d_ - c_)/(b_ - a_);
  } else {
    sigma_x_ = (sigma_y_2_lower_bound + 1.0/2.0 * 1.0/10.0 * (1-sigma_y_2_lower_bound)) *
      sigma_y_ * (b_ - a_)/(d_ - c_);
  }
  set_diffusion_parameters(sigma_x_,
			   sigma_y_,
			   rho_);
  double f2 = numerical_likelihood(input, h);
  double x2 = sigma_y_2_;

  // printf("sigma_y_2_lower_bound = %f, sigma_y_2_ = %f, x2 = %f\n",
  //sigma_y_2_lower_bound, sigma_y_2_, x2);

  if (!flipped_xy_flag) {
    sigma_y_ = (sigma_y_2_lower_bound + 1*1.0/10.0*(1-sigma_y_2_lower_bound)) *
      sigma_x_ * (d_ - c_)/(b_ - a_);
  } else {
    sigma_x_ = (sigma_y_2_lower_bound + 1*1.0/10.0*(1-sigma_y_2_lower_bound)) *
      sigma_y_ * (b_ - a_)/(d_ - c_);
  }
  set_diffusion_parameters(sigma_x_,
			   sigma_y_,
			   rho_);
  double f3 = numerical_likelihood(input, h);
  double x3 = sigma_y_2_;

  // printf("sigma_y_2_lower_bound = %f, sigma_y_2_ = %f, x3 = %f\n",
//	 sigma_y_2_lower_bound, sigma_y_2_, x3);

  double neg_alpha_min_1 = (log(f2/f3) - log(f1/f2)/(1/x1 - 1/x2)*(1/x2 - 1/x3)) /
    ( log(x2/x3) - log(x1/x2)*(1/x2-1/x3)/(1/x1 - 1/x2)  );
  double alpha = -1.0*neg_alpha_min_1 + 1;
  double beta = -1.0*(log(f1/f2) + (alpha+1)*log(x1/x2))/(1/x1 - 1/x2);
  double CC = exp(log(f1) + (alpha+1)*log(x1) + beta/x1);

  if (!std::signbit(alpha) && !std::signbit(beta)) {
    if (log(likelihood_upper_bound) + (alpha+1)*(log(beta) - log(alpha+1) + 1) >= log(CC)) {

      likelihood = CC*std::pow(sigma_y_2_current, -1.0*(alpha+1.0))*
	exp(-1.0*beta/sigma_y_2_current);
    } else {
      double log_CC = log(likelihood_upper_bound) + (alpha+1)*(log(beta) - log(alpha+1) + 1);
      CC = exp(log_CC);

      likelihood = CC*std::pow(sigma_y_2_current, -1.0*(alpha+1.0))*
	exp(-1.0*beta/sigma_y_2_current);
    }
  } else {
    if ( (f1-1.0) < std::numeric_limits<double>::epsilon() &&
	 std::signbit(f1-1.0) ) {
      beta = -1.0*log(f1)*x1;
      likelihood = exp(-beta/sigma_y_2_current);
    } else {
      beta = -1.0*log(1.1)*x1;
      likelihood = exp(-beta/sigma_y_2_current);
    }
  }

  // RESETTING sigma_x_, sigma_y_ //
  sigma_x_ = sigma_x_current;
  sigma_y_ = sigma_y_current;
  set_diffusion_parameters(sigma_x_,
			   sigma_y_,
			   rho_);
  // //

  // printf("function.vals = c(%f, %f, %f);\n",
	 // f1,f2,f3);
  // printf("xs = c(%f, %f, %f);\n",
	 // x1,x2,x3);
  // printf("params = c(%f,%f,%f);\n",
	 // alpha,beta,CC);
  // printf("sigma.2.y.current = %f;\n", sigma_y_2_);
  // printf("sigma.2.y.current input = %f;\n", sigma_y_2_current);
  // LAST TWO SHOULD BE THE SAME!!
  return likelihood;
}

void BivariateSolver::print_diffusion_params() const
{
  // printf("Lx = %f,\n Ly = %f,\n sigma_x_ = %f,\n sigma_y_ = %f,\n t_ = %f,\n\n",
  //	 b_-a_, d_-c_, sigma_x_, sigma_y_, t_);
}
