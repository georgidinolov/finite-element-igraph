#include <algorithm>
#include "BivariateSolver.hpp"
#include <cmath>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <iostream>
#include <limits>
#include <string>

BivariateImageWithTime::BivariateImageWithTime()
  : position_(gsl_vector_alloc(2)),
    t_(0.0),
    mult_factor_(1.0)
{}

BivariateImageWithTime::BivariateImageWithTime(const BivariateImageWithTime& image_with_time)
  : position_(gsl_vector_alloc(2)),
    t_(0.0),
    mult_factor_(1.0)
{
  gsl_vector_memcpy(position_, image_with_time.get_position());
  t_ = image_with_time.get_t();
  mult_factor_ = image_with_time.get_mult_factor();
}

BivariateImageWithTime& BivariateImageWithTime::operator=(const BivariateImageWithTime& rhs)
{
  if (this==&rhs) {
    return *this;
  } else {
    gsl_vector_memcpy(position_, rhs.get_position());
    t_ = rhs.get_t();
    mult_factor_ = rhs.get_mult_factor();
    return *this;
  }
}

BivariateImageWithTime::BivariateImageWithTime(const gsl_vector* position,
					       double time,
					       double mult_factor)
  : position_(gsl_vector_calloc(2)),
    t_(time),
    mult_factor_(mult_factor)
{
  gsl_vector_memcpy(position_, position);
}

BivariateImageWithTime::~BivariateImageWithTime()
{
  gsl_vector_free(position_);
}

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

void BivariateSolver::set_data_for_small_t(double a,
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
  // std::cout << "## analytic_solution start ##" << std::endl;
  // std::cout << "t_=" << t_ << ","
  // 	    << "sigma_x_=" << sigma_x_ << ","
  // 	    << "sigma_y_=" << sigma_y_ << "\n"
  // 	    << "(a_=" << a_ << "," << "x_0_=" << x_0_ << "," << "x_t_=" << gsl_vector_get(input,0) << "," << "b_=" << b_ << ")\n"
  //   	    << "(c_=" << c_ << "," << "y_0_=" << y_0_ << "," << "y_t_=" << gsl_vector_get(input,1) << "," << "d_=" << d_ << ")\n";
  // std::cout << "## analytic_solution end ##" << std::endl;;
  
  int little_n = 10000;

  std::vector<double> d1x (2*little_n + 1);
  std::vector<double> d2x (2*little_n + 1);

  std::vector<double> d1y (2*little_n + 1);
  std::vector<double> d2y (2*little_n + 1);

  double sum_x = 0.0;
  double sum_y = 0.0;

  for (int i=0; i<2*little_n+1; ++i) {
    int n = i - little_n;

    d1x[i] = std::pow(gsl_vector_get(input,0) - x_0_2_ - 2.0*n*(b_2_-a_2_), 2) /
      (2.0 * std::pow(sigma_x_2_, 2) * t_2_);
    d2x[i] = std::pow(gsl_vector_get(input,0) + x_0_2_ - 2.0*a_2_ - 2.0*n*(b_2_-a_2_), 2) /
      (2.0 * std::pow(sigma_x_2_, 2) * t_2_);

    d1y[i] = std::pow(gsl_vector_get(input,1) - y_0_2_ - 2.0*n*(d_2_-c_2_), 2) /
      (2.0 * std::pow(sigma_y_2_, 2) * t_2_);
    d2y[i] = std::pow(gsl_vector_get(input,1) + y_0_2_ - 2.0*c_2_ - 2.0*n*(d_2_-c_2_), 2) /
      (2.0 * std::pow(sigma_y_2_, 2) * t_2_);

    sum_x = sum_x +
      (std::exp(-d1x[i]) - std::exp(-d2x[i]));

    sum_y = sum_y +
      (std::exp(-d1y[i]) - std::exp(-d2y[i]));

    // std::cout << "d1x[" << n << "] = " << d1x[i] << " ";
  }
  // std::cout << std::endl;

  double out_x = (1.0/std::sqrt(2.0*M_PI*std::pow(sigma_x_2_,2)*t_2_))*
    sum_x;

  double out_y = (1.0/std::sqrt(2.0*M_PI*std::pow(sigma_y_2_,2)*t_2_))*
    sum_y;

  double out = out_x*out_y;
  // std::cout << "t = " << t_ << ";" << std::endl;
  // std::cout << "ax = " << a_ << ";" << std::endl;
  // std::cout << "bx = " << b_ << ";" << std::endl;
  // std::cout << "x.ic = " << x_0_ << ";" << std::endl;
  // std::cout << "x.fc = " << gsl_vector_get(input,0) << ";" << std::endl;
  // std::cout << "sigma.2.x = " << std::pow(sigma_x_, 2) << ";" << std::endl;
  // std::cout << "out_x = " << out_x << ";" << std::endl;

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
  // std::cout << "## analytic_likelihood start ##" << std::endl;
  // std::cout << "t_=" << t_ << ","
  // 	    << "sigma_x_=" << sigma_x_ << ","
  // 	    << "sigma_y_=" << sigma_y_ << "\n"
  // 	    << "(a_=" << a_ << "," << "x_0_=" << x_0_ << "," << "x_t_=" << gsl_vector_get(input,0) << "," << "b_=" << b_ << ")\n"
  //   	    << "(c_=" << c_ << "," << "y_0_=" << y_0_ << "," << "y_t_=" << gsl_vector_get(input,1) << "," << "d_=" << d_ << ")\n";
  // std::cout << "## analytic_likelihood end ##" << std::endl;;
    
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


double BivariateSolver::analytic_likelihood_ax(const gsl_vector* input,
					       int little_n)
{
  std::vector<double> d1x (2*little_n + 1);
  std::vector<double> d2x (2*little_n + 1);

  std::vector<double> d1y (2*little_n + 1);
  std::vector<double> d2y (2*little_n + 1);

  double sum_x = 0.0;
  double sum_y = 0.0;

  for (int i=0; i<2*little_n+1; ++i) {
    int n = i - little_n;

    d1x[i] = std::pow(gsl_vector_get(input,0) - x_0_2_ - 2.0*n*(b_2_-a_2_), 2) /
      (2.0 * std::pow(sigma_x_2_, 2) * t_2_);
    d2x[i] = std::pow(gsl_vector_get(input,0) + x_0_2_ - 2.0*a_2_ - 2.0*n*(b_2_-a_2_), 2) /
      (2.0 * std::pow(sigma_x_2_, 2) * t_2_);

    d1y[i] = std::pow(gsl_vector_get(input,1) - y_0_2_ - 2.0*n*(d_2_-c_2_), 2) /
      (2.0 * std::pow(sigma_y_2_, 2) * t_2_);
    d2y[i] = std::pow(gsl_vector_get(input,1) + y_0_2_ - 2.0*c_2_ - 2.0*n*(d_2_-c_2_), 2) /
      (2.0 * std::pow(sigma_y_2_, 2) * t_2_);

    sum_x = sum_x +
      (-1.0*std::exp(-d1x[i])*(gsl_vector_get(input,0) - x_0_2_ - 2.0*n*(b_2_-a_2_))/(std::pow(sigma_x_2_, 2) * t_2_)*(2.0*n) -
       //
       -1.0*std::exp(-d2x[i])*(gsl_vector_get(input,0) + x_0_2_ - 2.0*a_2_ - 2.0*n*(b_2_-a_2_))/(std::pow(sigma_x_2_, 2) * t_2_)*(-2.0+2.0*n));

    sum_y = sum_y +
      (std::exp(-d1y[i]) - std::exp(-d2y[i]));

    // std::cout << "d1x[" << n << "] = " << d1x[i] << " ";
  }
  // std::cout << std::endl;

  double out_x = (1.0/std::sqrt(2.0*M_PI*std::pow(sigma_x_2_,2)*t_2_))*
    sum_x;

  double out_y = (1.0/std::sqrt(2.0*M_PI*std::pow(sigma_y_2_,2)*t_2_))*
    sum_y;

  double out = out_x*out_y;
  // std::cout << "t = " << t_ << ";" << std::endl;
  // std::cout << "ax = " << a_ << ";" << std::endl;
  // std::cout << "bx = " << b_ << ";" << std::endl;
  // std::cout << "x.ic = " << x_0_ << ";" << std::endl;
  // std::cout << "x.fc = " << gsl_vector_get(input,0) << ";" << std::endl;
  // std::cout << "sigma.2.x = " << std::pow(sigma_x_, 2) << ";" << std::endl;
  // std::cout << "out_x = " << out_x << ";" << std::endl;

  return out;
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

double BivariateSolver::numerical_likelihood_first_order_small_t(const gsl_vector* raw_input,
								 double small_t,
								 double h)
{
  printf("in NUMERICAL_likelihood_first_order_small_t\n");
  // there are 16 solutions to be computed
  double current_a = a_;
  double current_b = b_;
  double current_c = c_;
  double current_d = d_;
  double current_x_0 = x_0_;
  double current_y_0 = y_0_;

  std::vector<int> a_indeces {1,-1};
  std::vector<int> b_indeces {1, 0};
  std::vector<int> c_indeces {0,-1};
  std::vector<int> d_indeces {1, 0};

  int a_power=1;
  int b_power=1;
  int c_power=1;
  int d_power=1;

  double derivative = 0;
  double derivative_with_sol = 0;
  double h_x = h*(b_ - a_);
  double h_y = h*(d_ - c_);

  double log_CC = -1.0*(log(2.0)+log(t_)+log(1-rho_*rho_));

  gsl_matrix* cov_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_set(cov_matrix, 0,0,
		 sigma_x_*sigma_x_*t_);
  gsl_matrix_set(cov_matrix, 0,1,
		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,0,
		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,1,
		 sigma_y_*sigma_y_*t_);

  std::vector<BivariateImageWithTime> positions = small_t_image_positions_ax();
  std::cout << "positions.size()=" << positions.size() << std::endl;

  MultivariateNormal mvtnorm = MultivariateNormal();
  double before_small_t = mvtnorm.dmvnorm(2,
					      positions[0].get_position(),
					      raw_input,
					      cov_matrix);
  double log_before_small_t = mvtnorm.dmvnorm_log(2,
					      positions[0].get_position(),
					      raw_input,
					      cov_matrix);
  double log_before_small_t_2 =
    -1.0*(log(2*M_PI) + log(sigma_x_) + log(sigma_y_) + log(t_) + 0.5*log(1-rho_*rho_))
    -1.0/(t_*2*(1-rho_*rho_))*
    (
     std::pow((gsl_vector_get(raw_input,0)-gsl_vector_get(positions[0].get_position(),0)),2)/
     std::pow(sigma_x_,2) +
     // //
     std::pow((gsl_vector_get(raw_input,1)-gsl_vector_get(positions[0].get_position(),1)),2)/
     std::pow(sigma_y_,2) +
     // //
     -2.0*rho_*(gsl_vector_get(raw_input,0)-gsl_vector_get(positions[0].get_position(),0))*
     (gsl_vector_get(raw_input,1)-gsl_vector_get(positions[0].get_position(),1))/
     (sigma_x_*sigma_y_)
     );
      
  std::cout << "before_small_t=" << before_small_t << std::endl;
  std::cout << "log_before_small_t=" << log_before_small_t << std::endl;
  std::cout << "log_before_small_t_2=" << log_before_small_t_2 << std::endl;
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=1; } else { a_power=0; };

  	  set_data(current_a + a_indeces[i]*h_x,
  		   current_x_0,
  		   current_b,
  		   current_c,
  		   current_y_0,
  		   current_d);

  	  positions = small_t_image_positions_ax();
	  double x_0_star = gsl_vector_get(positions[0].get_position(),0)*(b_-a_) + a_indeces[i]*h_x;
	  double y_0_star = gsl_vector_get(positions[0].get_position(),1)*(d_-c_);
	  double x = gsl_vector_get(raw_input, 0);
	  double y = gsl_vector_get(raw_input, 1);
	  gsl_vector* original_scale_position = gsl_vector_alloc(2);
	  gsl_vector_set(original_scale_position,0,x_0_star);
	  gsl_vector_set(original_scale_position,1,y_0_star);
	  gsl_vector* scaled_input = scale_input(raw_input);

  	  gsl_matrix_set(cov_matrix, 0,0,
  			 sigma_x_*sigma_x_*t_);
  	  gsl_matrix_set(cov_matrix, 0,1,
  			 rho_*sigma_x_*sigma_y_*t_);
  	  gsl_matrix_set(cov_matrix, 1,0,
  			 rho_*sigma_x_*sigma_y_*t_);
  	  gsl_matrix_set(cov_matrix, 1,1,
  			 sigma_y_*sigma_y_*t_);

	  double polynomial = std::pow((x-x_0_star)/sigma_x_,2) +
	    std::pow((y-y_0_star)/sigma_y_,2) -
	    2*rho_/(sigma_x_*sigma_y_)*(x-x_0_star)*(y-y_0_star);
	  
  	  double smallt = mvtnorm.dmvnorm_log(2,
  					      original_scale_position,
  					      raw_input,
  					      cov_matrix);
	  double analytic_sol = analytic_solution(scaled_input);

	  std::cout << "a_=" << a_ << ", x_0_=" << x_0_ << ", x_0_reflected=" << (gsl_vector_get(positions[0].get_position(),0)*(b_-a_) + a_indeces[i]*h_x)
		    << ", (b_-a_)=" << b_-a_
		    << ", a_2_=" << a_2_ << ", x_0_2_=" << x_0_2_ << ", (b_2_-a_2_)=" << b_2_-a_2_
		    << ", t_2_min=" << positions[0].get_t()
		    << ", t_min=" << scale_back_t(positions[0].get_t())
		    << ", log_analytic_sol=" << log(analytic_sol) - log(b_-a_) - log(d_-c_)
		    << ", analytic_sol=" << exp(log(analytic_sol) - log(b_-a_) - log(d_-c_))
		    << "\n";

	  derivative = derivative +
	    // (gsl_vector_get(positions[0].get_position(),0)*(b_-a_) + a_indeces[i]*h_x)*
	    std::exp(smallt-log_before_small_t-log_CC)*
	    //polynomial*
  	    std::pow(-1, a_power);

	  derivative_with_sol = derivative_with_sol +
	    exp(log(analytic_solution(scaled_input)) - log(b_-a_) - log(d_-c_))*
  	    std::pow(-1, a_power);

	  gsl_vector_free(original_scale_position);
	  gsl_vector_free(scaled_input);
  }
  std::cout << std::endl;
  	// }
   //   }
   // }
  gsl_matrix_free(cov_matrix);
  
  set_data(current_a,
  	   current_x_0,
  	   current_b,
  	   current_c,
  	   current_y_0,
  	   current_d);
  positions = small_t_image_positions_ax();

  double x = gsl_vector_get(raw_input,0);
  double y = gsl_vector_get(raw_input,1);
  double x_0 = gsl_vector_get(positions[0].get_position(),0);
  double y_0 = gsl_vector_get(positions[0].get_position(),1);
  
  std::cout << "The analytic deriv is = "
	    << -1.0*std::exp(log_before_small_t + log_CC + log(std::abs(2*(x-x_0)/std::pow(sigma_x_,2)*(-2) - 2*rho_/(sigma_x_*sigma_y_)*(-2)*(y-y_0))))
	    << std::endl;
  derivative = derivative/(2*h_x)* exp(log_before_small_t + log_CC);
  std::cout << "The solution with numerical polynomail deriv is = " << derivative << std::endl;
  derivative_with_sol = derivative_with_sol/(2*h_x);
  std::cout << "The deriv with solution is = "
	    << derivative_with_sol << std::endl;
  std::cout << "The analytic deriv from _ax() is = " << analytic_likelihood_ax(raw_input, 10000) << std::endl;

  return derivative;
}

double BivariateSolver::numerical_likelihood_first_order_small_t_ax(const gsl_vector* raw_input,
								    double small_t,
								    double h)
{
  printf("in NUMERICAL_likelihood_first_order_small_t\n");
  // there are 16 solutions to be computed
  double current_a = a_;
  double current_b = b_;
  double current_c = c_;
  double current_d = d_;
  double current_x_0 = x_0_;
  double current_y_0 = y_0_;

  std::vector<int> a_indeces {1,-1};
  std::vector<int> b_indeces {1, 0};
  std::vector<int> c_indeces {0,-1};
  std::vector<int> d_indeces {1, 0};

  int a_power=1;
  int b_power=1;
  int c_power=1;
  int d_power=1;

  double derivative = 0;
  double derivative_with_sol = 0;
  double h_x = h*(b_ - a_);
  double h_y = h*(d_ - c_);

  double log_CC = -1.0*(log(2.0)+log(t_)+log(1-rho_*rho_));

  gsl_matrix* cov_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_set(cov_matrix, 0,0,
		 sigma_x_*sigma_x_*t_);
  gsl_matrix_set(cov_matrix, 0,1,
		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,0,
		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,1,
		 sigma_y_*sigma_y_*t_);

  std::vector<BivariateImageWithTime> positions = small_t_image_positions_ax();
  std::cout << "positions.size()=" << positions.size() << std::endl;

  MultivariateNormal mvtnorm = MultivariateNormal();
  double before_small_t = mvtnorm.dmvnorm(2,
					      positions[0].get_position(),
					      raw_input,
					      cov_matrix);
  double log_before_small_t = mvtnorm.dmvnorm_log(2,
					      positions[0].get_position(),
					      raw_input,
					      cov_matrix);
  double log_before_small_t_2 =
    -1.0*(log(2*M_PI) + log(sigma_x_) + log(sigma_y_) + log(t_) + 0.5*log(1-rho_*rho_))
    -1.0/(t_*2*(1-rho_*rho_))*
    (
     std::pow((gsl_vector_get(raw_input,0)-gsl_vector_get(positions[0].get_position(),0)),2)/
     std::pow(sigma_x_,2) +
     // //
     std::pow((gsl_vector_get(raw_input,1)-gsl_vector_get(positions[0].get_position(),1)),2)/
     std::pow(sigma_y_,2) +
     // //
     -2.0*rho_*(gsl_vector_get(raw_input,0)-gsl_vector_get(positions[0].get_position(),0))*
     (gsl_vector_get(raw_input,1)-gsl_vector_get(positions[0].get_position(),1))/
     (sigma_x_*sigma_y_)
     );
      
  std::cout << "before_small_t=" << before_small_t << std::endl;
  std::cout << "log_before_small_t=" << log_before_small_t << std::endl;
  std::cout << "log_before_small_t_2=" << log_before_small_t_2 << std::endl;
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=1; } else { a_power=0; };

  	  set_data(current_a + a_indeces[i]*h_x,
  		   current_x_0,
  		   current_b,
  		   current_c,
  		   current_y_0,
  		   current_d);

  	  positions = small_t_image_positions_ax();
	  double x_0_star = gsl_vector_get(positions[0].get_position(),0)*(b_-a_) + a_indeces[i]*h_x;
	  double y_0_star = gsl_vector_get(positions[0].get_position(),1)*(d_-c_);
	  double x = gsl_vector_get(raw_input, 0);
	  double y = gsl_vector_get(raw_input, 1);
	  gsl_vector* original_scale_position = gsl_vector_alloc(2);
	  gsl_vector_set(original_scale_position,0,x_0_star);
	  gsl_vector_set(original_scale_position,1,y_0_star);
	  gsl_vector* scaled_input = scale_input(raw_input);

  	  gsl_matrix_set(cov_matrix, 0,0,
  			 sigma_x_*sigma_x_*t_);
  	  gsl_matrix_set(cov_matrix, 0,1,
  			 rho_*sigma_x_*sigma_y_*t_);
  	  gsl_matrix_set(cov_matrix, 1,0,
  			 rho_*sigma_x_*sigma_y_*t_);
  	  gsl_matrix_set(cov_matrix, 1,1,
  			 sigma_y_*sigma_y_*t_);

	  double polynomial = std::pow((x-x_0_star)/sigma_x_,2) +
	    std::pow((y-y_0_star)/sigma_y_,2) -
	    2*rho_/(sigma_x_*sigma_y_)*(x-x_0_star)*(y-y_0_star);
	  
  	  double smallt = mvtnorm.dmvnorm_log(2,
  					      original_scale_position,
  					      raw_input,
  					      cov_matrix);
	  double analytic_sol = analytic_solution(scaled_input);

	  std::cout << "a_=" << a_ << ", x_0_=" << x_0_ << ", x_0_reflected=" << (gsl_vector_get(positions[0].get_position(),0)*(b_-a_) + a_indeces[i]*h_x)
		    << ", (b_-a_)=" << b_-a_
		    << ", a_2_=" << a_2_ << ", x_0_2_=" << x_0_2_ << ", (b_2_-a_2_)=" << b_2_-a_2_
		    << ", t_2_min=" << positions[0].get_t()
		    << ", t_min=" << scale_back_t(positions[0].get_t())
		    << ", log_analytic_sol=" << log(analytic_sol) - log(b_-a_) - log(d_-c_)
		    << ", analytic_sol=" << exp(log(analytic_sol) - log(b_-a_) - log(d_-c_))
		    << "\n";

	  derivative = derivative +
	    // (gsl_vector_get(positions[0].get_position(),0)*(b_-a_) + a_indeces[i]*h_x)*
	    std::exp(smallt-log_before_small_t-log_CC)*
	    //polynomial*
  	    std::pow(-1, a_power);

	  derivative_with_sol = derivative_with_sol +
	    exp(log(analytic_solution(scaled_input)) - log(b_-a_) - log(d_-c_))*
  	    std::pow(-1, a_power);

	  gsl_vector_free(original_scale_position);
	  gsl_vector_free(scaled_input);
  }
  std::cout << std::endl;
  	// }
   //   }
   // }
  gsl_matrix_free(cov_matrix);
  
  set_data(current_a,
  	   current_x_0,
  	   current_b,
  	   current_c,
  	   current_y_0,
  	   current_d);
  positions = small_t_image_positions_ax();

  double x = gsl_vector_get(raw_input,0);
  double y = gsl_vector_get(raw_input,1);
  double x_0 = gsl_vector_get(positions[0].get_position(),0);
  double y_0 = gsl_vector_get(positions[0].get_position(),1);
  
  std::cout << "The analytic deriv is = "
	    << -1.0*std::exp(log_before_small_t + log_CC + log(std::abs(2*(x-x_0)/std::pow(sigma_x_,2)*(-2) - 2*rho_/(sigma_x_*sigma_y_)*(-2)*(y-y_0))))
	    << std::endl;
  derivative = derivative/(2*h_x)* exp(log_before_small_t + log_CC);
  std::cout << "The solution with numerical polynomail deriv is = " << derivative << std::endl;
  derivative_with_sol = derivative_with_sol/(2*h_x);
  std::cout << "The deriv with solution is = "
	    << derivative_with_sol << std::endl;
  std::cout << "The analytic deriv from _ax() is = " << analytic_likelihood_ax(raw_input, 10000) << std::endl;

  return derivative;
}

double BivariateSolver::numerical_likelihood_first_order_small_t_ax_bx(const gsl_vector* raw_input,
								       double small_t,
								       double h)
{
  printf("in NUMERICAL_likelihood_first_order_small_t\n");
  // there are 16 solutions to be computed
  double current_a = a_;
  double current_b = b_;
  double current_c = c_;
  double current_d = d_;
  double current_x_0 = x_0_;
  double current_y_0 = y_0_;

  std::vector<int> a_indeces {0,-1};
  std::vector<int> b_indeces {1, 0};
  std::vector<int> c_indeces {0,-1};
  std::vector<int> d_indeces {1, 0};

  int a_power=1;
  int b_power=1;
  int c_power=1;
  int d_power=1;

  double derivative = 0;
  double derivative_with_sol = 0;
  double h_x = h*(b_ - a_);
  double h_y = h*(d_ - c_);

  double log_CC = -1.0*(log(2.0)+log(t_)+log(1-rho_*rho_));

  gsl_matrix* cov_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_set(cov_matrix, 0,0,
		 sigma_x_*sigma_x_*t_);
  gsl_matrix_set(cov_matrix, 0,1,
		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,0,
		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,1,
		 sigma_y_*sigma_y_*t_);

  std::vector<BivariateImageWithTime> positions = small_t_image_positions_ax_bx();
  std::cout << "positions.size()=" << positions.size() << std::endl;

  MultivariateNormal mvtnorm = MultivariateNormal();
  double before_small_t = mvtnorm.dmvnorm(2,
					  positions[1].get_position(),
					  raw_input,
					  cov_matrix);
  double log_before_small_t = mvtnorm.dmvnorm_log(2,
						  positions[0].get_position(),
						  raw_input,
						  cov_matrix);
  std::cout << "before_small_t=" << before_small_t << std::endl;
  std::cout << "log_before_small_t=" << log_before_small_t << std::endl;

  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=1; } else { a_power=0; };

    for (unsigned j=0; j<b_indeces.size(); ++j) {
      if (j==0) { b_power=1; } else { b_power=0; };


      set_data(current_a + a_indeces[i]*h_x,
	       current_x_0,
	       current_b + b_indeces[j]*h_x,
	       current_c,
	       current_y_0,
	       current_d);

      positions = small_t_image_positions_ax_bx();
      double x_0_star = gsl_vector_get(positions[0].get_position(),0)*(b_-a_) + a_indeces[i]*h_x;
      double y_0_star = gsl_vector_get(positions[0].get_position(),1)*(d_-c_);
      double x = gsl_vector_get(raw_input, 0);
      double y = gsl_vector_get(raw_input, 1);
      gsl_vector* original_scale_position = gsl_vector_alloc(2);
      gsl_vector_set(original_scale_position,0,x_0_star);
      gsl_vector_set(original_scale_position,1,y_0_star);
      gsl_vector* scaled_input = scale_input(raw_input);

      gsl_matrix_set(cov_matrix, 0,0,
		     sigma_x_*sigma_x_*t_);
      gsl_matrix_set(cov_matrix, 0,1,
		     rho_*sigma_x_*sigma_y_*t_);
      gsl_matrix_set(cov_matrix, 1,0,
		     rho_*sigma_x_*sigma_y_*t_);
      gsl_matrix_set(cov_matrix, 1,1,
		     sigma_y_*sigma_y_*t_);

      double polynomial = std::pow((x-x_0_star)/sigma_x_,2) +
	std::pow((y-y_0_star)/sigma_y_,2) -
	2*rho_/(sigma_x_*sigma_y_)*(x-x_0_star)*(y-y_0_star);
	  
      double smallt = mvtnorm.dmvnorm_log(2,
					  original_scale_position,
					  raw_input,
					  cov_matrix);
      double analytic_sol = analytic_solution(scaled_input);

      std::cout << "a_=" << a_ << ", x_0_=" << x_0_ << ", x_0_reflected=" << (gsl_vector_get(positions[0].get_position(),0)*(b_-a_) + a_indeces[i]*h_x)
		<< ", (b_-a_)=" << b_-a_
		<< ", a_2_=" << a_2_ << ", x_0_2_=" << x_0_2_ << ", (b_2_-a_2_)=" << b_2_-a_2_
		<< ", t_2_min=" << positions[0].get_t()
		<< ", t_min=" << scale_back_t(positions[0].get_t())
		<< ", log_analytic_sol=" << log(analytic_sol) - log(b_-a_) - log(d_-c_)
		<< ", analytic_sol=" << exp(log(analytic_sol) - log(b_-a_) - log(d_-c_))
		<< "\n";

      derivative = derivative +
	// (gsl_vector_get(positions[0].get_position(),0)*(b_-a_) + a_indeces[i]*h_x)*
	std::exp(smallt-log_before_small_t-log_CC - 2.0*log(h_x))*
	//polynomial*
	std::pow(-1, a_power)*
	std::pow(-1, b_power);

      derivative_with_sol = derivative_with_sol +
	exp(log(analytic_solution(scaled_input)) - log(b_-a_) - log(d_-c_))*
	std::pow(-1, a_power)*
	std::pow(-1, b_power);

      gsl_vector_free(original_scale_position);
      gsl_vector_free(scaled_input);
      std::cout << std::endl;
    }
  }
  //   }
  // }
  gsl_matrix_free(cov_matrix);
  
  set_data(current_a,
  	   current_x_0,
  	   current_b,
  	   current_c,
  	   current_y_0,
  	   current_d);
  positions = small_t_image_positions_ax_bx();

  double x = gsl_vector_get(raw_input,0);
  double y = gsl_vector_get(raw_input,1);
  double x_0 = gsl_vector_get(positions[0].get_position(),0);
  double y_0 = gsl_vector_get(positions[0].get_position(),1);
  
  std::cout << "The analytic deriv is = "
	    << -1.0*std::exp(log_before_small_t + log_CC + log(std::abs(2*(x-x_0)/std::pow(sigma_x_,2)*(-2) - 2*rho_/(sigma_x_*sigma_y_)*(-2)*(y-y_0))))
	    << std::endl;
  derivative = derivative* exp(log_before_small_t + log_CC);
  std::cout << "The solution with numerical polynomail deriv is = " << derivative << std::endl;
  derivative_with_sol = derivative_with_sol/(h_x*h_x);
  std::cout << "The deriv with solution is = "
	    << derivative_with_sol << std::endl;
  std::cout << "The analytic deriv from _ax() is = " << analytic_likelihood_ax(raw_input, 10000) << std::endl;

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

double BivariateSolver::scale_back_t(double t_2) const
{
  double t = 0;

  // STEP 1 SHIFTING
  double x_0_1 = x_0_ - a_;
  double b_1 = b_ - a_;
  double a_1 = a_ - a_;
  
  double y_0_1 = y_0_ - c_;
  double c_1 = c_ - c_;
  double d_1 = d_ - c_;
  
  // STEP 2 SCALING
  double Lx_2 = b_1 - a_1;
  double x_0_2 =  x_0_1 / Lx_2;
  double a_2 = a_1 / Lx_2;
  double b_2 = b_1 / Lx_2;
  double sigma_x_2 = sigma_x_ / Lx_2;
  
  double Ly_2 = d_1 - c_1;
  double y_0_2 =  y_0_1 / Ly_2;
  double c_2 = c_1 / Ly_2;
  double d_2 = d_1 / Ly_2;
  double sigma_y_2 = sigma_y_ / Ly_2;

  if (flipped_xy_flag_) {
    t = t_2/(sigma_y_2*sigma_y_2);
  } else {
    t = t_2/(sigma_x_2*sigma_x_2);
  }

  return t;
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

std::vector<BivariateImageWithTime> BivariateSolver::
small_t_image_positions_ax() const
{
  gsl_matrix* Rotation_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_memcpy(Rotation_matrix, small_t_solution_->get_rotation_matrix());

  double Rotation_matrix_inv [4];
  gsl_matrix_view Rotation_matrix_inv_view =
    gsl_matrix_view_array(Rotation_matrix_inv, 2, 2);

  gsl_matrix_memcpy(&Rotation_matrix_inv_view.matrix,
  		    small_t_solution_->get_rotation_matrix());

  double cc = std::sin(M_PI/4.0);
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 0,0,
		 0.5/cc * sigma_x_2_*std::sqrt(1-rho_));
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 0,1,
		  0.5/cc * sigma_x_2_*std::sqrt(1+rho_));
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 1,0,
		 -0.5/cc * sigma_y_2_*std::sqrt(1-rho_));
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 1,1,
		 0.5/cc * sigma_y_2_*std::sqrt(1+rho_));
  
  // int s = 0;
  // gsl_permutation * p = gsl_permutation_alloc(2);
  // gsl_linalg_LU_decomp(Rotation_matrix, p, &s);
  // gsl_linalg_LU_invert(Rotation_matrix, p, &Rotation_matrix_inv_view.matrix);
  // gsl_permutation_free(p);
  // gsl_matrix_free(Rotation_matrix);

  // double product [4];
  // gsl_matrix_view product_view =
  //   gsl_matrix_view_array(product, 2,2);
  // // C = alpha*op(A)*op(B) + beta*C
  // gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  // 		 CblasNoTrans, //op(B) = B
  // 		 1.0, //alpha=1
  // 		 small_t_solution_->get_rotation_matrix(), //A
  // 		 &Rotation_matrix_inv_view.matrix, //B
  // 		 0.0, //beta=0
  // 		 &product_view.matrix); //C
  // std::cout << gsl_matrix_get(&product_view.matrix,0,0) << " "
  // 	    << gsl_matrix_get(&product_view.matrix,0,1) << "\n"
  // 	    << gsl_matrix_get(&product_view.matrix,1,0) << " "
  //   	    << gsl_matrix_get(&product_view.matrix,1,1) << std::endl;

  double corner_points_array [10] = {get_a_2(),
  				     get_b_2(),
  				     get_b_2(),
  				     get_a_2(),
  				     get_x_0_2(),
  				     // // //
  				     get_c_2(),
  				     get_c_2(),
  				     get_d_2(),
  				     get_d_2(),
  				     get_y_0_2()};
  
  gsl_matrix_view corner_points_view = gsl_matrix_view_array(corner_points_array,
  							     2, 5);

  double corner_points_transformed_array [10];
  gsl_matrix_view corner_points_transformed_view =
    gsl_matrix_view_array(corner_points_transformed_array, 2, 5);
	  
  double images_array [4];
  for (unsigned i=0; i<2; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+2] = get_y_0_2();
  }
  double images_transformed_array [4];

  gsl_matrix_view images_view = gsl_matrix_view_array(images_array, 2, 2);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, 2);

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
  
  gsl_vector_view lower_left_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,0);
  gsl_vector_view lower_right_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,1);
  gsl_vector_view upper_right_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,2);
  gsl_vector_view upper_left_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,3);
  gsl_vector_view initial_condition_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,4);

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
  				      &initial_condition_transformed_view.vector),
      small_t_solution_->
      distance_from_point_to_axis_raw(lines[1][0],
  				      lines[1][1],
  				      &initial_condition_transformed_view.vector,
  				      &initial_condition_transformed_view.vector),
      small_t_solution_->
      distance_from_point_to_axis_raw(lines[2][0],
  				      lines[2][1],
  				      &initial_condition_transformed_view.vector,
  				      &initial_condition_transformed_view.vector),
      small_t_solution_->
      distance_from_point_to_axis_raw(lines[3][0],
  				      lines[3][1],
  				      &initial_condition_transformed_view.vector,
  				      &initial_condition_transformed_view.vector)};

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

  std::vector<gsl_vector_view> images_vector (2);
  for (unsigned i=0; i<2; ++i) {
    images_vector[i] =
      gsl_matrix_column(&images_transformed_view.matrix,i);
  }

  std::vector<std::vector<double>> distance_from_image_to_line (2, std::vector<double> (4));
  std::vector<double> max_admissible_times (1);
  std::vector<BivariateImageWithTime> final_images (1);
  std::vector<double> signs_vector = std::vector<double> (2,1.0);

  unsigned image_counter = 0;
  unsigned p = 3;
  // C = alpha*op(A)*op(B) + beta*C
  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
		 CblasNoTrans, //op(B) = B
		 1.0, //alpha=1
		 small_t_solution_->get_rotation_matrix(), //A
		 &images_view.matrix, //B
		 0.0, //beta=0
		 &images_transformed_view.matrix); //C
    
  signs_vector = std::vector<double> (2,1.0);
  unsigned counter = 0;
  for (unsigned i=0; i<2; ++i) {
    gsl_vector* current_image = &images_vector[counter].vector;
    if (i==1) {
      small_t_solution_->reflect_point_raw(lines[p][0],
					   lines[p][1],
					   current_image);
      signs_vector[counter] = signs_vector[counter]*(-1.0);
    }

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
      distance_from_point_to_axis_raw(lines[distance_to_line_indeces[3]][0],
				      lines[distance_to_line_indeces[3]][1],
				      &initial_condition_transformed_view.vector,
				      current_image);
    // printf("%g %g %g %g\n", d1,d2,d3,d4);
	  
    distance_from_image_to_line[counter][0] = d1;
    distance_from_image_to_line[counter][1] = d2;
    distance_from_image_to_line[counter][2] = d3;
    distance_from_image_to_line[counter][3] = d4;

    // printf("image.%i = c(%g,%g);\n",
    // 	   counter,
    // 	   gsl_vector_get(current_image, 0),
    // 	   gsl_vector_get(current_image, 1));
    counter = counter + 1;
  }
		
  int sign = 1;
  for (unsigned i=1; i<2; ++i) {
    std::vector<double>::iterator result = std::min_element(distance_from_image_to_line[i].begin(),
							    distance_from_image_to_line[i].end());
    if (!std::signbit(*result)) {
      sign = -1;
      break;
    }
  }

  // calculating max admissible time
  double mmax = 1.0;
  double max_admissible_t = 1.0; //(1.0/0.9) * std::pow(distance_from_image_to_line[0][1]/6.0, 2);

  while (mmax > 1e-8) {
    max_admissible_t = 0.9*max_admissible_t;
    mmax = 0.0;
    for (unsigned k=0; k<4; ++k) { // iterating over boundaries
      unsigned M = 50;
      double t_max = 1.0;
      double t=0.0;
      double dt=t_max/M;
      std::vector<double> ts(M);
      std::vector<double> ls = std::vector<double>(M, 0.0);
      std::generate(ts.begin(), ts.end(), [&] () mutable {t = t + dt; return t; });
    
      for (unsigned j=0; j<M; ++j) { // iterating over points on boundaries
	double tt = ts[j];
    
	double x_current = gsl_vector_get(lines[k][0], 0) +
	  tt*( gsl_vector_get(lines[k][1], 0) - gsl_vector_get(lines[k][0], 0) );

	double y_current = gsl_vector_get(lines[k][0], 1) +
	  tt*( gsl_vector_get(lines[k][1], 1) - gsl_vector_get(lines[k][0], 1) );
    
	for (unsigned i=0; i<2; ++i) { // iterating over images
	  gsl_vector* current_image = &images_vector[i].vector;
	  double x_not = gsl_vector_get(current_image, 0);
	  double y_not = gsl_vector_get(current_image, 1);

	  ls[j] = ls[j] +
	    signs_vector[i]*
	    gsl_ran_gaussian_pdf(x_current-x_not, std::sqrt(max_admissible_t))*
	    gsl_ran_gaussian_pdf(y_current-y_not, std::sqrt(max_admissible_t))*
	    1.0/(sigma_x_2_*sigma_y_2_*sqrt(1-rho_)*sqrt(1+rho_));
	}
      }
      
      for (double ll : ls) {
	if (std::abs(ll) > mmax) {
	  mmax = std::abs(ll);
	}
      }
    }
  }

  max_admissible_times[image_counter] = sign*max_admissible_t;
	  
  gsl_vector* current_image = gsl_vector_alloc(2);
  // C = alpha*op(A)*x + beta*C
  gsl_blas_dgemv(CblasNoTrans, //op(A) = A
		 1.0, //alpha=1
		 &Rotation_matrix_inv_view.matrix, //A
		 &images_vector[1].vector, //x
		 0.0, //beta=0
		 current_image); //C
	  
  final_images[image_counter] = BivariateImageWithTime(current_image,
						       sign*max_admissible_t,
						       1.0);
  gsl_vector_free(current_image);
	  
  image_counter = image_counter + 1;
	  
  std::vector<double>::iterator result = std::max_element(max_admissible_times.begin(),
  							  max_admissible_times.end());
  double biggest_time = *result;
  result = std::min_element(max_admissible_times.begin(),
  			    max_admissible_times.end());
  double smallest_time = *result;

  std::vector<BivariateImageWithTime> max_t_images;
  for (const BivariateImageWithTime& current_image : final_images) {
    if (std::abs( current_image.get_t() - biggest_time) <= std::numeric_limits<double>::epsilon()) {
      max_t_images.push_back( current_image );
    }
  }
  
  return max_t_images;
}

std::vector<BivariateImageWithTime> BivariateSolver::
small_t_image_positions_ax_bx() const
{
  gsl_matrix* Rotation_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_memcpy(Rotation_matrix, small_t_solution_->get_rotation_matrix());

  double Rotation_matrix_inv [4];
  gsl_matrix_view Rotation_matrix_inv_view =
    gsl_matrix_view_array(Rotation_matrix_inv, 2, 2);

  gsl_matrix_memcpy(&Rotation_matrix_inv_view.matrix,
  		    small_t_solution_->get_rotation_matrix());

  double cc = std::sin(M_PI/4.0);
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 0,0,
		 0.5/cc * sigma_x_2_*std::sqrt(1-rho_));
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 0,1,
		  0.5/cc * sigma_x_2_*std::sqrt(1+rho_));
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 1,0,
		 -0.5/cc * sigma_y_2_*std::sqrt(1-rho_));
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 1,1,
		 0.5/cc * sigma_y_2_*std::sqrt(1+rho_));
  
  double corner_points_array [10] = {get_a_2(),
  				     get_b_2(),
  				     get_b_2(),
  				     get_a_2(),
  				     get_x_0_2(),
  				     // // //
  				     get_c_2(),
  				     get_c_2(),
  				     get_d_2(),
  				     get_d_2(),
  				     get_y_0_2()};
  
  gsl_matrix_view corner_points_view = gsl_matrix_view_array(corner_points_array,
  							     2, 5);

  double corner_points_transformed_array [10];
  gsl_matrix_view corner_points_transformed_view =
    gsl_matrix_view_array(corner_points_transformed_array, 2, 5);
	  
  double images_array [8];
  for (unsigned i=0; i<4; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+2] = get_y_0_2();
  }
  double images_transformed_array [8];

  gsl_matrix_view images_view = gsl_matrix_view_array(images_array, 2, 4);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, 4);

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
  
  gsl_vector_view lower_left_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,0);
  gsl_vector_view lower_right_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,1);
  gsl_vector_view upper_right_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,2);
  gsl_vector_view upper_left_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,3);
  gsl_vector_view initial_condition_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,4);

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
  				      &initial_condition_transformed_view.vector),
      small_t_solution_->
      distance_from_point_to_axis_raw(lines[1][0],
  				      lines[1][1],
  				      &initial_condition_transformed_view.vector,
  				      &initial_condition_transformed_view.vector),
      small_t_solution_->
      distance_from_point_to_axis_raw(lines[2][0],
  				      lines[2][1],
  				      &initial_condition_transformed_view.vector,
  				      &initial_condition_transformed_view.vector),
      small_t_solution_->
      distance_from_point_to_axis_raw(lines[3][0],
  				      lines[3][1],
  				      &initial_condition_transformed_view.vector,
  				      &initial_condition_transformed_view.vector)};

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

  std::vector<gsl_vector_view> images_vector (4);
  for (unsigned i=0; i<4; ++i) {
    images_vector[i] =
      gsl_matrix_column(&images_transformed_view.matrix,i);
  }

  std::vector<std::vector<double>> distance_from_image_to_line (4, std::vector<double> (4));
  std::vector<double> max_admissible_times (2);
  std::vector<BivariateImageWithTime> final_images (2);
  std::vector<double> signs_vector = std::vector<double> (4,1.0);

  unsigned image_counter = 0;
  std::vector<unsigned> p_indeces {3,1};
  for (unsigned p : p_indeces) {
    std::vector<unsigned> o_indeces {3,1};
    std::vector<unsigned>::iterator it;
    it = std::find(o_indeces.begin(), o_indeces.end(), p);
    o_indeces.erase(it);

    for (unsigned o : o_indeces) {

      // C = alpha*op(A)*op(B) + beta*C
      gsl_blas_dgemm(CblasNoTrans, //op(A) = A
		     CblasNoTrans, //op(B) = B
		     1.0, //alpha=1
		     small_t_solution_->get_rotation_matrix(), //A
		     &images_view.matrix, //B
		     0.0, //beta=0
		     &images_transformed_view.matrix); //C

      signs_vector = std::vector<double> (4,1.0);
      unsigned counter = 0;
      for (unsigned j=0; j<2; ++j) {
	for (unsigned i=0; i<2; ++i) {
	  gsl_vector* current_image = &images_vector[counter].vector;
	  if (i==1) {
	    small_t_solution_->reflect_point_raw(lines[p][0],
						 lines[p][1],
						 current_image);
	    signs_vector[counter] = signs_vector[counter]*(-1.0);
	  }

	  if (j==1) {
	    small_t_solution_->reflect_point_raw(lines[o][0],
						 lines[o][1],
						 current_image);
	    signs_vector[counter] = signs_vector[counter]*(-1.0);
	  }

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
	    distance_from_point_to_axis_raw(lines[distance_to_line_indeces[3]][0],
					    lines[distance_to_line_indeces[3]][1],
					    &initial_condition_transformed_view.vector,
					    current_image);
	  // printf("%g %g %g %g\n", d1,d2,d3,d4);
	  
	  distance_from_image_to_line[counter][0] = d1;
	  distance_from_image_to_line[counter][1] = d2;
	  distance_from_image_to_line[counter][2] = d3;
	  distance_from_image_to_line[counter][3] = d4;

	  // printf("image.%i = c(%g,%g);\n",
	  // 	   counter,
	  // 	   gsl_vector_get(current_image, 0),
	  // 	   gsl_vector_get(current_image, 1));
	  counter = counter + 1;
	}
      }
		
      int sign = 1;
      for (unsigned i=1; i<4; ++i) {
	std::vector<double>::iterator result = std::min_element(distance_from_image_to_line[i].begin(),
								distance_from_image_to_line[i].end());
	if (!std::signbit(*result)) {
	  sign = -1;
	  break;
	}
      }

      // calculating max admissible time
      double mmax = 1.0;
      double max_admissible_t = 1.0; //(1.0/0.9) * std::pow(distance_from_image_to_line[0][1]/6.0, 2);

      while (mmax > 1e-8) {
	max_admissible_t = 0.9*max_admissible_t;
	mmax = 0.0;
	for (unsigned k=0; k<4; ++k) { // iterating over boundaries
	  unsigned M = 50;
	  double t_max = 1.0;
	  double t=0.0;
	  double dt=t_max/M;
	  std::vector<double> ts(M);
	  std::vector<double> ls = std::vector<double>(M, 0.0);
	  std::generate(ts.begin(), ts.end(), [&] () mutable {t = t + dt; return t; });
    
	  for (unsigned j=0; j<M; ++j) { // iterating over points on boundaries
	    double tt = ts[j];
    
	    double x_current = gsl_vector_get(lines[k][0], 0) +
	      tt*( gsl_vector_get(lines[k][1], 0) - gsl_vector_get(lines[k][0], 0) );

	    double y_current = gsl_vector_get(lines[k][0], 1) +
	      tt*( gsl_vector_get(lines[k][1], 1) - gsl_vector_get(lines[k][0], 1) );
    
	    for (unsigned i=0; i<4; ++i) { // iterating over images
	      gsl_vector* current_image = &images_vector[i].vector;
	      double x_not = gsl_vector_get(current_image, 0);
	      double y_not = gsl_vector_get(current_image, 1);

	      ls[j] = ls[j] +
		signs_vector[i]*
		gsl_ran_gaussian_pdf(x_current-x_not, std::sqrt(max_admissible_t))*
		gsl_ran_gaussian_pdf(y_current-y_not, std::sqrt(max_admissible_t))*
		1.0/(sigma_x_2_*sigma_y_2_*sqrt(1-rho_)*sqrt(1+rho_));
	    }
	  }
      
	  for (double ll : ls) {
	    if (std::abs(ll) > mmax) {
	      mmax = std::abs(ll);
	    }
	  }
	}
      }

      max_admissible_times[image_counter] = sign*max_admissible_t;
	  
      gsl_vector* current_image = gsl_vector_alloc(2);
      // C = alpha*op(A)*x + beta*C
      gsl_blas_dgemv(CblasNoTrans, //op(A) = A
		     1.0, //alpha=1
		     &Rotation_matrix_inv_view.matrix, //A
		     &images_vector[3].vector, //x
		     0.0, //beta=0
		     current_image); //C
	  
      final_images[image_counter] = BivariateImageWithTime(current_image,
							   sign*max_admissible_t,
							   1.0);
      gsl_vector_free(current_image);
	  
      image_counter = image_counter + 1;
  
    }
  }

  std::vector<double>::iterator result = std::max_element(max_admissible_times.begin(),
  							  max_admissible_times.end());
  double biggest_time = *result;
  result = std::min_element(max_admissible_times.begin(),
  			    max_admissible_times.end());
  double smallest_time = *result;

  std::vector<BivariateImageWithTime> max_t_images;
  for (const BivariateImageWithTime& current_image : final_images) {
    if (std::abs( current_image.get_t() - biggest_time) <= std::numeric_limits<double>::epsilon()) {
      max_t_images.push_back( current_image );
    }
  }
  
  return max_t_images;
}

std::vector<BivariateImageWithTime> BivariateSolver::small_t_image_positions() const
{
  gsl_matrix* Rotation_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_memcpy(Rotation_matrix, small_t_solution_->get_rotation_matrix());

  double Rotation_matrix_inv [4];
  gsl_matrix_view Rotation_matrix_inv_view =
    gsl_matrix_view_array(Rotation_matrix_inv, 2, 2);

  gsl_matrix_memcpy(&Rotation_matrix_inv_view.matrix,
  		    small_t_solution_->get_rotation_matrix());

  double cc = std::sin(M_PI/4.0);
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 0,0,
		 0.5/cc * sigma_x_2_*std::sqrt(1-rho_));
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 0,1,
		  0.5/cc * sigma_x_2_*std::sqrt(1+rho_));
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 1,0,
		 -0.5/cc * sigma_y_2_*std::sqrt(1-rho_));
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 1,1,
		 0.5/cc * sigma_y_2_*std::sqrt(1+rho_));
  
  // int s = 0;
  // gsl_permutation * p = gsl_permutation_alloc(2);
  // gsl_linalg_LU_decomp(Rotation_matrix, p, &s);
  // gsl_linalg_LU_invert(Rotation_matrix, p, &Rotation_matrix_inv_view.matrix);
  // gsl_permutation_free(p);
  // gsl_matrix_free(Rotation_matrix);

  // double product [4];
  // gsl_matrix_view product_view =
  //   gsl_matrix_view_array(product, 2,2);
  // // C = alpha*op(A)*op(B) + beta*C
  // gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  // 		 CblasNoTrans, //op(B) = B
  // 		 1.0, //alpha=1
  // 		 small_t_solution_->get_rotation_matrix(), //A
  // 		 &Rotation_matrix_inv_view.matrix, //B
  // 		 0.0, //beta=0
  // 		 &product_view.matrix); //C
  // std::cout << gsl_matrix_get(&product_view.matrix,0,0) << " "
  // 	    << gsl_matrix_get(&product_view.matrix,0,1) << "\n"
  // 	    << gsl_matrix_get(&product_view.matrix,1,0) << " "
  //   	    << gsl_matrix_get(&product_view.matrix,1,1) << std::endl;

  double corner_points_array [10] = {get_a_2(),
  				     get_b_2(),
  				     get_b_2(),
  				     get_a_2(),
  				     get_x_0_2(),
  				     // // //
  				     get_c_2(),
  				     get_c_2(),
  				     get_d_2(),
  				     get_d_2(),
  				     get_y_0_2()};
  
  gsl_matrix_view corner_points_view = gsl_matrix_view_array(corner_points_array,
  							     2, 5);

  double corner_points_transformed_array [10];
  gsl_matrix_view corner_points_transformed_view =
    gsl_matrix_view_array(corner_points_transformed_array, 2, 5);
	  
  double images_array [32];
  for (unsigned i=0; i<16; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+16] = get_y_0_2();
  }
  double images_transformed_array [32];

  gsl_matrix_view images_view = gsl_matrix_view_array(images_array, 2, 16);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, 16);

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
  
  gsl_vector_view lower_left_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,0);
  gsl_vector_view lower_right_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,1);
  gsl_vector_view upper_right_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,2);
  gsl_vector_view upper_left_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,3);
  gsl_vector_view initial_condition_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,4);

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
  				      &initial_condition_transformed_view.vector),
      small_t_solution_->
      distance_from_point_to_axis_raw(lines[1][0],
  				      lines[1][1],
  				      &initial_condition_transformed_view.vector,
  				      &initial_condition_transformed_view.vector),
      small_t_solution_->
      distance_from_point_to_axis_raw(lines[2][0],
  				      lines[2][1],
  				      &initial_condition_transformed_view.vector,
  				      &initial_condition_transformed_view.vector),
      small_t_solution_->
      distance_from_point_to_axis_raw(lines[3][0],
  				      lines[3][1],
  				      &initial_condition_transformed_view.vector,
  				      &initial_condition_transformed_view.vector)};

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

  std::vector<std::vector<double>> distance_from_image_to_line (16, std::vector<double> (4));
  std::vector<double> max_admissible_times (24);
  std::vector<BivariateImageWithTime> final_images (24);
  std::vector<double> signs_vector = std::vector<double> (16,1.0);

  // printf("pdf(\"./src/kernel-expansion/documentation/chapter-3/illustration-rho-0-all-configurations.pdf\", 8, 8);\n");
  // printf("par(mfrow=c(5,5), mar=c(1,1,1,1));\n");
  unsigned image_counter = 0;
  for (unsigned p=0; p<4; ++p) {
    std::vector<unsigned> o_indeces(4);
    std::iota(o_indeces.begin(), o_indeces.end(), 0);
    o_indeces.erase(o_indeces.begin() + p);
        
    for (unsigned o : o_indeces) {
      std::vector<unsigned> n_indeces(4);
      std::iota(n_indeces.begin(), n_indeces.end(), 0);

      std::vector<unsigned>::iterator it;
      it = std::find(n_indeces.begin(), n_indeces.end(), p);
      n_indeces.erase(it);
      //
      it = std::find(n_indeces.begin(), n_indeces.end(), o);
      n_indeces.erase(it);

      for (unsigned n : n_indeces) {
  	std::vector<unsigned> m_indeces(4);
  	std::iota(m_indeces.begin(), m_indeces.end(), 0);

  	std::vector<unsigned>::iterator it;
  	it = std::find(m_indeces.begin(), m_indeces.end(), p);
  	m_indeces.erase(it);
  	//
  	it = std::find(m_indeces.begin(), m_indeces.end(), o);
  	m_indeces.erase(it);
  	//
  	it = std::find(m_indeces.begin(), m_indeces.end(), n);
  	m_indeces.erase(it);

  	for (unsigned m : m_indeces) {
  	  // std::cout << "## (p=" << p
  	  // 	    << ",o=" << o
  	  // 	    << ",n=" << n
  	  // 	    << ",m=" << m << ")" << std::endl;

  	  // C = alpha*op(A)*op(B) + beta*C
  	  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  	  		 CblasNoTrans, //op(B) = B
  	  		 1.0, //alpha=1
  	  		 small_t_solution_->get_rotation_matrix(), //A
  	  		 &images_view.matrix, //B
  	  		 0.0, //beta=0
  	  		 &images_transformed_view.matrix); //C

  	  signs_vector = std::vector<double> (16,1.0);
  	  unsigned counter = 0;
  	  for (unsigned l=0; l<2; ++l) {
  	    for (unsigned k=0; k<2; ++k) {
  	      for (unsigned j=0; j<2; ++j) {
  		for (unsigned i=0; i<2; ++i) {
  		  gsl_vector* current_image = &images_vector[counter].vector;
  		  if (i==1) {
  		    small_t_solution_->reflect_point_raw(lines[p][0],
  							 lines[p][1],
  							 current_image);
  		    signs_vector[counter] = signs_vector[counter]*(-1.0);
  		  }
  		  if (j==1) {
  		    small_t_solution_->reflect_point_raw(lines[o][0],
  							 lines[o][1],
  							 current_image);
  		    signs_vector[counter] = signs_vector[counter]*(-1.0);
  		  }
  		  if (k==1) {
  		    small_t_solution_->reflect_point_raw(lines[n][0],
  							 lines[n][1],
  							 current_image);
  		    signs_vector[counter] = signs_vector[counter]*(-1.0);
  		  }
  		  if (l==1) {
  		    small_t_solution_->reflect_point_raw(lines[m][0],
  							 lines[m][1],
  							 current_image);
  		    signs_vector[counter] = signs_vector[counter]*(-1.0);
  		  }
  		  // printf("## image %i distances: ", counter);

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
  		    distance_from_point_to_axis_raw(lines[distance_to_line_indeces[3]][0],
  						    lines[distance_to_line_indeces[3]][1],
  						    &initial_condition_transformed_view.vector,
  						    current_image);
  		  // printf("%g %g %g %g\n", d1,d2,d3,d4);
	  
  		  distance_from_image_to_line[counter][0] = d1;
  		  distance_from_image_to_line[counter][1] = d2;
  		  distance_from_image_to_line[counter][2] = d3;
  		  distance_from_image_to_line[counter][3] = d4;

  		  // printf("image.%i = c(%g,%g);\n",
  			 // counter,
  			 // gsl_vector_get(current_image, 0),
  			 // gsl_vector_get(current_image, 1));
	  
  		  counter = counter + 1;
  		}
  	      }
  	    }
  	  }
  	  // printf("lower.left.corner=c(%g,%g);\n",
  	  // 	 gsl_vector_get(&lower_left_transformed_view.vector, 0),
  	  // 	 gsl_vector_get(&lower_left_transformed_view.vector, 1));
  	  // printf("lower.right.corner=c(%g,%g);\n",
  	  // 	 gsl_vector_get(&lower_right_transformed_view.vector, 0),
  	  // 	 gsl_vector_get(&lower_right_transformed_view.vector, 1));
  	  // printf("upper.left.corner=c(%g,%g);\n",
  	  // 	 gsl_vector_get(&upper_left_transformed_view.vector, 0),
  	  // 	 gsl_vector_get(&upper_left_transformed_view.vector, 1));
  	  // printf("upper.right.corner=c(%g,%g);\n",
  	  // 	 gsl_vector_get(&upper_right_transformed_view.vector, 0),
  	  // 	 gsl_vector_get(&upper_right_transformed_view.vector, 1));
  	  int sign = 1;
  	  for (unsigned i=1; i<16; ++i) {
  	    std::vector<double>::iterator result = std::min_element(distance_from_image_to_line[i].begin(),
  								    distance_from_image_to_line[i].end());
  	    if (!std::signbit(*result)) {
  	      sign = -1;
  	      break;
  	    }
  	  }

  	  // calculating max admissible time
  	  double mmax = 1.0;
  	  double max_admissible_t = 1.0; //(1.0/0.9) * std::pow(distance_from_image_to_line[0][1]/6.0, 2);

  	  while (mmax > 1e-8) {
  	    max_admissible_t = 0.9*max_admissible_t;
  	    mmax = 0.0;
  	    for (unsigned k=0; k<4; ++k) { // iterating over boundaries
  	      unsigned M = 50;
  	      double t_max = 1.0;
  	      std::vector<double> ts(M);
  	      std::vector<double> ls = std::vector<double>(M, 0.0);
  	      std::generate(ts.begin(), ts.end(), [t=0.0, dt = t_max/M] () mutable {t = t + dt; return t; });
    
  	      for (unsigned j=0; j<M; ++j) { // iterating over points on boundaries
  		double tt = ts[j];
    
  		double x_current = gsl_vector_get(lines[k][0], 0) +
  		  tt*( gsl_vector_get(lines[k][1], 0) - gsl_vector_get(lines[k][0], 0) );

  		double y_current = gsl_vector_get(lines[k][0], 1) +
  		  tt*( gsl_vector_get(lines[k][1], 1) - gsl_vector_get(lines[k][0], 1) );
    
  		for (unsigned i=0; i<16; ++i) { // iterating over images
  		  gsl_vector* current_image = &images_vector[i].vector;
  		  double x_not = gsl_vector_get(current_image, 0);
  		  double y_not = gsl_vector_get(current_image, 1);

  		  ls[j] = ls[j] +
  		    signs_vector[i]*
  		    gsl_ran_gaussian_pdf(x_current-x_not, std::sqrt(max_admissible_t))*
  		    gsl_ran_gaussian_pdf(y_current-y_not, std::sqrt(max_admissible_t))*
  		    1.0/(sigma_x_2_*sigma_y_2_*sqrt(1-rho_)*sqrt(1+rho_));
  		}
  	      }
      
  	      for (double ll : ls) {
  		if (std::abs(ll) > mmax) {
  		  mmax = std::abs(ll);
  		}
  	      }
  	    }
  	  }

  	  // printf("rect(par(\"usr\")[1], par(\"usr\")[2], par(\"usr\")[3], par(\"usr\")[4], col=\"grey\");\n");
  	  // printf("plot(x=0, type=\"n\", xlim = 3*c(-max(abs(c(lower.left.corner, lower.right.corner, upper.left.corner, upper.right.corner))), max(abs(c(lower.left.corner, lower.right.corner, upper.left.corner, upper.right.corner)))), ylim = 3*c(-max(abs(c(lower.left.corner, lower.right.corner, upper.left.corner, upper.right.corner))), max(abs(c(lower.left.corner, lower.right.corner, upper.left.corner, upper.right.corner)))), xlab=\"x\", ylab=\"y\");\n");
  	  // printf("lines(x=c(lower.left.corner[1],lower.right.corner[1]), y=c(lower.left.corner[2],lower.right.corner[2]), lwd=2, col=\"black\");\n"); // border 1
  	  // printf("lines(x=c(lower.right.corner[1],upper.right.corner[1]), y=c(lower.right.corner[2],upper.right.corner[2]), lwd=2, col=\"black\");\n"); // border 2
  	  // printf("lines(x=c(upper.right.corner[1],upper.left.corner[1]), y=c(upper.right.corner[2],upper.left.corner[2]), lwd=2, col=\"black\");\n"); // border 3
  	  // printf("lines(x=c(upper.left.corner[1],lower.left.corner[1]), y=c(upper.left.corner[2],lower.left.corner[2]), lwd=2, col=\"black\");\n"); // border 4

  	  // for (unsigned ii=0; ii<16; ++ii) {
  	  //   if (std::signbit(signs_vector[ii])) {
  	  //     printf("points(x=image.%i[1],y=image.%i[2],pch=20,lwd=2,col=\"blue\");\n", ii, ii);	      
  	  //   } else {
  	  //     printf("points(x=image.%i[1],y=image.%i[2],pch=20,lwd=2,col=\"green\");\n", ii, ii);	      
  	  //   }
  	  //   // printf("text(x=image.%i[1],y=image.%i[2],%f);\n", ii, ii, signs_vector[ii]);
  	  // }
  	  // printf("points(x=image.%i[1],y=image.%i[2],lwd=2,pch=20,col=\"red\");\n", 15, 15);
  	  // std::cout << "## max_admissible_t=" << max_admissible_t << std::endl;
  	  max_admissible_times[image_counter] = sign*max_admissible_t;
	  
  	  gsl_vector* current_image = gsl_vector_alloc(2);
  	  // C = alpha*op(A)*x + beta*C
  	  gsl_blas_dgemv(CblasNoTrans, //op(A) = A
  			 1.0, //alpha=1
  			 &Rotation_matrix_inv_view.matrix, //A
  			 &images_vector[15].vector, //x
  			 0.0, //beta=0
  			 current_image); //C
	  
  	  final_images[image_counter] = BivariateImageWithTime(current_image,
  							       sign*max_admissible_t,
  							       1.0);
  	  gsl_vector_free(current_image);
	  
  	  image_counter = image_counter + 1;
  	}
      }
    }
  }

  // printf("dev.off();\n");

  // for (const BivariateImageWithTime& current_image : final_images) {
  //   std::cout << "## (" << gsl_vector_get(current_image.get_position(),0)
  // 	      << "," << gsl_vector_get(current_image.get_position(),1)
  // 	      << "," << current_image.get_t() << ")\n";
  // }
  
  std::vector<double>::iterator result = std::max_element(max_admissible_times.begin(),
  							  max_admissible_times.end());
  double biggest_time = *result;
  result = std::min_element(max_admissible_times.begin(),
  			    max_admissible_times.end());
  double smallest_time = *result;

  std::vector<BivariateImageWithTime> max_t_images;
  for (const BivariateImageWithTime& current_image : final_images) {
    if (std::abs( current_image.get_t() - biggest_time) <= std::numeric_limits<double>::epsilon()) {
      max_t_images.push_back( current_image );
    }
  }
  
  // std::cout << "## biggest_time = " << biggest_time << std::endl;
  // for (const BivariateImageWithTime& big_t_image : max_t_images) {
  //   std::cout << "## (" << gsl_vector_get(big_t_image.get_position(),0)
  // 	      << "," << gsl_vector_get(big_t_image.get_position(),1)
  // 	      << "," << big_t_image.get_t() << ");\n";
  // }

  return max_t_images;
}


double BivariateSolver::analytic_solution_small_t(const gsl_vector* input) const
{
  printf("in small_t_image_positions()\n");
  gsl_matrix* Rotation_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_memcpy(Rotation_matrix, small_t_solution_->get_rotation_matrix());

  double Rotation_matrix_inv [4];
  gsl_matrix_view Rotation_matrix_inv_view =
    gsl_matrix_view_array(Rotation_matrix_inv, 2, 2);

  gsl_matrix_memcpy(&Rotation_matrix_inv_view.matrix,
  		    small_t_solution_->get_rotation_matrix());

  double cc = std::sin(M_PI/4.0);
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 0,0,
		 0.5/cc * sigma_x_2_*std::sqrt(1-rho_));
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 0,1,
		  0.5/cc * sigma_x_2_*std::sqrt(1+rho_));
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 1,0,
		 -0.5/cc * sigma_y_2_*std::sqrt(1-rho_));
  gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 1,1,
		 0.5/cc * sigma_y_2_*std::sqrt(1+rho_));
  
  double corner_points_array [10] = {get_a_2(),
  				     get_b_2(),
  				     get_b_2(),
  				     get_a_2(),
  				     get_x_0_2(),
  				     // // //
  				     get_c_2(),
  				     get_c_2(),
  				     get_d_2(),
  				     get_d_2(),
  				     get_y_0_2()};
  
  gsl_matrix_view corner_points_view = gsl_matrix_view_array(corner_points_array,
  							     2, 5);

  double corner_points_transformed_array [10];
  gsl_matrix_view corner_points_transformed_view =
    gsl_matrix_view_array(corner_points_transformed_array, 2, 5);
	  
  double images_array [32];
  for (unsigned i=0; i<16; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+16] = get_y_0_2();
  }
  double images_transformed_array [32];

  gsl_matrix_view images_view = gsl_matrix_view_array(images_array, 2, 16);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, 16);

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
  
  gsl_vector_view lower_left_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,0);
  gsl_vector_view lower_right_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,1);
  gsl_vector_view upper_right_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,2);
  gsl_vector_view upper_left_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,3);
  gsl_vector_view initial_condition_transformed_view =
    gsl_matrix_column(&corner_points_transformed_view.matrix,4);

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
  				      &initial_condition_transformed_view.vector),
      small_t_solution_->
      distance_from_point_to_axis_raw(lines[1][0],
  				      lines[1][1],
  				      &initial_condition_transformed_view.vector,
  				      &initial_condition_transformed_view.vector),
      small_t_solution_->
      distance_from_point_to_axis_raw(lines[2][0],
  				      lines[2][1],
  				      &initial_condition_transformed_view.vector,
  				      &initial_condition_transformed_view.vector),
      small_t_solution_->
      distance_from_point_to_axis_raw(lines[3][0],
  				      lines[3][1],
  				      &initial_condition_transformed_view.vector,
  				      &initial_condition_transformed_view.vector)};

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

  std::vector<std::vector<double>> distance_from_image_to_line (16, std::vector<double> (4));
  std::vector<double> max_admissible_times (24);
  std::vector<std::vector<BivariateImageWithTime>> final_images =
    std::vector<std::vector<BivariateImageWithTime>> (24, std::vector<BivariateImageWithTime> (16));
  std::vector<double> signs_vector = std::vector<double> (16,1.0);

  unsigned image_counter = 0;
  for (unsigned p=0; p<4; ++p) {
    std::vector<unsigned> o_indeces(4);
    std::iota(o_indeces.begin(), o_indeces.end(), 0);
    o_indeces.erase(o_indeces.begin() + p);
        
    for (unsigned o : o_indeces) {
      std::vector<unsigned> n_indeces(4);
      std::iota(n_indeces.begin(), n_indeces.end(), 0);

      std::vector<unsigned>::iterator it;
      it = std::find(n_indeces.begin(), n_indeces.end(), p);
      n_indeces.erase(it);
      //
      it = std::find(n_indeces.begin(), n_indeces.end(), o);
      n_indeces.erase(it);

      for (unsigned n : n_indeces) {
  	std::vector<unsigned> m_indeces(4);
  	std::iota(m_indeces.begin(), m_indeces.end(), 0);

  	std::vector<unsigned>::iterator it;
  	it = std::find(m_indeces.begin(), m_indeces.end(), p);
  	m_indeces.erase(it);
  	//
  	it = std::find(m_indeces.begin(), m_indeces.end(), o);
  	m_indeces.erase(it);
  	//
  	it = std::find(m_indeces.begin(), m_indeces.end(), n);
  	m_indeces.erase(it);

  	for (unsigned m : m_indeces) {
  	  // C = alpha*op(A)*op(B) + beta*C
  	  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  	  		 CblasNoTrans, //op(B) = B
  	  		 1.0, //alpha=1
  	  		 small_t_solution_->get_rotation_matrix(), //A
  	  		 &images_view.matrix, //B
  	  		 0.0, //beta=0
  	  		 &images_transformed_view.matrix); //C

	  signs_vector = std::vector<double> (16,1.0);
	  unsigned counter = 0;
	  for (unsigned l=0; l<2; ++l) {
	    for (unsigned k=0; k<2; ++k) {
	      for (unsigned j=0; j<2; ++j) {
		for (unsigned i=0; i<2; ++i) {
		  gsl_vector* current_image = &images_vector[counter].vector;
		  if (i==1) {
		    small_t_solution_->reflect_point_raw(lines[p][0],
							 lines[p][1],
							 current_image);
		    signs_vector[counter] = signs_vector[counter]*(-1.0);
		  }
		  if (j==1) {
		    small_t_solution_->reflect_point_raw(lines[o][0],
							 lines[o][1],
							 current_image);
		    signs_vector[counter] = signs_vector[counter]*(-1.0);
		  }
		  if (k==1) {
		    small_t_solution_->reflect_point_raw(lines[n][0],
							 lines[n][1],
							 current_image);
		    signs_vector[counter] = signs_vector[counter]*(-1.0);
		  }
		  if (l==1) {
		    small_t_solution_->reflect_point_raw(lines[m][0],
							 lines[m][1],
							 current_image);
		    signs_vector[counter] = signs_vector[counter]*(-1.0);
		  }
		  // printf("## image %i distances: ", counter);

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
		    distance_from_point_to_axis_raw(lines[distance_to_line_indeces[3]][0],
						    lines[distance_to_line_indeces[3]][1],
						    &initial_condition_transformed_view.vector,
						    current_image);
	  
		  distance_from_image_to_line[counter][0] = d1;
		  distance_from_image_to_line[counter][1] = d2;
		  distance_from_image_to_line[counter][2] = d3;
		  distance_from_image_to_line[counter][3] = d4;

		  gsl_vector* current_image_pos = gsl_vector_alloc(2);
		  // C = alpha*op(A)*x + beta*C
		  gsl_blas_dgemv(CblasNoTrans, //op(A) = A
				 1.0, //alpha=1
				 &Rotation_matrix_inv_view.matrix, //A
				 current_image, //x
				 0.0, //beta=0
				 current_image_pos); //C

		  final_images[image_counter][counter] =
		    BivariateImageWithTime(current_image_pos,
					   1.0,
					   signs_vector[counter]*1.0);
		  gsl_vector_free(current_image_pos);

		  counter = counter + 1;
		}
	      }
	    }
	  }
	  
	  int sign = 1;
	  for (unsigned i=1; i<16; ++i) {
	    std::vector<double>::iterator result = std::min_element(distance_from_image_to_line[i].begin(),
								    distance_from_image_to_line[i].end());
	    if (!std::signbit(*result)) {
	      sign = -1;
	      break;
	    }
	  }

	  // calculating max admissible time
	  double mmax = 1.0;
	  double max_admissible_t = 1.0; //(1.0/0.9) * std::pow(distance_from_image_to_line[0][1]/6.0, 2);

	  while (mmax > 1e-8) {
	    max_admissible_t = 0.9*max_admissible_t;
	    mmax = 0.0;
	    for (unsigned k=0; k<4; ++k) { // iterating over boundaries
	      unsigned M = 100;
	      double t_max = 1.0;
	      std::vector<double> ts(M);
	      std::vector<double> ls = std::vector<double>(M, 0.0);
	      std::generate(ts.begin(), ts.end(), [t=0.0, dt = t_max/M] () mutable {t = t + dt; return t; });
    
	      for (unsigned j=0; j<M; ++j) { // iterating over points on boundaries
		double tt = ts[j];
    
		double x_current = gsl_vector_get(lines[k][0], 0) +
		  tt*( gsl_vector_get(lines[k][1], 0) - gsl_vector_get(lines[k][0], 0) );

		double y_current = gsl_vector_get(lines[k][0], 1) +
		  tt*( gsl_vector_get(lines[k][1], 1) - gsl_vector_get(lines[k][0], 1) );
    
		for (unsigned i=0; i<16; ++i) { // iterating over images
		  gsl_vector* current_image = &images_vector[i].vector;
		  double x_not = gsl_vector_get(current_image, 0);
		  double y_not = gsl_vector_get(current_image, 1);

		  ls[j] = ls[j] +
		    signs_vector[i]*
		    gsl_ran_gaussian_pdf(x_current-x_not, std::sqrt(max_admissible_t))*
		    gsl_ran_gaussian_pdf(y_current-y_not, std::sqrt(max_admissible_t))*
		    1.0/(sigma_x_2_*sigma_y_2_*sqrt(1-rho_)*sqrt(1+rho_));
		}
	      }
      
	      for (double ll : ls) {
		if (std::abs(ll) > mmax) {
		  mmax = std::abs(ll);
		}
	      }
	    }
	  }

	  max_admissible_times[image_counter] = sign*max_admissible_t;
	  
	  image_counter = image_counter + 1;
	}
      }
    }
  }
  std::vector<double>::iterator result = std::max_element(max_admissible_times.begin(),
							  max_admissible_times.end());
  double biggest_time = *result;
  result = std::min_element(max_admissible_times.begin(),
			    max_admissible_times.end());
  double smallest_time = *result;

  std::vector<std::vector<BivariateImageWithTime>> max_t_images;
  for (unsigned i=0; i<final_images.size(); ++i) {
    if (std::abs( max_admissible_times[i] - biggest_time) <= std::numeric_limits<double>::epsilon()) {
      max_t_images.push_back( final_images[i] );
    }
  }

  biggest_time = std::min(biggest_time, t_2_);
  gsl_matrix* cov_mat = gsl_matrix_alloc(2,2);
  double out = 0;
  for (unsigned i=0; i<max_t_images.size(); ++i) {
    MultivariateNormal mvtnorm = MultivariateNormal();
    gsl_matrix_set(cov_mat, 0,0, sigma_x_2_*sigma_x_2_*t_2_);
    gsl_matrix_set(cov_mat, 0,1, rho_*sigma_x_2_*sigma_y_2_*t_2_);
    gsl_matrix_set(cov_mat, 1,0, rho_*sigma_x_2_*sigma_y_2_*t_2_);
    gsl_matrix_set(cov_mat, 1,1, sigma_y_2_*sigma_y_2_*t_2_);

    for (unsigned j=0; j<max_t_images[i].size(); ++j) {
      out = out + mvtnorm.dmvnorm(2,
				  input,
				  max_t_images[i][j].get_position(),
				  cov_mat)*max_t_images[i][j].get_mult_factor()/max_t_images.size();
    }
    
  }
  gsl_matrix_free(cov_mat);
  printf("in analytic_solution_small_t, biggest_time = %g\n", biggest_time);
  return out;
}
