#include <algorithm>
#include "BivariateSolver.hpp"
#include <chrono>
#include <cmath>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <stdexcept>

BivariateImageWithTime::BivariateImageWithTime()
  : position_(gsl_vector_alloc(2)),
    t_(0.0),
    mult_factor_(1.0),
    reflection_sequence_(std::vector<unsigned> (0))
{}

BivariateImageWithTime::BivariateImageWithTime(const BivariateImageWithTime& image_with_time)
  : position_(gsl_vector_alloc(2)),
    t_(0.0),
    mult_factor_(1.0),
    reflection_sequence_(std::vector<unsigned> (0))
{
  gsl_vector_memcpy(position_, image_with_time.get_position());
  t_ = image_with_time.get_t();
  mult_factor_ = image_with_time.get_mult_factor();
  reflection_sequence_ = image_with_time.get_reflection_sequence();
}

BivariateImageWithTime& BivariateImageWithTime::operator=(const BivariateImageWithTime& rhs)
{
  if (this==&rhs) {
    return *this;
  } else {
    gsl_vector_memcpy(position_, rhs.get_position());
    t_ = rhs.get_t();
    mult_factor_ = rhs.get_mult_factor();
    reflection_sequence_ = rhs.get_reflection_sequence();
    return *this;
  }
}

BivariateImageWithTime::BivariateImageWithTime(const gsl_vector* position,
					       double time,
					       double mult_factor)
  : position_(gsl_vector_calloc(2)),
    t_(time),
    mult_factor_(mult_factor),
    reflection_sequence_(std::vector<unsigned> (0))
{
  gsl_vector_memcpy(position_, position);
}

BivariateImageWithTime::BivariateImageWithTime(const gsl_vector* position,
					       double time,
					       double mult_factor,
					       const std::vector<unsigned>& refl)
  : position_(gsl_vector_calloc(2)),
    t_(time),
    mult_factor_(mult_factor),
    reflection_sequence_(refl)
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
						   y_0_2_);

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
  rho_ = rho;

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

void BivariateSolver::set_diffusion_parameters_and_data_small_t(double sigma_x,
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
  rho_ = rho;
  set_scaled_data();
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

    int x_int_for_linear = std::trunc(gsl_vector_get(scaled_input, 0)/dx_);
    int y_int_for_linear = std::trunc(gsl_vector_get(scaled_input, 1)/dx_);

    if (x_int_for_linear == 1/dx_) {
      x_int_for_linear = 1/dx_ - 1;
    }
    if (y_int_for_linear == 1/dx_) {
      y_int_for_linear = 1/dx_ - 1;
    }

    int x_int = std::trunc(gsl_vector_get(scaled_input, 0)/dx_);
    int y_int = std::trunc(gsl_vector_get(scaled_input, 1)/dx_);
    int N = 1/dx_ + 1;

    if (x_int % 2 != 0) {
      x_int -= 1;
    }
    if (x_int == N-1) {
      x_int = x_int-2;
    }

    if (y_int % 2 != 0) {
      y_int -= 1;
    }
    if (y_int == N-1) {
      y_int = y_int-2;
    }


    if ((x_int + 1 > N) || (x_int + 2 > N)) {
      throw std::out_of_range("x index out of range");
    }

    if ((y_int + 1 > N) || (y_int + 2 > N)) {
      std::cout << "N = " << N << std::endl;
      std::cout << "y = " << gsl_vector_get(scaled_input, 1) << std::endl;
      std::cout << "y_int = " << y_int << std::endl;
      throw std::out_of_range("y index out of range");
    }

    double x = gsl_vector_get(scaled_input, 0);
    double y = gsl_vector_get(scaled_input, 1);

    double x_1 = x_int*dx_;
    double x_2 = (x_int+1)*dx_;
    double x_3 = (x_int+2)*dx_;

    double y_1 = y_int*dx_;
    double y_2 = (y_int+1)*dx_;
    double y_3 = (y_int+2)*dx_;


    double x_1_for_linear = x_int_for_linear*dx_;
    double x_2_for_linear = (x_int_for_linear+1)*dx_;
    double y_1_for_linear = y_int_for_linear*dx_;
    double y_2_for_linear = (y_int_for_linear+1)*dx_;


    double f_11 = 0;
    double f_12 = 0;
    double f_13 = 0;

    double f_21 = 0;
    double f_22 = 0;
    double f_23 = 0;

    double f_31 = 0;
    double f_32 = 0;
    double f_33 = 0;


    for (unsigned i=0; i<basis_->get_orthonormal_elements().size(); ++i) {
      double current_f = 0;

      f_11 = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
      			    x_int,
      			    y_int);
      f_12 = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
      			    x_int,
      			    y_int+1);
      f_13 = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
      			    x_int,
      			    y_int+2);

      f_21 = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
      			    x_int+1,
      			    y_int);
      f_22 = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
      			    x_int+1,
      			    y_int+1);
      f_23 = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
      			    x_int+1,
      			    y_int+2);

      f_31 = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
      			    x_int+2,
      			    y_int);
      f_32 = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
      			    x_int+2,
      			    y_int+1);
      f_33 = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
      			    x_int+2,
      			    y_int+2);

      std::vector<double> function_vals = {f_11, f_12, f_13,
      					   f_21, f_22, f_23,
      					   f_31, f_32, f_33};

      double x_1_term = (x-x_2)*(x-x_3)/((x_1-x_2)*(x_1-x_3));
      double x_2_term = (x-x_1)*(x-x_3)/((x_2-x_1)*(x_2-x_3));
      double x_3_term = (x-x_1)*(x-x_2)/((x_3-x_1)*(x_3-x_2));

      double y_1_term = (y-y_2)*(y-y_3)/((y_1-y_2)*(y_1-y_3));
      double y_2_term = (y-y_1)*(y-y_3)/((y_2-y_1)*(y_2-y_3));
      double y_3_term = (y-y_1)*(y-y_2)/((y_3-y_1)*(y_3-y_2));


      current_f = 
	f_11 * x_1_term * y_1_term +
	f_12 * x_1_term * y_2_term +
	f_13 * x_1_term * y_3_term +
	// // 
	f_21 * x_2_term * y_1_term +
	f_22 * x_2_term * y_2_term +
	f_23 * x_2_term * y_3_term +
	// // 
	f_31 * x_3_term * y_1_term +
	f_32 * x_3_term * y_2_term +
	f_33 * x_3_term * y_3_term;

      // for (unsigned ii=0; ii<3; ++ii) {
      //  	for (unsigned jj=0; jj<3; ++jj) {
      // 	  std::vector<double> xs = {x_1, x_2, x_3};
      // 	  double x_curr = xs[ii];
      // 	  xs.erase(xs.begin() + ii);

      // 	  std::vector<double> ys = {y_1, y_2, y_3};
      // 	  double y_curr = ys[jj];
      // 	  ys.erase(ys.begin() + jj);

      // 	  std::vector<double> numerator = std::vector<double> (4);
      // 	  std::vector<double> denominator = std::vector<double> (4);
      // 	  // populate numerator vec
      // 	  std::generate(numerator.begin(), numerator.end(),
      // 			[&, n=0] () mutable {
      // 			  int x_ind = std::trunc(n/2);
      // 			  int y_ind = n-2*x_ind;
      // 			  n++;
      // 			  return (x - xs[x_ind])*(y - ys[y_ind]);
      // 			} );

      // 	  // populate denominator vec
      // 	  std::generate(denominator.begin(), denominator.end(),
      // 			[&, n=0] () mutable {
      // 			  int x_ind = std::trunc(n/2);
      // 			  int y_ind = n-2*x_ind;
      // 			  n++;
      // 			  return (x_curr - xs[x_ind])*(y_curr - ys[y_ind]);
      // 			} );

      // 	  double num_prod = 1.0;
      // 	  double denom_prod = 1.0;
      // 	  for (const double& nn : numerator) { num_prod = num_prod*nn; }
      // 	  for (const double& dd : denominator) { denom_prod = denom_prod*dd; }

      // 	  if (std::isinf(num_prod/denom_prod)) {
      // 	    for (const double& dd : numerator) {
      // 	      std::cout << dd << " ";
      // 	    }
      // 	    std::cout << std::endl;
      // 	    for (const double& dd : denominator) {
      // 	      std::cout << dd << " ";
      // 	    }
      // 	    std::cout << std::endl;

      // 	    throw std::domain_error("denom is zero");
      // 	  }

      // 	  current_f += num_prod/denom_prod * function_vals[ii*3 + jj];
      //  	}
      // }

      double current_f_for_linear = 0.0;
      double f_11_for_linear = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
					      x_int_for_linear,
					      y_int_for_linear);
      double f_12_for_linear = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
					      x_int_for_linear,
					      y_int_for_linear+1);
      double f_21_for_linear = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
					      x_int_for_linear+1,
					      y_int_for_linear);
      double f_22_for_linear = gsl_matrix_get(basis_->get_orthonormal_element(i).get_function_grid(),
					      x_int_for_linear+1,
					      y_int_for_linear+1);
      current_f_for_linear = 1.0/((x_2_for_linear-x_1_for_linear)*(y_2_for_linear-y_1_for_linear)) *
    	((x_2_for_linear - x) * (f_11_for_linear*(y_2_for_linear-y) + f_12_for_linear*(y-y_1_for_linear)) +
    	 (x - x_1_for_linear) * (f_21_for_linear*(y_2_for_linear-y) + f_22_for_linear*(y-y_1_for_linear)));

      // if (std::abs(current_f_for_linear - current_f) > std::numeric_limits<double>::epsilon()) {
      // 	std::vector<double> xs = {x_1, x_2, x_3};
      // 	std::vector<double> ys = {y_1, y_2, y_3};
      // 	unsigned n = 0;

      // 	std::cout << "\n";
      // 	std::cout << "x = " << std::setprecision(16) << x << ";\n";
      // 	std::cout << "y = " << std::setprecision(16) << y << ";\n";
      // 	std::cout << "dx = " << std::setprecision(16) << dx_ << ";\n";
      // 	for (const double& ff : function_vals) {
      // 	  unsigned ii = std::trunc(n/3);
      // 	  unsigned jj = n-3*ii;
      // 	  n ++;
      // 	  std::cout << "f_" << ii << jj << " = " << std::setprecision(16) <<  ff << ";\n";
      // 	}
      // 	n = 0;

      // 	for (n=0; n<3; ++n) {
      // 	  std::cout << "x_" << n << " = " << xs[n] << ";\n";
      // 	  std::cout << "y_" << n << " = " << ys[n] << ";\n";
      // 	}
      // 	n=0;

      // 	std::cout << "f_11_for_linear = " << std::setprecision(16) << f_11_for_linear << ";\n";
      // 	std::cout << "f_12_for_linear = " << std::setprecision(16) << f_12_for_linear << ";\n";
      // 	std::cout << "f_21_for_linear = " << std::setprecision(16) << f_21_for_linear << ";\n";
      // 	std::cout << "f_22_for_linear = " << std::setprecision(16) << f_22_for_linear << ";\n";

      // 	std::cout << "current_f_for_linear = " << std::setprecision(16) << current_f_for_linear << ";\n";
      // 	std::cout << "current_f = " << std::setprecision(16) << current_f << ";" << std::endl;



      // 	throw std::domain_error("difference between linear and quad interpolation too big");
      // }

      // current_f = 1.0/((x_2-x_1)*(y_2-y_1)) *
      // 	((x_2 - x) * (f_11*(y_2-y) + f_12*(y-y_1)) +
      // 	 (x - x_1) * (f_21*(y_2-y) + f_22*(y-y_1)));

      current_f = current_f * gsl_vector_get(solution_coefs_, i);

      out = out + current_f;
    }

  }

  double Lx_2 = b_ - a_;
  double Ly_2 = d_ - c_;
  out = out / (Lx_2 * Ly_2); // * std::max(sigma_x_2_*sigma_x_2_,
			    // 	       sigma_y_2_*sigma_y_2_);

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

    // std::cout << "d1x[" << n << "] = " << d1x[i] << " ";
  }
  // std::cout << std::endl;

  double out_x = (1.0/std::sqrt(2.0*M_PI*std::pow(sigma_x_,2)*t_))*
    sum_x;

  double out_y = (1.0/std::sqrt(2.0*M_PI*std::pow(sigma_y_,2)*t_))*
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

  double log_sum_x = 0.0;
  double log_sum_y = 0.0;

  std::vector<double> log_terms_x (2*little_n+1);
  std::vector<double> log_terms_y (2*little_n+1);

  std::vector<int> sign_terms_x (2*little_n+1);
  std::vector<int> sign_terms_y (2*little_n+1);

  for (int i=0; i<2*little_n+1; ++i) {
    int n = i - little_n;

    double term_x = (4.0*n*n*(2.0*d1x[i] - 1)*std::exp(-d1x[i]) - 4.0*n*(n-1)*(2.0*d2x[i] - 1)*std::exp(-d2x[i]));
    double term_y = (4.0*n*n*(2.0*d1y[i] - 1)*std::exp(-d1y[i]) - 4.0*n*(n-1)*(2.0*d2y[i] - 1)*std::exp(-d2y[i]));

    if (term_x > 0.0) {
      sign_terms_x[i] = 1;
    } else {
      sign_terms_x[i] = -1;
    }

    if (term_y > 0.0) {
      sign_terms_y[i] = 1;
    } else {
      sign_terms_y[i] = -1;
    }

    log_terms_x[i] =
      std::log(std::abs((4.0*n*n*(2.0*d1x[i] - 1)*std::exp(-d1x[i]) - 4.0*n*(n-1)*(2.0*d2x[i] - 1)*std::exp(-d2x[i]))));

    log_terms_y[i] =
      std::log(std::abs((4.0*n*n*(2.0*d1y[i] - 1)*std::exp(-d1y[i]) - 4.0*n*(n-1)*(2.0*d2y[i] - 1)*std::exp(-d2y[i]))));
  }

  std::vector<double>::iterator max_x = std::max_element(log_terms_x.begin(), log_terms_x.end());
  std::vector<double>::iterator max_y = std::max_element(log_terms_y.begin(), log_terms_y.end());

  double factored_sum_x = 0;
  double factored_sum_y = 0;
  for (int i=0; i<2*little_n+1; ++i) {
    factored_sum_x = factored_sum_x + sign_terms_x[i]*std::exp(log_terms_x[i]-*max_x);
    factored_sum_y = factored_sum_y + sign_terms_y[i]*std::exp(log_terms_y[i]-*max_y);
  }

  double log_deriv_x = -std::log(std::sqrt(2*M_PI)*std::pow(sigma_x_*std::sqrt(t_), 3)) + *max_x + log(factored_sum_x);
  double log_deriv_y = -std::log(std::sqrt(2*M_PI)*std::pow(sigma_y_*std::sqrt(t_), 3)) + *max_y + log(factored_sum_y);

  double out = log_deriv_x + log_deriv_y;
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

double BivariateSolver::analytic_likelihood_ax_bx(const gsl_vector* input,
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
  }

  double deriv_x = 0.0;
  double out_y = 0.0;

  double sum_x = 0.0;
  double sum_y = 0.0;

  for (int i=0; i<2*little_n+1; ++i) {
    int n = i - little_n;

    sum_x = sum_x +
      (4.0*n*n*(2.0*d1x[i] - 1)*std::exp(-d1x[i]) - 4.0*n*(n-1)*(2.0*d2x[i] - 1)*std::exp(-d2x[i]));

    sum_y = sum_y +
      (std::exp(-d1y[i]) - std::exp(-d2y[i]));

  }

  deriv_x = 1.0/(std::sqrt(2*M_PI)*std::pow(sigma_x_*std::sqrt(t_), 3)) * sum_x;
  out_y = (1.0/std::sqrt(2.0*M_PI*std::pow(sigma_y_,2)*t_)) * sum_y;

  double out = deriv_x*out_y;

  return (out);
}

double BivariateSolver::analytic_likelihood_ax_bx_ay(const gsl_vector* input,
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
      (-1.0*std::exp(-d1y[i])*(gsl_vector_get(input,1) - y_0_ - 2.0*n*(b_-a_))/(std::pow(sigma_y_, 2) * t_)*(2.0*n) -
       //
       -1.0*std::exp(-d2y[i])*(gsl_vector_get(input,1) + y_0_ - 2.0*a_ - 2.0*n*(b_-a_))/(std::pow(sigma_y_, 2) * t_)*(-2.0+2.0*n));
  }

  deriv_x = 1.0/(std::sqrt(2*M_PI)*std::pow(sigma_x_*std::sqrt(t_), 3)) * sum_x;
  deriv_y = (1.0/std::sqrt(2.0*M_PI*std::pow(sigma_y_,2)*t_))*sum_y;

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

  double dPdax = 0;
  double log_CCC = -1.0*(log(2.0)+log(t_)+log(1-rho_*rho_)+2.0*log(sigma_y_));
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=1; } else { a_power=0; };

    set_data(current_a + a_indeces[i]*h_x,
	     current_x_0,
	     current_b,
	     current_c,
	     current_y_0,
	     current_d);

    positions = small_t_image_positions_ax();

    double x_0 = gsl_vector_get(positions[0].get_position(),0)*(b_-a_) +
      a_indeces[i]*h_x;
    double y_0 = gsl_vector_get(positions[0].get_position(),1)*(d_-c_);
    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);

    double polynomial = std::pow((x-x_0)/sigma_x_,2) +
	    std::pow((y-y_0)/sigma_y_,2) -
	    2*rho_/(sigma_x_*sigma_y_)*(x-x_0)*(y-y_0);

    dPdax = dPdax +
      polynomial*
      std::pow(-1, a_power);
  }
  dPdax = dPdax / h_x;
  std::cout << "DP_DAX = " << dPdax << std::endl;
  std::cout << "G*C*dPdax = "
	    << exp(log_before_small_t + log_CC)*dPdax << "\n  "
	    << std::exp(log_before_small_t + log_CC + log(dPdax)) << std::endl;


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
	    //std::exp(smallt-log_before_small_t-log_CC)*
	    polynomial*
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
  derivative = derivative/(h_x)* exp(log_before_small_t + log_CC);
  std::cout << "The solution with numerical polynomail deriv is = " << derivative << std::endl;
  derivative_with_sol = derivative_with_sol/(h_x);
  std::cout << "The deriv with solution is = "
	    << derivative_with_sol << std::endl;
  std::cout << "The analytic deriv from _ax() is = " << analytic_likelihood_ax(raw_input, 10000) << std::endl;

  return derivative;
}

double BivariateSolver::
numerical_likelihood_first_order_small_t_ax_bx(const gsl_vector* raw_input,
					       double h)
{
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
  double CC = -1.0/(2.0*t_*(1-rho_*rho_));

  gsl_matrix* cov_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_set(cov_matrix, 0,0,
  		 sigma_x_*sigma_x_*t_);
  gsl_matrix_set(cov_matrix, 0,1,
  		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,0,
  		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,1,
  		 sigma_y_*sigma_y_*t_);


  std::vector<BivariateImageWithTime> positions =
    small_t_image_positions_1_3(false);


  // precalculating positions
  std::vector<std::vector<BivariateImageWithTime>> perturbed_positions (16, std::vector<BivariateImageWithTime> (0));
  std::vector<std::vector<double>> perturbed_polynomials (16, std::vector<double> (positions.size()));

  auto t1 = std::chrono::high_resolution_clock::now();
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    for (unsigned j=0; j<b_indeces.size(); ++j) {
      if (j==0) { b_power=0; } else { b_power=1; };

      for (unsigned k=0; k<c_indeces.size(); ++k) {
	if (k==0) { c_power=0; } else { c_power=1; };

	for (unsigned l=0; l<d_indeces.size(); ++l) {
	  if (l==0) { d_power=0; } else { d_power=1; };

	  set_data_for_small_t(current_a + a_indeces[i]*h_x,
			       current_x_0,
			       current_b + b_indeces[j]*h_x,
			       current_c + c_indeces[k]*h_y,
			       current_y_0,
			       current_d + d_indeces[l]*h_y);

	  positions = small_t_image_positions_1_3(false);
	  unsigned index = i*1 + j*2 + k*4 + l*8;

	  perturbed_positions[index] = positions;

	  double x = gsl_vector_get(raw_input, 0);
	  double y = gsl_vector_get(raw_input, 1);
	  for (unsigned ii=0; ii<positions.size(); ++ii) {
	    double x_0 =
	      gsl_vector_get(positions[ii].get_position(),0)*(b_-a_) +
	      a_indeces[i]*h_x;
	    double y_0 =
	      gsl_vector_get(positions[ii].get_position(),1)*(d_-c_) +
	      c_indeces[k]*h_y;

	    double polynomial =
	      std::pow((x-x_0)/sigma_x_,2) +
	      std::pow((y-y_0)/sigma_y_,2) -
	      2*rho_/(sigma_x_*sigma_y_)*(x-x_0)*(y-y_0);

	    perturbed_polynomials[index][ii] = polynomial;
	  }
	}
      }
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for positions = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  unsigned counter = 0;
  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdaxs (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };
    unsigned index = i*1 + 1*2 + 0*4 + 1*8;
    positions = perturbed_positions[index];

    for (unsigned ii=0; ii<positions.size(); ++ii) {
      dPdaxs[ii] = dPdaxs[ii] +
      	perturbed_polynomials[index][ii]*
      	std::pow(-1, a_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdaxs[ii] = dPdaxs[ii]/h_x;
    printf("dPdax[%i] = %g\n", ii, dPdaxs[ii]);
  }

  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for dPdaxs = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdbxs (positions.size(), 0);
  for (unsigned j=0; j<b_indeces.size(); ++j) {
    if (j==0) { b_power=0; } else { b_power=1; };
    unsigned index = 0*1 + j*2 + 0*4 + 1*8;
    positions = perturbed_positions[index];

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      dPdbxs[ii] = dPdbxs[ii] +
	perturbed_polynomials[index][ii]*
  	std::pow(-1, b_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdbxs[ii] = dPdbxs[ii]/h_x;
    // printf("dPdbxs[%i] = %g\n", ii, dPdbxs[ii]);
  }
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for dPdbxs = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;


  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdays (positions.size(), 0.0);
  for (unsigned k=0; k<c_indeces.size(); ++k) {
    if (k==0) { c_power=0; } else { c_power=1; };
    unsigned index = 0*1 + 1*2 + k*4 + 1*8;
    positions = perturbed_positions[index];

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      dPdays[ii] = dPdays[ii] +
	perturbed_polynomials[index][ii]*
  	std::pow(-1, c_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdays[ii] = dPdays[ii]/h_y;
    printf("dPdays[%i] = %g\n", ii, dPdays[ii]);
  }
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for dPdays = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  std::vector<double> dPdbys (positions.size(), 0);
  for (unsigned l=0; l<d_indeces.size(); ++l) {
    if (l==0) { d_power=0; } else { d_power=1; };
    unsigned index = 0*1 + 1*2 + 0*4 + l*8;
    positions = perturbed_positions[index];

    for (unsigned ii=0; ii<positions.size(); ++ii) {
      dPdbys[ii] = dPdbys[ii] +
	perturbed_polynomials[index][ii]*
	std::pow(-1, d_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdbys[ii] = dPdbys[ii]/h_y;
    printf("dPdbys[%i] = %g\n", ii, dPdbys[ii]);
  }

  std::vector<double> ddPdaxdbxs (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    for (unsigned j=0; j<b_indeces.size(); ++j) {
      if (j==0) { b_power=0; } else { b_power=1; };

      unsigned index = i*1 + j*2 + 0*4 + 1*8;
      for (unsigned ii=0; ii<positions.size(); ++ii) {
	ddPdaxdbxs[ii] = ddPdaxdbxs[ii] +
	  perturbed_polynomials[index][ii]/(h_x*h_x)*
	  std::pow(-1, a_power)*std::pow(-1, b_power);
      }
    }
  }
  std::vector<double> ddPdaxdays (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    for (unsigned k=0; k<c_indeces.size(); ++k) {
      if (k==0) { c_power=0; } else { c_power=1; };

      unsigned index = i*1 + 1*2 + k*4 + 1*8;
      for (unsigned ii=0; ii<positions.size(); ++ii) {
	ddPdaxdays[ii] = ddPdaxdays[ii] +
	  perturbed_polynomials[index][ii]/(h_x*h_y)*
	  std::pow(-1, a_power)*std::pow(-1, c_power);
      }
    }
  }
  std::vector<double> ddPdaxdbys (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    for (unsigned l=0; l<c_indeces.size(); ++l) {
      if (l==0) { d_power=0; } else { d_power=1; };

      unsigned index = i*1 + 1*2 + 0*4 + l*8;
      for (unsigned ii=0; ii<positions.size(); ++ii) {
	ddPdaxdbys[ii] = ddPdaxdbys[ii] +
	  perturbed_polynomials[index][ii]/(h_x*h_y)*
	  std::pow(-1, a_power)*std::pow(-1, d_power);
      }
    }
  }
  std::vector<double> ddPdbxdays (positions.size(), 0);
  for (unsigned j=0; j<a_indeces.size(); ++j) {
    if (j==0) { b_power=0; } else { b_power=1; };

    for (unsigned k=0; k<c_indeces.size(); ++k) {
      if (k==0) { c_power=0; } else { c_power=1; };

      unsigned index = 0*1 + j*2 + k*4 + 1*8;
      for (unsigned ii=0; ii<positions.size(); ++ii) {
	ddPdbxdays[ii] = ddPdbxdays[ii] +
	  perturbed_polynomials[index][ii]/(h_x*h_y)*
	  std::pow(-1, b_power)*std::pow(-1, c_power);
      }
    }
  }
  std::vector<double> ddPdbxdbys (positions.size(), 0);
  for (unsigned j=0; j<b_indeces.size(); ++j) {
    if (j==0) { b_power=0; } else { b_power=1; };

    for (unsigned l=0; l<d_indeces.size(); ++l) {
      if (l==0) { d_power=0; } else { d_power=1; };

      unsigned index = 0*1 + j*2 + 0*4 + l*8;
      for (unsigned ii=0; ii<positions.size(); ++ii) {
	ddPdbxdbys[ii] = ddPdbxdbys[ii] +
	  perturbed_polynomials[index][ii]/(h_x*h_y)*
	  std::pow(-1, b_power)*std::pow(-1, d_power);
      }
    }
  }
  std::vector<double> ddPdaydbys (positions.size(), 0);
  for (unsigned k=0; k<c_indeces.size(); ++k) {
    if (k==0) { c_power=0; } else { c_power=1; };

    for (unsigned l=0; l<c_indeces.size(); ++l) {
      if (l==0) { d_power=0; } else { d_power=1; };

      unsigned index = 0*1 + 1*2 + k*4 + l*8;
      for (unsigned ii=0; ii<positions.size(); ++ii) {
	ddPdaydbys[ii] = ddPdaydbys[ii] +
	  perturbed_polynomials[index][ii]/(h_y*h_y)*
	  std::pow(-1, c_power)*std::pow(-1, d_power);
      }
    }
  }

  std::vector<double> dddPdaxdbxday (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    for (unsigned j=0; j<b_indeces.size(); ++j) {
      if (j==0) { b_power=0; } else { b_power=1; };

      for (unsigned k=0; k<c_indeces.size(); ++k) {
	if (k==0) { c_power=0; } else { c_power=1; };

	unsigned index = i*1 + j*2 + k*4 + 1*8;
	for (unsigned ii=0; ii<positions.size(); ++ii) {
	  dddPdaxdbxday[ii] = dddPdaxdbxday[ii] +
	    perturbed_polynomials[index][ii]/(h_x*h_x*h_y)*
	    std::pow(-1, a_power)*std::pow(-1, b_power)*std::pow(-1, c_power);
	}
      }
    }
  }
  std::vector<double> dddPdaxdbxdby (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    for (unsigned j=0; j<b_indeces.size(); ++j) {
      if (j==0) { b_power=0; } else { b_power=1; };

	for (unsigned l=0; l<d_indeces.size(); ++l) {
	  if (l==0) { d_power=0; } else { d_power=1; };

	  unsigned index = i*1 + j*2 + 0*4 + l*8;
	  for (unsigned ii=0; ii<positions.size(); ++ii) {
	    dddPdaxdbxdby[ii] = dddPdaxdbxdby[ii] +
	      perturbed_polynomials[index][ii]/(h_x*h_x*h_y)*
	      std::pow(-1, a_power)*std::pow(-1, b_power)*
	      std::pow(-1, d_power);
	  }
	}
    }
  }
  std::vector<double> dddPdaxdaydby (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

      for (unsigned k=0; k<c_indeces.size(); ++k) {
	if (k==0) { c_power=0; } else { c_power=1; };

	for (unsigned l=0; l<d_indeces.size(); ++l) {
	  if (l==0) { d_power=0; } else { d_power=1; };

	  unsigned index = i*1 + 1*2 + k*4 + l*8;
	  for (unsigned ii=0; ii<positions.size(); ++ii) {
	    dddPdaxdaydby[ii] = dddPdaxdaydby[ii] +
	      perturbed_polynomials[index][ii]/(h_x*h_y*h_y)*
	      std::pow(-1, a_power)*std::pow(-1, c_power)*std::pow(-1, d_power);
	  }
	}
      }
  }
  std::vector<double> dddPdbxdaydby (positions.size(), 0);
  for (unsigned j=0; j<b_indeces.size(); ++j) {
    if (j==0) { b_power=0; } else { b_power=1; };

    for (unsigned k=0; k<c_indeces.size(); ++k) {
      if (k==0) { c_power=0; } else { c_power=1; };

      for (unsigned l=0; l<d_indeces.size(); ++l) {
	if (l==0) { d_power=0; } else { d_power=1; };

	unsigned index = 0*1 + j*2 + k*4 + l*8;
	for (unsigned ii=0; ii<positions.size(); ++ii) {
	  dddPdbxdaydby[ii] = dddPdbxdaydby[ii] +
	    perturbed_polynomials[index][ii]/(h_x*h_y*h_y)*
	    std::pow(-1, b_power)*
	    std::pow(-1, c_power)*std::pow(-1, d_power);
	}
      }
    }
  }


  std::vector<double> ddddP (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    for (unsigned j=0; j<b_indeces.size(); ++j) {
      if (j==0) { b_power=0; } else { b_power=1; };

      for (unsigned k=0; k<c_indeces.size(); ++k) {
	if (k==0) { c_power=0; } else { c_power=1; };

	for (unsigned l=0; l<d_indeces.size(); ++l) {
	  if (l==0) { d_power=0; } else { d_power=1; };

	  unsigned index = i*1 + j*2 + k*4 + l*8;
	  for (unsigned ii=0; ii<positions.size(); ++ii) {
	    ddddP[ii] = ddddP[ii] +
	      perturbed_polynomials[index][ii]/(h_x*h_x*h_y*h_y)*
	      std::pow(-1, a_power)*std::pow(-1, b_power)*
	      std::pow(-1, c_power)*std::pow(-1, d_power);
	  }
	}
      }
    }
  }

  // COMPUTING CONTRIBUTIONS OF EACH IMAGE
  t1 = std::chrono::high_resolution_clock::now();
  set_data_for_small_t(current_a,
		       current_x_0,
		       current_b,
		       current_c,
		       current_y_0,
		       current_d);
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for big set data = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  positions = small_t_image_positions_1_3(false);
  double out = 0;
  MultivariateNormal mvtnorm = MultivariateNormal();
  std::vector<int> terms_signs = std::vector<int> (positions.size());
  std::vector<double> log_terms = std::vector<double> (positions.size());

  for (unsigned ii=0; ii<positions.size(); ++ii) {
    const BivariateImageWithTime& differentiable_image = positions[ii];

    // log( G*C^4*dPdax*dPdbx*dPday*dPdby )
    double log_G = mvtnorm.dmvnorm_log(2,
				       raw_input,
				       differentiable_image.get_position(),
				       cov_matrix);

    terms_signs[ii] = 1;
    if (differentiable_image.get_mult_factor()*
	((std::exp(4*log_CC)*
	 dPdaxs[ii]*dPdbxs[ii]*
	  dPdays[ii]*dPdbys[ii])) //  +
	 // // // // // //
	 // -std::exp(3*log_CC)*
	 // (ddPdaxdbxs[ii]*dPdbys[ii]*dPdays[ii] +
	 //  ddPdaxdays[ii]*dPdbxs[ii]*dPdbys[ii] +
	 //  ddPdaxdbys[ii]*dPdbxs[ii]*dPdays[ii] +
	 //  ddPdbxdays[ii]*dPdaxs[ii]*dPdbys[ii] +
	 //  ddPdbxdbys[ii]*dPdaxs[ii]*dPdays[ii] +
	 //  ddPdaydbys[ii]*dPdaxs[ii]*dPdbxs[ii]) +
	 // // // // // //
	 // std::exp(2*log_CC)*
	 // (dddPdaxdbxday[ii]*dPdbys[ii] +
	 //  ddPdaxdbxs[ii]*ddPdaydbys[ii] +
	 //  dddPdaxdbxdby[ii]*dPdays[ii] +
	 //  dddPdaxdaydby[ii]*dPdbxs[ii] +
	 //  ddPdaxdays[ii]*ddPdbxdbys[ii] +
	 //  dddPdbxdaydby[ii]*dPdaxs[ii] +
	 //  ddPdbxdays[ii]*dPdaxs[ii]*dPdbys[ii])
	 // // // // //
	//-std::exp(log_CC)*ddddP[ii])
	< 0)
      {
	terms_signs[ii] = -1;
      }

    log_terms[ii] = log_G +
      log(std::exp(4*log_CC)*
	  dPdaxs[ii]*dPdbxs[ii]*
	  dPdays[ii]*dPdbys[ii]);//  +
	  // // // // //
	  // -std::exp(3*log_CC)*
	  // (ddPdaxdbxs[ii]*dPdbys[ii]*dPdays[ii] +
	  //  ddPdaxdays[ii]*dPdbxs[ii]*dPdbys[ii] +
	  //  ddPdaxdbys[ii]*dPdbxs[ii]*dPdays[ii] +
	  //  ddPdbxdays[ii]*dPdaxs[ii]*dPdbys[ii] +
	  //  ddPdbxdbys[ii]*dPdaxs[ii]*dPdays[ii] +
	  //  ddPdaydbys[ii]*dPdaxs[ii]*dPdbxs[ii]) +
	  // // // // // //
	  //  std::exp(2*log_CC)*
	  //  (dddPdaxdbxday[ii]*dPdbys[ii] +
	  //   ddPdaxdbxs[ii]*ddPdaydbys[ii] +
	  //   dddPdaxdbxdby[ii]*dPdays[ii] +
	  //   dddPdaxdaydby[ii]*dPdbxs[ii] +
	  //   ddPdaxdays[ii]*ddPdbxdbys[ii] +
	  //   dddPdbxdaydby[ii]*dPdaxs[ii] +
	  //   ddPdbxdays[ii]*dPdaxs[ii]*dPdbys[ii]) +
	  // // // // // // //
	  // -std::exp(log_CC)*ddddP[ii]);
    std::cout << "term[" << ii << "] = " << exp(log_terms[ii]-log_G)*terms_signs[ii]
	      << " ";
    for (const unsigned& reflection : differentiable_image.get_reflection_sequence()) {
      std::cout << reflection << " ";
    }
    std::cout << " sign = " << terms_signs[ii];
    std::cout << " log_G = " << log_G;
    std::cout << " dPdax = " << dPdaxs[ii]
	      << " dPdbx = " << dPdbxs[ii]
	      << " dPday = " << dPdays[ii]
	      << " dPdby = " << dPdbys[ii];
    std::cout << std::endl;

   }

  std::vector<double>::iterator result = std::max_element(log_terms.begin(),
							  log_terms.end());
  out = 0;
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    out = out +
      terms_signs[ii]*std::exp(log_terms[ii]-*result);
  }
  std::cout << "out before taking log = " << out << std::endl;
  out = std::log(out) + *result;

  gsl_matrix_free(cov_matrix);
  return (out);
}

double BivariateSolver::numerical_likelihood_first_order_small_t_ax_bx_ay(const gsl_vector* raw_input,
									  double small_t,
									  double h)
{
  printf("in NUMERICAL_likelihood_first_order_small_t with rho = %f\n",
	 rho_);
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
  double CC = -1.0/(2.0*t_*(1-rho_*rho_));

  gsl_matrix* cov_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_set(cov_matrix, 0,0,
  		 sigma_x_*sigma_x_*t_);
  gsl_matrix_set(cov_matrix, 0,1,
  		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,0,
  		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,1,
  		 sigma_y_*sigma_y_*t_);

  std::vector<BivariateImageWithTime> positions =
    small_t_image_positions_ax_bx_ay();
  for (const BivariateImageWithTime& position : positions) {
    printf("(%g, %g), (%g,%g), t=%g\n",
	   x_0_,
	   y_0_,
	   gsl_vector_get(position.get_position(),0),
	   gsl_vector_get(position.get_position(),1),
	   position.get_t());
  }
  std::cout << "positions.size()=" << positions.size() << std::endl;

  MultivariateNormal mvtnorm = MultivariateNormal();
  std::vector<double> log_before_small_ts (positions.size(), 0.0);
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    log_before_small_ts[ii] =
       mvtnorm.dmvnorm_log(2,
			   positions[ii].get_position(),
			   raw_input,
			   cov_matrix);
  }

  std::vector<double> dPdaxs (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    set_data(current_a + a_indeces[i]*h_x,
  	     current_x_0,
  	     current_b,
  	     current_c,
  	     current_y_0,
  	     current_d);
    positions = small_t_image_positions_ax_bx_ay();

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);

    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0_star = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_) +
	a_indeces[i]*h_x;
      double y_0_star = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      double polynomial = std::pow((x-x_0_star)/sigma_x_,2) +
	std::pow((y-y_0_star)/sigma_y_,2) -
	2*rho_/(sigma_x_*sigma_y_)*(x-x_0_star)*(y-y_0_star);

      dPdaxs[ii] = dPdaxs[ii] + polynomial*
	std::pow(-1, a_power);
    }
  }
  for (unsigned ii=0; ii<dPdaxs.size(); ++ii) {
    dPdaxs[ii] = dPdaxs[ii]/h_x;
  }
  std::for_each(dPdaxs.begin(), dPdaxs.end(),
		[](double &dPdax){ std::cout << "dPdax = " << dPdax << std::endl; });

  std::vector<double> dPdbxs (positions.size(),0);
  for (unsigned j=0; j<b_indeces.size(); ++j) {
    if (j==0) { b_power=0; } else { b_power=1; };

    set_data(current_a,
  	     current_x_0,
  	     current_b + b_indeces[j]*h_x,
  	     current_c,
  	     current_y_0,
  	     current_d);
    positions = small_t_image_positions_ax_bx_ay();

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      dPdbxs[ii] = dPdbxs[ii] + (std::pow((x-x_0)/sigma_x_,2.0) +
				 std::pow((y-y_0)/sigma_y_,2.0) +
				 -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
	std::pow(-1, b_power);
    }
  }
  for (unsigned ii=0; ii<dPdbxs.size(); ++ii) {
    dPdbxs[ii] = dPdbxs[ii]/h_x;
  }
  std::for_each(dPdbxs.begin(), dPdbxs.end(),
		[](double &dPdbx){ std::cout << "dPdbx = "
					     << dPdbx << std::endl; });

  std::vector<double> dPdays (positions.size(), 0);
  for (unsigned k=0; k<c_indeces.size(); ++k) {
    if (k==0) { c_power=0; } else { c_power=1; };

    set_data(current_a,
  	     current_x_0,
  	     current_b,
  	     current_c + h_y*c_indeces[k],
  	     current_y_0,
  	     current_d);
    positions = small_t_image_positions_ax_bx_ay();

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);

    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_) +
	h_y*c_indeces[k];

      dPdays[ii] = dPdays[ii] + (std::pow((x-x_0)/sigma_x_,2.0) +
  		     std::pow((y-y_0)/sigma_y_,2.0) +
  		     -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
      std::pow(-1, c_power);
    }
  }
  for (unsigned ii=0; ii<dPdays.size(); ++ii) {
    dPdays[ii] = dPdays[ii]/h_x;
  }
  std::for_each(dPdays.begin(), dPdays.end(),
		[](double &dPday){ std::cout << "dPday = "
					     << dPday << std::endl; });

  double ddPdaxdbx = 0;
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    for (unsigned j=0; j<b_indeces.size(); ++j) {
      if (j==0) { b_power=0; } else { b_power=1; };

      set_data(current_a + a_indeces[i]*h_x,
	       current_x_0,
	       current_b + b_indeces[j]*h_x,
	       current_c,
	       current_y_0,
	       current_d);
      positions = small_t_image_positions_ax_bx_ay();

      double x_0 = gsl_vector_get(positions[1].get_position(),0)*(b_-a_) +
	a_indeces[i]*h_x;
      double y_0 = gsl_vector_get(positions[1].get_position(),1)*(d_-c_);
      double x = gsl_vector_get(raw_input, 0);
      double y = gsl_vector_get(raw_input, 1);

      ddPdaxdbx = ddPdaxdbx + (std::pow((x-x_0)/sigma_x_,2.0) +
			       std::pow((y-y_0)/sigma_y_,2.0) +
			       -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
	std::pow(-1, a_power)*
	std::pow(-1, b_power);
    }
  }
  ddPdaxdbx = ddPdaxdbx / (h_x*h_x);

  double ddPdaxday = 0;
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    for (unsigned k=0; k<b_indeces.size(); ++k) {
      if (k==0) { c_power=0; } else { c_power=1; };

      set_data(current_a + a_indeces[i]*h_x,
	       current_x_0,
	       current_b,
	       current_c + c_indeces[k]*h_y,
	       current_y_0,
	       current_d);
      positions = small_t_image_positions_ax_bx_ay();

      double x_0 = gsl_vector_get(positions[1].get_position(),0)*(b_-a_) +
	a_indeces[i]*h_x;
      double y_0 = gsl_vector_get(positions[1].get_position(),1)*(d_-c_) +
	c_indeces[k]*h_y;
      double x = gsl_vector_get(raw_input, 0);
      double y = gsl_vector_get(raw_input, 1);

      ddPdaxday = ddPdaxday + (std::pow((x-x_0)/sigma_x_,2.0) +
			       std::pow((y-y_0)/sigma_y_,2.0) +
			       -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
	std::pow(-1, a_power)*
	std::pow(-1, c_power);
    }
  }
  ddPdaxday = ddPdaxday / (h_x*h_y);
  std::cout << "ddPdaxday = " << ddPdaxday << std::endl;

  double ddPdbxday = 0;
  for (unsigned j=0; j<b_indeces.size(); ++j) {
    if (j==0) { b_power=0; } else { b_power=1; };

    for (unsigned k=0; k<b_indeces.size(); ++k) {
      if (k==0) { c_power=0; } else { c_power=1; };

      set_data(current_a,
	       current_x_0,
	       current_b + b_indeces[j]*h_x,
	       current_c + c_indeces[k]*h_y,
	       current_y_0,
	       current_d);
      positions = small_t_image_positions_ax_bx_ay();

      double x_0 = gsl_vector_get(positions[1].get_position(),0)*(b_-a_);
      double y_0 = gsl_vector_get(positions[1].get_position(),1)*(d_-c_) +
	c_indeces[k]*h_y;
      double x = gsl_vector_get(raw_input, 0);
      double y = gsl_vector_get(raw_input, 1);

      ddPdbxday = ddPdbxday + (std::pow((x-x_0)/sigma_x_,2.0) +
			       std::pow((y-y_0)/sigma_y_,2.0) +
			       -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
	std::pow(-1, b_power)*
	std::pow(-1, c_power);
    }
  }
  ddPdbxday = ddPdbxday / (h_x*h_y);
  std::cout << "ddPdbxday = " << ddPdbxday << std::endl;

  double dddPdaxdbxday = 0;
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    for (unsigned j=0; j<b_indeces.size(); ++j) {
      if (j==0) { b_power=0; } else { b_power=1; };

      for (unsigned k=0; k<b_indeces.size(); ++k) {
	if (k==0) { c_power=0; } else { c_power=1; };

	set_data(current_a + a_indeces[i]*h_x,
		 current_x_0,
		 current_b + b_indeces[j]*h_x,
		 current_c + c_indeces[k]*h_y,
		 current_y_0,
		 current_d);
	positions = small_t_image_positions_ax_bx_ay();

	double x_0 = gsl_vector_get(positions[1].get_position(),0)*(b_-a_) +
	  a_indeces[i]*h_x;
	double y_0 = gsl_vector_get(positions[1].get_position(),1)*(d_-c_) +
	  c_indeces[k]*h_y;
	double x = gsl_vector_get(raw_input, 0);
	double y = gsl_vector_get(raw_input, 1);

	ddPdbxday = ddPdbxday + (std::pow((x-x_0)/sigma_x_,2.0) +
				 std::pow((y-y_0)/sigma_y_,2.0) +
				 -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
	  std::pow(-1, b_power)*
	  std::pow(-1, c_power)*
	  std::pow(-1, a_power);
      }
    }
  }
  dddPdaxdbxday = dddPdaxdbxday / (h_x*h_x*h_y);
  std::cout << "dddPdaxdbxday = " << dddPdaxdbxday << std::endl;

  // std::cout << "G*C^3*dPdax*dPdbx*dPday = "
  // 	    << -2*std::exp(log_before_small_ts[0] + 3*log_CC)*dPdaxs[0]*dPdbx*dPday
  // 	    << std::endl;

  // std::cout << "G*C^3*dPdax*dPdbx*dPday + G*C^2*second order = "
  // 	    << -2*std::exp(log_before_small_t + 3*log_CC)*dPdax*dPdbx*dPday
  //   + 2*std::exp(log_before_small_t + 2*log_CC)*ddPdaxdbx*dPday
  //   + 2*std::exp(log_before_small_t + 2*log_CC)*ddPdaxday*dPdbx
  //   + 2*std::exp(log_before_small_t + 2*log_CC)*ddPdbxday*dPdax
  // 	    << std::endl;

  // std::cout << "G*C^3*dPdax*dPdbx*dPday + 2nd + 3rd order = "
  // 	    << -2*std::exp(log_before_small_t + 3*log_CC)*dPdax*dPdbx*dPday
  //   + 2*std::exp(log_before_small_t + 2*log_CC)*ddPdaxdbx*dPday
  //   + 2*std::exp(log_before_small_t + 2*log_CC)*ddPdaxday*dPdbx
  //   + 2*std::exp(log_before_small_t + 2*log_CC)*ddPdbxday*dPdax
  //   - 2*std::exp(log_before_small_t + log_CC)*dddPdaxdbxday
  // 	    << std::endl;

  set_data(current_a,
	   current_x_0,
	   current_b,
	   current_c,
	   current_y_0,
	   current_d);

  std::cout << "analytic_ax_bx_ay = "
	    << analytic_likelihood_ax_bx_ay(raw_input, 10000)
	    << std::endl;

  double out = 0;
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    out = out +
      -std::exp(log_before_small_ts[ii] + 3*log_CC)*
      dPdaxs[ii]*dPdbxs[ii]*dPdays[ii];
    std::cout << "out[" << ii << "] = "
	      << -std::exp(log_before_small_ts[ii] + 3*log_CC)*
      dPdaxs[ii]*dPdbxs[ii]*dPdays[ii]
	      << std::endl;
  }

  // double ddx0daxdbx = 0;
  // double ddy0daxdbx = 0;
  // double ddPdaxdbx = 0;
  // for (unsigned i=0; i<a_indeces.size(); ++i) {
  //   if (i==0) { a_power=0; } else { a_power=1; };

  //   for (unsigned j=0; j<b_indeces.size(); ++j) {
  //     if (j==0) { b_power=0; } else { b_power=1; };

  //     set_data(current_a + a_indeces[i]*h_x,
  // 	       current_x_0,
  // 	       current_b + b_indeces[j]*h_x,
  // 	       current_c,
  // 	       current_y_0,
  // 	       current_d);
  //     positions = small_t_image_positions_ax_bx();

  //     double x_0 = gsl_vector_get(positions[1].get_position(),0)*(b_-a_) +
  // 	a_indeces[i]*h_x;
  //     double y_0 = gsl_vector_get(positions[1].get_position(),1)*(d_-c_);
  //     double x = gsl_vector_get(raw_input,0);
  //     double y = gsl_vector_get(raw_input,1);

  //     ddx0daxdbx = ddx0daxdbx +
  // 	x_0*
  // 	std::pow(-1, a_power)*
  // 	std::pow(-1, b_power);

  //     ddy0daxdbx = ddy0daxdbx +
  // 	y_0*
  // 	std::pow(-1, a_power)*
  // 	std::pow(-1, b_power);

  //     ddPdaxdbx = ddPdaxdbx +
  // 	(std::pow((x-x_0)/sigma_x_,2.0) +
  // 	 std::pow((y-y_0)/sigma_y_,2.0) +
  // 	 -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
  // 	std::pow(-1, a_power)*
  // 	std::pow(-1, b_power);
  //   }
  // }
  // ddx0daxdbx = ddx0daxdbx / (h_x*h_x);
  // ddy0daxdbx = ddy0daxdbx / (h_x*h_x);
  // ddPdaxdbx = ddPdaxdbx / (h_x*h_x);
  // std::cout << "ddx0daxdbx = " << ddx0daxdbx << std::endl;
  // std::cout << "ddy0daxdbx = " << ddy0daxdbx << std::endl;

  // set_data(current_a,
  // 	   current_x_0,
  // 	   current_b,
  // 	   current_c,
  // 	   current_y_0,
  // 	   current_d);
  // positions = small_t_image_positions_ax_bx();
  // double x_0 = gsl_vector_get(positions[1].get_position(),0)*(b_-a_);
  // double y_0 = gsl_vector_get(positions[1].get_position(),1)*(d_-c_);
  // double x = gsl_vector_get(raw_input, 0);
  // double y = gsl_vector_get(raw_input, 1);

  // double dPdax_analytic = -2.0*(x-x_0)*dx0dax/std::pow(sigma_x_,2) +
  //   -2.0*(y-y_0)*dy0dax/std::pow(sigma_y_,2) -
  //   2*rho_/(sigma_x_*sigma_y_)*-1.0*dx0dax*(y-y_0) -
  //   2*rho_/(sigma_x_*sigma_y_)*-1.0*dy0dax*(x-x_0);
  // std::cout << "dPdax_analytic = " << dPdax_analytic << std::endl;

  // double dPdbx_analytic = -2.0*(x-x_0)*dx0dbx/std::pow(sigma_x_,2) +
  //   -2.0*(y-y_0)*dy0dbx/std::pow(sigma_y_,2) -
  //   2*rho_/(sigma_x_*sigma_y_)*-1.0*dx0dbx*(y-y_0) -
  //   2*rho_/(sigma_x_*sigma_y_)*-1.0*dy0dbx*(x-x_0);
  // std::cout << "dPdbx_analytic = " << dPdbx_analytic << std::endl;

  // double ddPdaxdbx_analytic =
  //   -2.0*(-dx0dbx*dx0dax + (x-x_0)*ddx0daxdbx)/std::pow(sigma_x_,2) +
  //   -2.0*(-dy0dbx*dy0dax + (y-y_0)*ddy0daxdbx)/std::pow(sigma_y_,2) -
  //   2*rho_/(sigma_x_*sigma_y_)*-1.0*(ddx0daxdbx*(y-y_0) + dx0dax*-1.0*dy0dbx) -
  //   2*rho_/(sigma_x_*sigma_y_)*-1.0*(ddy0daxdbx*(x-x_0) + dy0dax*-1.0*dx0dbx);
  // std::cout << "ddPdaxdbx_analytic = " << ddPdaxdbx_analytic << std::endl;

  // std::cout << "dPdax = " << dPdax << "\n";
  // std::cout << "dPdbx = " << dPdbx << "\n";
  // std::cout << "ddPdaxdbx = " << ddPdaxdbx << "\n";
  // std::cout << "G*C^2*dPdax*dPdbx + G*C*ddPdaxdbx = "
  // 	    << std::exp(log_before_small_t + 2*log_CC)*dPdax*dPdbx -
  //   std::exp(log_before_small_t + log_CC)*ddPdaxdbx
  // 	    << std::endl;

  // std::cout << "G*C^2*dPdax_analytic*dPdbx_analytic + G*C*ddPdaxdbx_analytic = "
  // 	    << 2*(exp(log_before_small_t + 2*log_CC)*dPdax_analytic*dPdbx_analytic- exp(log_before_small_t + log_CC)*ddPdaxdbx_analytic)
  // 	    << std::endl;
  // std::cout << "analytic _ax_bx() = "
  // 	    << analytic_likelihood_ax_bx(raw_input, 10)
  // 	    << std::endl;

  return out;
}

double BivariateSolver::
numerical_likelihood_first_order_small_t_ax_bx_ay_by_type_41(const gsl_vector* raw_input,
							     double small_t,
							     double h)
{
  printf("in NUMERICAL_likelihood_first_order_small_t with rho = %f\n",
	 rho_);
  // there are 16 solutions to be computed
  double current_a = a_;
  double current_b = b_;
  double current_c = c_;
  double current_d = d_;
  double current_x_0 = x_0_;
  double current_y_0 = y_0_;

  std::vector<int> a_indeces {1,-1};
  std::vector<int> b_indeces {1,-1};
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
  double CC = -1.0/(2.0*t_*(1-rho_*rho_));

  gsl_matrix* cov_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_set(cov_matrix, 0,0,
  		 sigma_x_*sigma_x_*t_);
  gsl_matrix_set(cov_matrix, 0,1,
  		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,0,
  		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,1,
  		 sigma_y_*sigma_y_*t_);


  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<BivariateImageWithTime> positions =
    small_t_image_positions_type_41(false);
  auto t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "time for positions = "
  // 	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  // 	    << std::endl;

  unsigned counter = 0;

  double dx0dax = 0;
  double dy0dax = 0;
  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdaxs (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    set_data_for_small_t(current_a + a_indeces[i]*h_x,
			 current_x_0,
			 current_b,
			 current_c,
			 current_y_0,
			 current_d);
    positions = small_t_image_positions_type_41(false);

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0_star =
  	gsl_vector_get(positions[ii].get_position(),0)*(b_-a_) +
  	a_indeces[i]*h_x;
      double y_0_star =
  	gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      double polynomial = std::pow((x-x_0_star)/sigma_x_,2) +
  	std::pow((y-y_0_star)/sigma_y_,2) -
  	2*rho_/(sigma_x_*sigma_y_)*(x-x_0_star)*(y-y_0_star);

      dPdaxs[ii] = dPdaxs[ii] + polynomial*
  	std::pow(-1, a_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdaxs[ii] = dPdaxs[ii]/(2*h_x);
    // printf("dPdaxs[%i] = %g\n", ii, dPdaxs[ii]);
  }
  t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "time for dPdaxs = "
  // 	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  // 	    << std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdbxs (positions.size(), 0);
  for (unsigned j=0; j<b_indeces.size(); ++j) {
    if (j==0) { b_power=0; } else { b_power=1; };

    set_data_for_small_t(current_a,
  	     current_x_0,
  	     current_b + b_indeces[j]*h_x,
  	     current_c,
  	     current_y_0,
  	     current_d);
    positions = small_t_image_positions_type_41(false);


    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      dPdbxs[ii] = dPdbxs[ii] +
  	(std::pow((x-x_0)/sigma_x_,2.0) +
  	 std::pow((y-y_0)/sigma_y_,2.0) +
  	 -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
  	std::pow(-1, b_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdbxs[ii] = dPdbxs[ii]/(2*h_x);
    // printf("dPdbxs[%i] = %g\n", ii, dPdbxs[ii]);
  }
  t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "time for dPdbxs = "
  // 	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  // 	    << std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdays (positions.size(), 0.0);
  for (unsigned k=0; k<c_indeces.size(); ++k) {
    if (k==0) { c_power=0; } else { c_power=1; };

    set_data_for_small_t(current_a,
  	     current_x_0,
  	     current_b,
  	     current_c + h_y*c_indeces[k],
  	     current_y_0,
  	     current_d);
    positions = small_t_image_positions_type_41(false);

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 =
  	gsl_vector_get(positions[ii].get_position(),1)*(d_-c_) +
  	h_y*c_indeces[k];

      dPdays[ii] = dPdays[ii] +
  	(std::pow((x-x_0)/sigma_x_,2.0) +
  	 std::pow((y-y_0)/sigma_y_,2.0) +
  	 -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
  	std::pow(-1, c_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdays[ii] = dPdays[ii]/h_y;
    // printf("dPdays[%i] = %g\n", ii, dPdays[ii]);
  }
  t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "time for dPdays = "
  // 	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  // 	    << std::endl;

  std::vector<double> dPdbys (positions.size(), 0);
  for (unsigned l=0; l<d_indeces.size(); ++l) {
    if (l==0) { d_power=0; } else { d_power=1; };

    set_data_for_small_t(current_a,
  	     current_x_0,
  	     current_b,
  	     current_c,
  	     current_y_0,
  	     current_d + d_indeces[l]*h_y);
    positions = small_t_image_positions_type_41(false);

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);

    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      dPdbys[ii] = dPdbys[ii] +  + (std::pow((x-x_0)/sigma_x_,2.0) +
  		     std::pow((y-y_0)/sigma_y_,2.0) +
  		     -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
      std::pow(-1, d_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdbys[ii] = dPdbys[ii]/h_y;
    // printf("dPdbys[%i] = %g\n", ii, dPdbys[ii]);
  }

  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> ddPdaxdbxs (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    for (unsigned j=0; j<b_indeces.size(); ++j) {
      if (j==0) { b_power=0; } else { b_power=1; };

      set_data_for_small_t(current_a + a_indeces[i]*h_x,
  	       current_x_0,
  	       current_b + b_indeces[j]*h_x,
  	       current_c,
  	       current_y_0,
  	       current_d);
      positions = small_t_image_positions_type_41(false);

      double x = gsl_vector_get(raw_input, 0);
      double y = gsl_vector_get(raw_input, 1);

      for (unsigned ii=0; ii<positions.size(); ++ii) {
	double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_) +
	  a_indeces[i]*h_x;
	double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

	ddPdaxdbxs[ii] = ddPdaxdbxs[ii] + (std::pow((x-x_0)/sigma_x_,2.0) +
					   std::pow((y-y_0)/sigma_y_,2.0) +
					   -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
	  std::pow(-1, a_power)*
	  std::pow(-1, b_power);
      }
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    ddPdaxdbxs[ii] = ddPdaxdbxs[ii]/(4*h_x*h_x);
  }
  t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "time for ddPdaxdbxs = "
  // 	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  // 	    << std::endl;

  std::vector<double> ddPdaxdays (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    for (unsigned k=0; k<c_indeces.size(); ++k) {
      if (k==0) { c_power=0; } else { c_power=1; };

      set_data_for_small_t(current_a + a_indeces[i]*h_x,
  	       current_x_0,
  	       current_b,
  	       current_c + b_indeces[k]*h_y,
  	       current_y_0,
  	       current_d);
      positions = small_t_image_positions_type_41(false);

      double x = gsl_vector_get(raw_input, 0);
      double y = gsl_vector_get(raw_input, 1);

      for (unsigned ii=0; ii<positions.size(); ++ii) {
	double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_) +
	  a_indeces[i]*h_x;
	double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_) +
	  c_indeces[k]*h_y;

	ddPdaxdays[ii] = ddPdaxdays[ii] + (std::pow((x-x_0)/sigma_x_,2.0) +
					   std::pow((y-y_0)/sigma_y_,2.0) +
					   -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
	  std::pow(-1, a_power)*
	  std::pow(-1, c_power);
      }
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    ddPdaxdays[ii] = ddPdaxdays[ii]/(2*h_x*h_y);
  }

  std::vector<double> ddPdaxdbys (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    for (unsigned l=0; l<d_indeces.size(); ++l) {
      if (l==0) { d_power=0; } else { d_power=1; };

      set_data_for_small_t(current_a + a_indeces[i]*h_x,
  	       current_x_0,
  	       current_b,
  	       current_c,
  	       current_y_0,
  	       current_d + d_indeces[l]*h_y);
      positions = small_t_image_positions_type_41(false);

      double x = gsl_vector_get(raw_input, 0);
      double y = gsl_vector_get(raw_input, 1);

      for (unsigned ii=0; ii<positions.size(); ++ii) {
	double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_) +
	  a_indeces[i]*h_x;
	double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

	ddPdaxdbys[ii] = ddPdaxdbys[ii] + (std::pow((x-x_0)/sigma_x_,2.0) +
					   std::pow((y-y_0)/sigma_y_,2.0) +
					   -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
	  std::pow(-1, a_power)*
	  std::pow(-1, d_power);
      }
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    ddPdaxdbys[ii] = ddPdaxdbys[ii]/(2*h_x*h_y);
  }

    std::vector<double> ddPdbxdays (positions.size(), 0);
  for (unsigned j=0; j<b_indeces.size(); ++j) {
    if (j==0) { b_power=0; } else { b_power=1; };

    for (unsigned l=0; l<d_indeces.size(); ++l) {
      if (l==0) { d_power=0; } else { d_power=1; };

      set_data_for_small_t(current_a,
  	       current_x_0,
  	       current_b + b_indeces[j]*h_x,
  	       current_c,
  	       current_y_0,
  	       current_d + d_indeces[l]*h_y);
      positions = small_t_image_positions_type_41(false);


      double x = gsl_vector_get(raw_input, 0);
      double y = gsl_vector_get(raw_input, 1);

      for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

	ddPdbxdays[ii] = ddPdbxdays[ii] + (std::pow((x-x_0)/sigma_x_,2.0) +
					   std::pow((y-y_0)/sigma_y_,2.0) +
					   -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
	  std::pow(-1, b_power)*
	  std::pow(-1, d_power);
      }
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    ddPdbxdays[ii] = ddPdbxdays[ii]/(2*h_x*h_y);
  }

  std::vector<double> ddPdbxdbys (positions.size(), 0);
  for (unsigned j=0; j<b_indeces.size(); ++j) {
    if (j==0) { b_power=0; } else { b_power=1; };

    for (unsigned l=0; l<d_indeces.size(); ++l) {
      if (l==0) { d_power=0; } else { d_power=1; };

      set_data_for_small_t(current_a,
  	       current_x_0,
  	       current_b + b_indeces[j]*h_x,
  	       current_c,
  	       current_y_0,
  	       current_d + d_indeces[l]*h_y);
      positions = small_t_image_positions_type_41(false);

      double x_0 = gsl_vector_get(positions[0].get_position(),0)*(b_-a_);
      double y_0 = gsl_vector_get(positions[0].get_position(),1)*(d_-c_);
      double x = gsl_vector_get(raw_input, 0);
      double y = gsl_vector_get(raw_input, 1);

      for (unsigned ii=0; ii<positions.size(); ++ii) {
	ddPdbxdbys[ii] = ddPdbxdbys[ii] + (std::pow((x-x_0)/sigma_x_,2.0) +
					   std::pow((y-y_0)/sigma_y_,2.0) +
					   -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
	  std::pow(-1, b_power)*
	  std::pow(-1, d_power);
      }
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    ddPdbxdbys[ii] = ddPdbxdbys[ii]/(2*h_x*h_y);
  }

  std::vector<double> ddPdaydbys (positions.size(), 0);
  for (unsigned k=0; k<c_indeces.size(); ++k) {
    if (k==0) { c_power=0; } else { c_power=1; };

    for (unsigned l=0; l<d_indeces.size(); ++l) {
      if (l==0) { d_power=0; } else { d_power=1; };

      set_data_for_small_t(current_a,
  	       current_x_0,
  	       current_b,
  	       current_c + h_y*c_indeces[k],
  	       current_y_0,
  	       current_d + h_y*d_indeces[l]);
      positions = small_t_image_positions_type_41(false);

      double x_0 = gsl_vector_get(positions[0].get_position(),0)*(b_-a_);
      double y_0 =
  	gsl_vector_get(positions[0].get_position(),1)*(d_-c_) +
  	h_y*c_indeces[k];
      double x = gsl_vector_get(raw_input, 0);
      double y = gsl_vector_get(raw_input, 1);

      for (unsigned ii=0; ii<positions.size(); ++ii) {
	ddPdaydbys[ii] = ddPdaydbys[ii] + (std::pow((x-x_0)/sigma_x_,2.0) +
					   std::pow((y-y_0)/sigma_y_,2.0) +
					   -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
	  std::pow(-1, c_power)*
	  std::pow(-1, d_power);
      }
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    ddPdaydbys[ii] = ddPdaydbys[ii]/(h_y*h_y);
  }

  // COMPUTING CONTRIBUTIONS OF EACH IMAGE
  t1 = std::chrono::high_resolution_clock::now();
  set_data_for_small_t(current_a,
		       current_x_0,
		       current_b,
		       current_c,
		       current_y_0,
		       current_d);
  t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "time for big set data = "
  // 	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  // 	    << std::endl;

  positions = small_t_image_positions_type_41(false);
  MultivariateNormal mvtnorm = MultivariateNormal();

  std::vector<double> terms = std::vector<double> (positions.size());
  std::vector<double> log_terms = std::vector<double> (positions.size());
  std::vector<int>  terms_signs = std::vector<int> (positions.size());

  for (unsigned ii=0; ii<positions.size(); ++ii) {
    const BivariateImageWithTime& differentiable_image = positions[ii];

    // log( G*C^4*dPdax*dPdbx*dPday*dPdby )
    double log_G = mvtnorm.dmvnorm_log(2,
				       raw_input,
				       differentiable_image.get_position(),
				       cov_matrix);
    int sub_term_1_sign = 1;
    if (std::signbit(differentiable_image.get_mult_factor()*
		     std::exp(log_G + 4*log_CC)*
		     dPdaxs[ii]*
		     dPdbxs[ii]*
		     dPdays[ii]*
		     dPdbys[ii])) {
      sub_term_1_sign = -1;
    }
    double log_sub_term_1 =
      log_G + 4*log_CC +
      log(std::abs(dPdaxs[ii])) +
      log(std::abs(dPdbxs[ii])) +
      log(std::abs(dPdays[ii])) +
      log(std::abs(dPdbys[ii]));

    int sub_term_2_sign = 1;
    if (std::signbit(-differentiable_image.get_mult_factor()*
		     std::exp(log_G + 3*log_CC)*
		     (ddPdaxdbxs[ii]*dPdays[ii]*dPdbys[ii] +
		      ddPdaxdays[ii]*dPdbxs[ii]*dPdbys[ii] +
		      ddPdaxdbys[ii]*dPdbxs[ii]*dPdays[ii] +
		      ddPdbxdays[ii]*dPdaxs[ii]*dPdbys[ii] +
		      ddPdbxdbys[ii]*dPdaxs[ii]*dPdays[ii] +
		      ddPdaydbys[ii]*dPdaxs[ii]*dPdbxs[ii]))) {
      sub_term_2_sign = -1;
    }
    double log_sub_term_2 =
      log_G + 3*log_CC +
      log(std::abs(ddPdaxdbxs[ii]*dPdays[ii]*dPdbys[ii] +
		   ddPdaxdays[ii]*dPdbxs[ii]*dPdbys[ii] +
		   ddPdaxdbys[ii]*dPdbxs[ii]*dPdays[ii] +
		   ddPdbxdays[ii]*dPdaxs[ii]*dPdbys[ii] +
		   ddPdbxdbys[ii]*dPdaxs[ii]*dPdays[ii] +
		   ddPdaydbys[ii]*dPdaxs[ii]*dPdbxs[ii]));

    std::vector<double> log_sub_terms = {log_sub_term_1, log_sub_term_2};
    std::vector<double>::iterator result = std::max_element(log_sub_terms.begin(),
							    log_sub_terms.end());


    terms[ii] =
      std::exp(*result)*(sub_term_1_sign*std::exp((log_sub_term_1-*result)) +
			 sub_term_2_sign*std::exp((log_sub_term_2-*result)));



    if (std::signbit(terms[ii])) {
      terms_signs[ii] = -1;
    } else {
      terms_signs[ii] = 1;
    }

    log_terms[ii] = *result +
      log(std::abs(sub_term_1_sign*std::exp((log_sub_term_1-*result)) +
		   sub_term_2_sign*std::exp((log_sub_term_2-*result))));

  }

  std::vector<double>::iterator result = std::max_element(log_terms.begin(),
							  log_terms.end());

  double out = 0.0;
  double log_out = 0.0;
  for (unsigned ii=0; ii<log_terms.size(); ++ii) {
    out = out +
      std::exp(log_terms[ii]-*result)*terms_signs[ii];

    std::cout << "out before taking log " << ii
	      << " = " << std::exp(log_terms[ii]-*result)*terms_signs[ii]
	      << "; reflection seq = ";

    for (unsigned reflection : positions[ii].get_reflection_sequence()) {
      std::cout << reflection << " ";
    }
    std::cout << "; log_terms[" << ii << "] = " << log_terms[ii];
    std::cout << std::endl;
  }
  std::cout << "out before taking log " << out << std::endl;
  log_out = *result + std::log(out);
  out = out*std::exp(*result);

  out = 0.0;
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    out = out + terms[ii];
  }


  gsl_matrix_free(cov_matrix);
  return (log_out);
}

double BivariateSolver::
likelihood_small_t_type_41_truncated(const gsl_vector* raw_input,
				     double small_t,
				     double h)
{
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
  double CC = -1.0/(2.0*t_*(1-rho_*rho_));

  gsl_matrix* cov_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_set(cov_matrix, 0,0,
  		 sigma_x_*sigma_x_*t_);
  gsl_matrix_set(cov_matrix, 0,1,
  		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,0,
  		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,1,
  		 sigma_y_*sigma_y_*t_);

  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<BivariateImageWithTime> positions =
    small_t_image_positions_type_41(false);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for positions = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  // printf("text(x=-7.5,y=10,\"t=%g, type 4\");\n", positions[0].get_t());
  // for (auto position : positions) {
  //   printf("points(x=%g, y=%g, col=\"red\");\n",
  // 	   gsl_vector_get(position.get_position(),0),
  // 	   gsl_vector_get(position.get_position(),1));
  // }

  unsigned counter = 0;

  double dx0dax = 0;
  double dy0dax = 0;
  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdaxs (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    set_data_for_small_t(current_a + a_indeces[i]*h_x,
			 current_x_0,
			 current_b,
			 current_c,
			 current_y_0,
			 current_d);
    positions = small_t_image_positions_type_41(false);

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0_star =
  	gsl_vector_get(positions[ii].get_position(),0)*(b_-a_) +
  	a_indeces[i]*h_x;
      double y_0_star =
  	gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      double polynomial = std::pow((x-x_0_star)/sigma_x_,2) +
  	std::pow((y-y_0_star)/sigma_y_,2) -
  	2*rho_/(sigma_x_*sigma_y_)*(x-x_0_star)*(y-y_0_star);

      dPdaxs[ii] = dPdaxs[ii] + polynomial*
  	std::pow(-1, a_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdaxs[ii] = dPdaxs[ii]/h_x;
    // printf("dPdaxs[%i] = %g\n", ii, dPdaxs[ii]);
  }
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for dPdaxs = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdbxs (positions.size(), 0);
  for (unsigned j=0; j<b_indeces.size(); ++j) {
    if (j==0) { b_power=0; } else { b_power=1; };

    set_data_for_small_t(current_a,
  	     current_x_0,
  	     current_b + b_indeces[j]*h_x,
  	     current_c,
  	     current_y_0,
  	     current_d);
    positions = small_t_image_positions_type_41(false);


    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      dPdbxs[ii] = dPdbxs[ii] +
  	(std::pow((x-x_0)/sigma_x_,2.0) +
  	 std::pow((y-y_0)/sigma_y_,2.0) +
  	 -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
  	std::pow(-1, b_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdbxs[ii] = dPdbxs[ii]/h_x;
    // printf("dPdbxs[%i] = %g\n", ii, dPdbxs[ii]);
  }
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for dPdbxs = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdays (positions.size(), 0.0);
  for (unsigned k=0; k<c_indeces.size(); ++k) {
    if (k==0) { c_power=0; } else { c_power=1; };

    set_data_for_small_t(current_a,
  	     current_x_0,
  	     current_b,
  	     current_c + h_y*c_indeces[k],
  	     current_y_0,
  	     current_d);
    positions = small_t_image_positions_type_41(false);

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 =
  	gsl_vector_get(positions[ii].get_position(),1)*(d_-c_) +
  	h_y*c_indeces[k];

      dPdays[ii] = dPdays[ii] +
  	(std::pow((x-x_0)/sigma_x_,2.0) +
  	 std::pow((y-y_0)/sigma_y_,2.0) +
  	 -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
  	std::pow(-1, c_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdays[ii] = dPdays[ii]/h_y;
    // printf("dPdays[%i] = %g\n", ii, dPdays[ii]);
  }
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for dPdays = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  std::vector<double> dPdbys (positions.size(), 0);
  for (unsigned l=0; l<d_indeces.size(); ++l) {
    if (l==0) { d_power=0; } else { d_power=1; };

    set_data_for_small_t(current_a,
  	     current_x_0,
  	     current_b,
  	     current_c,
  	     current_y_0,
  	     current_d + d_indeces[l]*h_y);
    positions = small_t_image_positions_type_41(false);

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);

    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      dPdbys[ii] = dPdbys[ii] +  + (std::pow((x-x_0)/sigma_x_,2.0) +
  		     std::pow((y-y_0)/sigma_y_,2.0) +
  		     -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
      std::pow(-1, d_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdbys[ii] = dPdbys[ii]/h_y;
    // printf("dPdbys[%i] = %g\n", ii, dPdbys[ii]);
  }


  // COMPUTING CONTRIBUTIONS OF EACH IMAGE
  t1 = std::chrono::high_resolution_clock::now();
  set_data_for_small_t(current_a,
		       current_x_0,
		       current_b,
		       current_c,
		       current_y_0,
		       current_d);
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for big set data = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  positions = small_t_image_positions_type_41(false);
  double out = 0;
  MultivariateNormal mvtnorm = MultivariateNormal();
  std::vector<int> terms_signs = std::vector<int> (positions.size());
  std::vector<double> log_terms = std::vector<double> (positions.size());

  for (unsigned ii=0; ii<positions.size(); ++ii) {
    const BivariateImageWithTime& differentiable_image = positions[ii];

    // log( G*C^4*dPdax*dPdbx*dPday*dPdby )
    double log_G = mvtnorm.dmvnorm_log(2,
				       raw_input,
				       differentiable_image.get_position(),
				       cov_matrix);

    terms_signs[ii] = 1;
    if (std::signbit(differentiable_image.get_mult_factor()*
		     dPdaxs[ii]*dPdbxs[ii]*dPdays[ii]*dPdbys[ii]))
      {
	terms_signs[ii] = -1;
      }

    log_terms[ii] = log_G + 4*log_CC +
      log(std::abs(dPdaxs[ii])) +
      log(std::abs(dPdbxs[ii])) +
      log(std::abs(dPdays[ii])) +
      log(std::abs(dPdbys[ii]));

    std::cout << "log_term[" << ii << "] = " << log_terms[ii]
	      << " ";
    for (const unsigned& reflection : differentiable_image.get_reflection_sequence()) {
      std::cout << reflection << " ";
    }
    std::cout << " sign = " << terms_signs[ii];
    std::cout << std::endl;

   }

  std::vector<double>::iterator result = std::max_element(log_terms.begin(),
							  log_terms.end());
  out = 0;
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    out = out +
      terms_signs[ii]*std::exp(log_terms[ii]-*result);
  }
  out = std::log(out) + *result;

  gsl_matrix_free(cov_matrix);
  return (out);
}

double BivariateSolver::
likelihood_small_t_type_31_truncated(const gsl_vector* raw_input,
				     double small_t,
				     double h)
{
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
  double CC = -1.0/(2.0*t_*(1-rho_*rho_));

  gsl_matrix* cov_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_set(cov_matrix, 0,0,
  		 sigma_x_*sigma_x_*t_);
  gsl_matrix_set(cov_matrix, 0,1,
  		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,0,
  		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,1,
  		 sigma_y_*sigma_y_*t_);

  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<BivariateImageWithTime> positions =
    small_t_image_positions_type_31(false);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for positions = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  // printf("text(x=-7.5,y=10,\"t=%g, type 4\");\n", positions[0].get_t());
  // for (auto position : positions) {
  //   printf("points(x=%g, y=%g, col=\"red\");\n",
  // 	   gsl_vector_get(position.get_position(),0),
  // 	   gsl_vector_get(position.get_position(),1));
  // }

  unsigned counter = 0;

  double dx0dax = 0;
  double dy0dax = 0;
  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdaxs (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    set_data_for_small_t(current_a + a_indeces[i]*h_x,
			 current_x_0,
			 current_b,
			 current_c,
			 current_y_0,
			 current_d);
    positions = small_t_image_positions_type_31(false);

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0_star =
  	gsl_vector_get(positions[ii].get_position(),0)*(b_-a_) +
  	a_indeces[i]*h_x;
      double y_0_star =
  	gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      double polynomial = std::pow((x-x_0_star)/sigma_x_,2) +
  	std::pow((y-y_0_star)/sigma_y_,2) -
  	2*rho_/(sigma_x_*sigma_y_)*(x-x_0_star)*(y-y_0_star);

      dPdaxs[ii] = dPdaxs[ii] + polynomial*
  	std::pow(-1, a_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdaxs[ii] = dPdaxs[ii]/h_x;
    // printf("dPdaxs[%i] = %g\n", ii, dPdaxs[ii]);
  }
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for dPdaxs = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdbxs (positions.size(), 0);
  for (unsigned j=0; j<b_indeces.size(); ++j) {
    if (j==0) { b_power=0; } else { b_power=1; };

    set_data_for_small_t(current_a,
  	     current_x_0,
  	     current_b + b_indeces[j]*h_x,
  	     current_c,
  	     current_y_0,
  	     current_d);
    positions = small_t_image_positions_type_31(false);


    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      dPdbxs[ii] = dPdbxs[ii] +
  	(std::pow((x-x_0)/sigma_x_,2.0) +
  	 std::pow((y-y_0)/sigma_y_,2.0) +
  	 -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
  	std::pow(-1, b_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdbxs[ii] = dPdbxs[ii]/h_x;
    // printf("dPdbxs[%i] = %g\n", ii, dPdbxs[ii]);
  }
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for dPdbxs = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdays (positions.size(), 0.0);
  for (unsigned k=0; k<c_indeces.size(); ++k) {
    if (k==0) { c_power=0; } else { c_power=1; };

    set_data_for_small_t(current_a,
  	     current_x_0,
  	     current_b,
  	     current_c + h_y*c_indeces[k],
  	     current_y_0,
  	     current_d);
    positions = small_t_image_positions_type_31(false);

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 =
  	gsl_vector_get(positions[ii].get_position(),1)*(d_-c_) +
  	h_y*c_indeces[k];

      dPdays[ii] = dPdays[ii] +
  	(std::pow((x-x_0)/sigma_x_,2.0) +
  	 std::pow((y-y_0)/sigma_y_,2.0) +
  	 -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
  	std::pow(-1, c_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdays[ii] = dPdays[ii]/h_y;
    // printf("dPdays[%i] = %g\n", ii, dPdays[ii]);
  }
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for dPdays = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  std::vector<double> dPdbys (positions.size(), 0);
  for (unsigned l=0; l<d_indeces.size(); ++l) {
    if (l==0) { d_power=0; } else { d_power=1; };

    set_data_for_small_t(current_a,
  	     current_x_0,
  	     current_b,
  	     current_c,
  	     current_y_0,
  	     current_d + d_indeces[l]*h_y);
    positions = small_t_image_positions_type_31(false);

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);

    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      dPdbys[ii] = dPdbys[ii] +  + (std::pow((x-x_0)/sigma_x_,2.0) +
  		     std::pow((y-y_0)/sigma_y_,2.0) +
  		     -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
      std::pow(-1, d_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdbys[ii] = dPdbys[ii]/h_y;
    // printf("dPdbys[%i] = %g\n", ii, dPdbys[ii]);
  }


  // COMPUTING CONTRIBUTIONS OF EACH IMAGE
  t1 = std::chrono::high_resolution_clock::now();
  set_data_for_small_t(current_a,
		       current_x_0,
		       current_b,
		       current_c,
		       current_y_0,
		       current_d);
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for big set data = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  positions = small_t_image_positions_type_31(false);
  double out = 0;
  MultivariateNormal mvtnorm = MultivariateNormal();
  std::vector<int> terms_signs = std::vector<int> (positions.size());
  std::vector<double> log_terms = std::vector<double> (positions.size());

  for (unsigned ii=0; ii<positions.size(); ++ii) {
    const BivariateImageWithTime& differentiable_image = positions[ii];

    // log( G*C^4*dPdax*dPdbx*dPday*dPdby )
    double log_G = mvtnorm.dmvnorm_log(2,
				       raw_input,
				       differentiable_image.get_position(),
				       cov_matrix);

    terms_signs[ii] = 1;
    if (std::signbit(differentiable_image.get_mult_factor()*
		     std::exp(log_G + 4*log_CC)*dPdaxs[ii]*dPdbxs[ii]*dPdays[ii]*dPdbys[ii]))
      {
	terms_signs[ii] = -1;
      }

    log_terms[ii] = log_G + 4*log_CC +
      log(std::abs(dPdaxs[ii])) +
      log(std::abs(dPdbxs[ii])) +
      log(std::abs(dPdays[ii])) +
      log(std::abs(dPdbys[ii]));

    std::cout << "log_term[" << ii << "] = " << log_terms[ii]
	      << " ";
    for (const unsigned& reflection : differentiable_image.get_reflection_sequence()) {
      std::cout << reflection << " ";
    }
    std::cout << std::endl;

   }

  std::vector<double>::iterator result = std::max_element(log_terms.begin(),
							  log_terms.end());
  out = 0;
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    out = out +
      terms_signs[ii]*std::exp(log_terms[ii]-*result);
  }
  out = std::log(std::abs(out)) + *result;

  gsl_matrix_free(cov_matrix);
  return (out);
}

double BivariateSolver::
likelihood_small_t_type_4_truncated(const gsl_vector* raw_input,
				    double small_t,
				    double h)
{
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
  double CC = -1.0/(2.0*t_*(1-rho_*rho_));

  gsl_matrix* cov_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_set(cov_matrix, 0,0,
  		 sigma_x_*sigma_x_*t_);
  gsl_matrix_set(cov_matrix, 0,1,
  		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,0,
  		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,1,
  		 sigma_y_*sigma_y_*t_);

  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<BivariateImageWithTime> positions =
    small_t_image_positions_type_4(false);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for positions = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  // printf("text(x=-7.5,y=10,\"t=%g, type 4\");\n", positions[0].get_t());
  // for (auto position : positions) {
  //   printf("points(x=%g, y=%g, col=\"red\");\n",
  // 	   gsl_vector_get(position.get_position(),0),
  // 	   gsl_vector_get(position.get_position(),1));
  // }

  unsigned counter = 0;

  double dx0dax = 0;
  double dy0dax = 0;
  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdaxs (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    set_data_for_small_t(current_a + a_indeces[i]*h_x,
			 current_x_0,
			 current_b,
			 current_c,
			 current_y_0,
			 current_d);
    positions = small_t_image_positions_type_4(false);

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0_star =
  	gsl_vector_get(positions[ii].get_position(),0)*(b_-a_) +
  	a_indeces[i]*h_x;
      double y_0_star =
  	gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      double polynomial = std::pow((x-x_0_star)/sigma_x_,2) +
  	std::pow((y-y_0_star)/sigma_y_,2) -
  	2*rho_/(sigma_x_*sigma_y_)*(x-x_0_star)*(y-y_0_star);

      dPdaxs[ii] = dPdaxs[ii] + polynomial*
  	std::pow(-1, a_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdaxs[ii] = dPdaxs[ii]/h_x;
    // printf("dPdaxs[%i] = %g\n", ii, dPdaxs[ii]);
  }
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for dPdaxs = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdbxs (positions.size(), 0);
  for (unsigned j=0; j<b_indeces.size(); ++j) {
    if (j==0) { b_power=0; } else { b_power=1; };

    set_data_for_small_t(current_a,
  	     current_x_0,
  	     current_b + b_indeces[j]*h_x,
  	     current_c,
  	     current_y_0,
  	     current_d);
    positions = small_t_image_positions_type_4(false);


    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      dPdbxs[ii] = dPdbxs[ii] +
  	(std::pow((x-x_0)/sigma_x_,2.0) +
  	 std::pow((y-y_0)/sigma_y_,2.0) +
  	 -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
  	std::pow(-1, b_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdbxs[ii] = dPdbxs[ii]/h_x;
    // printf("dPdbxs[%i] = %g\n", ii, dPdbxs[ii]);
  }
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for dPdbxs = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdays (positions.size(), 0.0);
  for (unsigned k=0; k<c_indeces.size(); ++k) {
    if (k==0) { c_power=0; } else { c_power=1; };

    set_data_for_small_t(current_a,
  	     current_x_0,
  	     current_b,
  	     current_c + h_y*c_indeces[k],
  	     current_y_0,
  	     current_d);
    positions = small_t_image_positions_type_4(false);

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 =
  	gsl_vector_get(positions[ii].get_position(),1)*(d_-c_) +
  	h_y*c_indeces[k];

      dPdays[ii] = dPdays[ii] +
  	(std::pow((x-x_0)/sigma_x_,2.0) +
  	 std::pow((y-y_0)/sigma_y_,2.0) +
  	 -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
  	std::pow(-1, c_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdays[ii] = dPdays[ii]/h_y;
    // printf("dPdays[%i] = %g\n", ii, dPdays[ii]);
  }
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for dPdays = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  std::vector<double> dPdbys (positions.size(), 0);
  for (unsigned l=0; l<d_indeces.size(); ++l) {
    if (l==0) { d_power=0; } else { d_power=1; };

    set_data_for_small_t(current_a,
  	     current_x_0,
  	     current_b,
  	     current_c,
  	     current_y_0,
  	     current_d + d_indeces[l]*h_y);
    positions = small_t_image_positions_type_4(false);

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);

    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      dPdbys[ii] = dPdbys[ii] +  + (std::pow((x-x_0)/sigma_x_,2.0) +
  		     std::pow((y-y_0)/sigma_y_,2.0) +
  		     -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
      std::pow(-1, d_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    dPdbys[ii] = dPdbys[ii]/h_y;
    // printf("dPdbys[%i] = %g\n", ii, dPdbys[ii]);
  }


  // COMPUTING CONTRIBUTIONS OF EACH IMAGE
  t1 = std::chrono::high_resolution_clock::now();
  set_data_for_small_t(current_a,
		       current_x_0,
		       current_b,
		       current_c,
		       current_y_0,
		       current_d);
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for big set data = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  positions = small_t_image_positions_type_4(false);
  double out = 0;
  MultivariateNormal mvtnorm = MultivariateNormal();
  std::vector<int> terms_signs = std::vector<int> (positions.size());
  std::vector<double> log_terms = std::vector<double> (positions.size());

  for (unsigned ii=0; ii<positions.size(); ++ii) {
    const BivariateImageWithTime& differentiable_image = positions[ii];

    // log( G*C^4*dPdax*dPdbx*dPday*dPdby )
    double log_G = mvtnorm.dmvnorm_log(2,
				       raw_input,
				       differentiable_image.get_position(),
				       cov_matrix);

    terms_signs[ii] = 1;
    if (std::signbit(differentiable_image.get_mult_factor()*
		     std::exp(log_G + 4*log_CC)*dPdaxs[ii]*dPdbxs[ii]*dPdays[ii]*dPdbys[ii]))
      {
	terms_signs[ii] = -1;
      }

    log_terms[ii] = log_G + 4*log_CC +
      log(std::abs(dPdaxs[ii])) +
      log(std::abs(dPdbxs[ii])) +
      log(std::abs(dPdays[ii])) +
      log(std::abs(dPdbys[ii]));

    std::cout << "log_term[" << ii << "] = " << log_terms[ii] << std::endl;
   }

  std::vector<double>::iterator result = std::max_element(log_terms.begin(),
							  log_terms.end());
  out = 0;
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    out = out +
      terms_signs[ii]*std::exp(log_terms[ii]-*result);
  }
  out = std::log(std::abs(out)) + *result;

  gsl_matrix_free(cov_matrix);
  return (out);
}

double BivariateSolver::
likelihood_small_t_41_truncated_symmetric(const gsl_vector* raw_input,
					  double small_t,
					  double h)
{
  // there are 16 solutions to be computed
  double current_a = a_;
  double current_b = b_;
  double current_c = c_;
  double current_d = d_;
  double current_x_0 = x_0_;
  double current_y_0 = y_0_;

  double log_CC = -1.0*(log(2.0)+log(small_t)+
			2.0*log(sigma_y_) +
			log(1-rho_*rho_));

  gsl_matrix* cov_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_set(cov_matrix, 0,0,
  		 sigma_x_*sigma_x_*small_t);
  gsl_matrix_set(cov_matrix, 0,1,
  		 rho_*sigma_x_*sigma_y_*small_t);
  gsl_matrix_set(cov_matrix, 1,0,
  		 rho_*sigma_x_*sigma_y_*small_t);
  gsl_matrix_set(cov_matrix, 1,1,
  		 sigma_y_*sigma_y_*small_t);

  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<BivariateImageWithTime> positions =
    small_t_image_positions_type_41_symmetric(false);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for positions = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  set_data_for_small_t(current_a,
		       current_x_0,
		       current_b,
		       current_c,
		       current_y_0,
		       current_d);

  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdaxs = dPdax(raw_input, h);
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for dPdaxs = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdbxs = dPdbx(raw_input, h);
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for dPdbxs = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> dPdays = dPday(raw_input, h);
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for dPdays = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  std::vector<double> dPdbys = dPdby(raw_input, h);

  // COMPUTING CONTRIBUTIONS OF EACH IMAGE
  t1 = std::chrono::high_resolution_clock::now();
  set_data_for_small_t(current_a,
		       current_x_0,
		       current_b,
		       current_c,
		       current_y_0,
		       current_d);
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time for big set data = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	    << std::endl;

  positions = small_t_image_positions_type_41_symmetric(false);
  double out = 0;
  MultivariateNormal mvtnorm = MultivariateNormal();
  std::vector<int> terms_signs = std::vector<int> (positions.size());
  std::vector<double> log_terms = std::vector<double> (positions.size());

  for (unsigned ii=0; ii<positions.size(); ++ii) {
    const BivariateImageWithTime& differentiable_image = positions[ii];

    // log( G*C^4*dPdax*dPdbx*dPday*dPdby )
    double log_G = mvtnorm.dmvnorm_log(2,
				       raw_input,
				       differentiable_image.get_position(),
				       cov_matrix);

    terms_signs[ii] = 1;
    if (std::signbit(differentiable_image.get_mult_factor()*
		     dPdaxs[ii]*dPdbxs[ii]*dPdays[ii]*dPdbys[ii]))
      {
	terms_signs[ii] = -1;
      }

    log_terms[ii] = log_G + 4*log_CC +
      log(std::abs(dPdaxs[ii])) +
      log(std::abs(dPdbxs[ii])) +
      log(std::abs(dPdays[ii])) +
      log(std::abs(dPdbys[ii]));

    std::cout << "log_term[" << ii << "] = " << log_terms[ii]
	      << " " << dPdaxs[ii]
	      << " " << dPdbxs[ii]
      	      << " " << dPdays[ii]
	      << " " << dPdbys[ii]
	      << " sign = " << terms_signs[ii]
	      << " position = ("
	      << gsl_vector_get(differentiable_image.get_position(),0)
	      << ","
	      << gsl_vector_get(differentiable_image.get_position(),1)
	      << ") ";
    for (unsigned ref : differentiable_image.get_reflection_sequence()) {
       std::cout << ref << " ";
    }
    std::cout << std::endl;
  }

  std::vector<double>::iterator result = std::max_element(log_terms.begin(),
							  log_terms.end());
  out = 0;
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    out = out +
      terms_signs[ii]*std::exp(log_terms[ii]-*result);
  }
  out = std::log(out) + *result;

  gsl_matrix_free(cov_matrix);
  return (out);
}

std::vector<double> BivariateSolver::dPdax(const gsl_vector* raw_input,
					   double h)
{
  // TODO(gdinolov): add sigma_x sigma_y before rescaling so that when
  // differentiating we only do so wrt to x_0, y_0 and not the
  // diffusion parameters

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

  double h_x = h*(b_ - a_);

  std::vector<BivariateImageWithTime> positions =
    small_t_image_positions_type_41_symmetric(false);

  std::vector<double> out (positions.size(), 0);
  for (unsigned i=0; i<a_indeces.size(); ++i) {
    if (i==0) { a_power=0; } else { a_power=1; };

    set_data_for_small_t(current_a + a_indeces[i]*h_x,
			 current_x_0,
			 current_b,
			 current_c,
			 current_y_0,
			 current_d);
    positions = small_t_image_positions_type_41_symmetric(false);

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0_star =
  	gsl_vector_get(positions[ii].get_position(),0)*(b_-a_) +
  	a_indeces[i]*h_x;
      double y_0_star =
  	gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      double polynomial = std::pow((x-x_0_star)/sigma_x_,2) +
  	std::pow((y-y_0_star)/sigma_y_,2) -
  	2*rho_/(sigma_x_*sigma_y_)*(x-x_0_star)*(y-y_0_star);

      out[ii] = out[ii] + polynomial*
  	std::pow(-1, a_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    out[ii] = out[ii]/h_x;
  }

  set_data_for_small_t(current_a,
		       current_x_0,
		       current_b,
		       current_c,
		       current_y_0,
		       current_d);

  return (out);
}

std::vector<double> BivariateSolver::dPdbx(const gsl_vector* raw_input,
					   double h)
{
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

  int b_power=1;

  double h_x = h*(b_ - a_);

  std::vector<BivariateImageWithTime> positions =
    small_t_image_positions_type_41_symmetric(false);

  std::vector<double> out (positions.size(), 0);
  for (unsigned j=0; j<b_indeces.size(); ++j) {
    if (j==0) { b_power=0; } else { b_power=1; };

    set_data_for_small_t(current_a,
			 current_x_0,
			 current_b + b_indeces[j]*h_x,
			 current_c,
			 current_y_0,
			 current_d);
    positions = small_t_image_positions_type_41_symmetric(false);


    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      out[ii] = out[ii] +
  	(std::pow((x-x_0)/sigma_x_,2.0) +
  	 std::pow((y-y_0)/sigma_y_,2.0) +
  	 -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
  	std::pow(-1, b_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    out[ii] = out[ii]/h_x;
  }

  set_data_for_small_t(current_a,
		       current_x_0,
		       current_b,
		       current_c,
		       current_y_0,
		       current_d);

  return (out);
}

std::vector<double> BivariateSolver::dPday(const gsl_vector* raw_input,
					   double h)
{
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

  int c_power=1;

  double h_y = h*(d_ - c_);

  std::vector<BivariateImageWithTime> positions =
    small_t_image_positions_type_41_symmetric(false);

  std::vector<double> out (positions.size(), 0.0);
  for (unsigned k=0; k<c_indeces.size(); ++k) {
    if (k==0) { c_power=0; } else { c_power=1; };

    set_data_for_small_t(current_a,
  	     current_x_0,
  	     current_b,
  	     current_c + h_y*c_indeces[k],
  	     current_y_0,
  	     current_d);
    positions = small_t_image_positions_type_41_symmetric(false);

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);
    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 =
  	gsl_vector_get(positions[ii].get_position(),1)*(d_-c_) +
  	h_y*c_indeces[k];

      out[ii] = out[ii] +
  	(std::pow((x-x_0)/sigma_x_,2.0) +
  	 std::pow((y-y_0)/sigma_y_,2.0) +
  	 -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
  	std::pow(-1, c_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    out[ii] = out[ii]/h_y;
  }

  set_data_for_small_t(current_a,
		       current_x_0,
		       current_b,
		       current_c,
		       current_y_0,
		       current_d);

  return (out);
}

std::vector<double> BivariateSolver::dPdby(const gsl_vector* raw_input,
					   double h)
{
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

  int d_power=1;

  double h_y = h*(d_ - c_);

  std::vector<BivariateImageWithTime> positions =
    small_t_image_positions_type_41_symmetric(false);

  std::vector<double> out (positions.size(), 0.0);
  for (unsigned l=0; l<d_indeces.size(); ++l) {
    if (l==0) { d_power=0; } else { d_power=1; };

    set_data_for_small_t(current_a,
			 current_x_0,
			 current_b,
			 current_c,
			 current_y_0,
			 current_d + d_indeces[l]*h_y);
    positions = small_t_image_positions_type_41_symmetric(false);

    double x = gsl_vector_get(raw_input, 0);
    double y = gsl_vector_get(raw_input, 1);

    for (unsigned ii=0; ii<positions.size(); ++ii) {
      double x_0 = gsl_vector_get(positions[ii].get_position(),0)*(b_-a_);
      double y_0 = gsl_vector_get(positions[ii].get_position(),1)*(d_-c_);

      out[ii] = out[ii] +  + (std::pow((x-x_0)/sigma_x_,2.0) +
				    std::pow((y-y_0)/sigma_y_,2.0) +
				    -2.0*rho_*(x-x_0)*(y-y_0)/(sigma_y_*sigma_x_))*
	std::pow(-1, d_power);
    }
  }
  for (unsigned ii=0; ii<positions.size(); ++ii) {
    out[ii] = out[ii]/h_y;
  }

  set_data_for_small_t(current_a,
		       current_x_0,
		       current_b,
		       current_c,
		       current_y_0,
		       current_d);

  return (out);
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
		   basis_->project_solver(*small_t_solution_,
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

  // evec %*% diag(exp(eval)*(t-t_small))
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
    images_array[i+4] = get_y_0_2();
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

std::vector<BivariateImageWithTime> BivariateSolver::
small_t_image_positions_ax_bx_ay() const
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

  double images_array [16];
  for (unsigned i=0; i<8; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+8] = get_y_0_2();
  }
  double images_transformed_array [16];

  gsl_matrix_view images_view = gsl_matrix_view_array(images_array, 2, 8);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, 8);

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

  std::vector<gsl_vector_view> images_vector (8);
  for (unsigned i=0; i<8; ++i) {
    images_vector[i] =
      gsl_matrix_column(&images_transformed_view.matrix,i);
  }

  std::vector<std::vector<double>> distance_from_image_to_line (8, std::vector<double> (4));
  std::vector<double> max_admissible_times (6);
  std::vector<BivariateImageWithTime> final_images (6);
  std::vector<double> signs_vector = std::vector<double> (8,1.0);

  unsigned image_counter = 0;
  std::vector<unsigned> p_indeces {3,1,0};
  for (unsigned p : p_indeces) {
    std::vector<unsigned> o_indeces {3,1,0};
    std::vector<unsigned>::iterator it;
    it = std::find(o_indeces.begin(), o_indeces.end(), p);
    o_indeces.erase(it);

    for (unsigned o : o_indeces) {
      std::vector<unsigned> n_indeces {3,1,0};

      std::vector<unsigned>::iterator it;
      it = std::find(n_indeces.begin(), n_indeces.end(), p);
      n_indeces.erase(it);
      it = std::find(n_indeces.begin(), n_indeces.end(), o);
      n_indeces.erase(it);

      for (unsigned n : n_indeces) {

	// C = alpha*op(A)*op(B) + beta*C
	gsl_blas_dgemm(CblasNoTrans, //op(A) = A
		       CblasNoTrans, //op(B) = B
		       1.0, //alpha=1
		       small_t_solution_->get_rotation_matrix(), //A
		       &images_view.matrix, //B
		       0.0, //beta=0
		       &images_transformed_view.matrix); //C

	signs_vector = std::vector<double> (8,1.0);
	unsigned counter = 0;
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
	}

	int sign = 1;
	for (unsigned i=1; i<8; ++i) {
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

	      for (unsigned i=0; i<8; ++i) { // iterating over images
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
		       &images_vector[7].vector, //x
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
  std::vector<std::vector<unsigned>> reflection_sequence_per_final_image (24, std::vector<unsigned> (4));
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
	  reflection_sequence_per_final_image[image_counter] =
	    std::vector<unsigned> {p,o,n,m};

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

  	  while (mmax > 1e-12) {
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

  	  final_images[image_counter] =
	    BivariateImageWithTime(current_image,
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
  for (unsigned i=0; i<final_images.size(); ++i) {
    const BivariateImageWithTime& current_image = final_images[i];
    if (std::abs( current_image.get_t() - biggest_time) <= std::numeric_limits<double>::epsilon()) {
      max_t_images.push_back( current_image );
      std::cout << "reflections: "
		<< reflection_sequence_per_final_image[i][0]
		<< ","
		<< reflection_sequence_per_final_image[i][1]
		<< ","
		<< reflection_sequence_per_final_image[i][2]
		<< ","
		<< reflection_sequence_per_final_image[i][3]
		<< std::endl;
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

// returns 32 images, 4 of which are differentiable with respect to
// all four boundaries. Type 1 refers to flipping around borders 1,2 first
std::vector<BivariateImageWithTime> BivariateSolver::
small_t_image_positions_type_1(bool PRINT) const
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

  unsigned number_images = 64;
  double images_array [number_images*2];
  for (unsigned i=0; i<number_images; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+number_images] = get_y_0_2();
  }
  double images_transformed_array [number_images*2];

  gsl_matrix_view images_view =
    gsl_matrix_view_array(images_array, 2, number_images);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, number_images);

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

  std::vector<gsl_vector_view> images_vector (number_images);
  std::vector<BivariateImageWithTime> all_images (number_images);

  for (unsigned i=0; i<number_images; ++i) {
    images_vector[i] =
      gsl_matrix_column(&images_transformed_view.matrix,i);
  }

  std::vector<std::vector<double>> distance_from_image_to_line (number_images, std::vector<double> (4));
  std::vector<double> max_admissible_times (1); // ONLY ONE SET OF IMAGES FOR NOW
  std::vector<BivariateImageWithTime> final_images (number_images);
  std::vector<std::vector<unsigned>> reflection_sequence_per_final_image (number_images, std::vector<unsigned> (0));
  std::vector<double> signs_vector = std::vector<double> (number_images,1.0);

  std::vector<BivariateImageWithTime> unique_images (0);
  unsigned image_counter = 0; // iterating over the number_images

  std::vector<unsigned> p_indeces {2};

  for (unsigned p : p_indeces) {
    std::vector<unsigned> o_indeces {0};

    for (unsigned o : o_indeces) {
      std::vector<unsigned> n_indeces {2};

      for (unsigned n : n_indeces) {
  	std::vector<unsigned> m_indeces {1};

  	for (unsigned m : m_indeces) {
	  std::vector<unsigned> q_indeces {3};

	  for (unsigned q : q_indeces) {
	    std::vector<unsigned> r_indeces {1};

	    for (unsigned r : r_indeces) {
	      // std::cout << "## (p=" << p
	      // 		<< "o=" << o
	      // 		<< ",n=" << n
	      // 		<< ",m=" << m
	      // 		<< ",q=" << q
	      // 		<< ",r=" << r
	      // 		<< ")" << std::endl;


	      gsl_blas_dgemm(CblasNoTrans, //op(A) = A
			     CblasNoTrans, //op(B) = B
			     1.0, //alpha=1
			     small_t_solution_->get_rotation_matrix(), //A
			     &images_view.matrix, //B
			     0.0, //beta=0
			     &images_transformed_view.matrix); //C

	      signs_vector = std::vector<double> (number_images,1.0);
	      unsigned counter = 0;
	      for (unsigned l=0; l<2; ++l) {
		for (unsigned k=0; k<2; ++k) {
		  for (unsigned j=0; j<2; ++j) {
		    for (unsigned i=0; i<2; ++i) {
		      for (unsigned h=0; h<2; ++h) {
			for (unsigned g=0; g<2; ++g) {

			  reflection_sequence_per_final_image[counter] =
			    std::vector<unsigned> (0);

			  gsl_vector* current_image = &images_vector[counter].vector;
			  if (i==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[p][0],
								 lines[p][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(p);
			  }
			  if (j==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[o][0],
								 lines[o][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(o);
			  }
			  if (k==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[n][0],
								 lines[n][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(n);
			  }
			  if (l==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[m][0],
								 lines[m][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(m);
			  }
			  if (h==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[q][0],
								 lines[q][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(q);
			  }
			  if (g==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[r][0],
								 lines[r][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(r);
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

			  distance_from_image_to_line[counter][0] = d1;
			  distance_from_image_to_line[counter][1] = d2;
			  distance_from_image_to_line[counter][2] = d3;
			  distance_from_image_to_line[counter][3] = d4;

			  all_images[counter] = BivariateImageWithTime(current_image,
								       1.0,
								       signs_vector[counter],
								       reflection_sequence_per_final_image[counter]);



			  counter = counter + 1;
			}
		      }
		    }
		  }
		}
	      }

	      // GENERATING UNIQUE IMAGES FROM THE SET
	      unsigned unique_image_counter = 0;
	      unique_images = std::vector<BivariateImageWithTime> (0);
	      for (const BivariateImageWithTime& current_image : all_images) {
		if (unique_images.empty()) {
		  unique_images.push_back(current_image);
		} else {
		  unsigned n=0;
		  std::vector<bool> distances_to_current_image (unique_images.size());

		  std::generate(distances_to_current_image.begin(),
				distances_to_current_image.end(),
				[&current_image, &n, &unique_images] () {
				  double distance =
				    std::sqrt(
					      std::pow(gsl_vector_get(unique_images[n].get_position(),0)-
						       gsl_vector_get(current_image.get_position(),0),
						       2) +
					      std::pow(gsl_vector_get(unique_images[n].get_position(),1)-
						       gsl_vector_get(current_image.get_position(),1),
						       2));

				  bool out = false;
				  if ((distance <= 1e-13) && //10*std::numeric_limits<double>::epsilon()) &&
				      (current_image.get_mult_factor() == unique_images[n].get_mult_factor())) {
				    out = true;
				  }

				  n++;
				  return out;
				});

		  std::vector<bool>::iterator it = std::find(distances_to_current_image.begin(), distances_to_current_image.end(), true);
		  if (it == std::end(distances_to_current_image)) {
		    unique_images.push_back(current_image);
		  }
		}
	      }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[0][0], 0),
		       gsl_vector_get(lines[0][1], 0),
		       gsl_vector_get(lines[0][0], 1),
		       gsl_vector_get(lines[0][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[1][0], 0),
		       gsl_vector_get(lines[1][1], 0),
		       gsl_vector_get(lines[1][0], 1),
		       gsl_vector_get(lines[1][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[2][0], 0),
		       gsl_vector_get(lines[2][1], 0),
		       gsl_vector_get(lines[2][0], 1),
		       gsl_vector_get(lines[2][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); \n",
		       gsl_vector_get(lines[3][0], 0),
		       gsl_vector_get(lines[3][1], 0),
		       gsl_vector_get(lines[3][0], 1),
		       gsl_vector_get(lines[3][1], 1));
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 2) &&
		       (reflection_sequence_per_final_image_current[1] == 0) &&
		       (reflection_sequence_per_final_image_current[2] == 1) &&
		       (reflection_sequence_per_final_image_current[3] == 3)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 2) &&
		       (reflection_sequence_per_final_image_current[1] == 0) &&
		       (reflection_sequence_per_final_image_current[2] == 3) &&
		       (reflection_sequence_per_final_image_current[3] == 1)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 0) &&
		       (reflection_sequence_per_final_image_current[1] == 2) &&
		       (reflection_sequence_per_final_image_current[2] == 1) &&
		       (reflection_sequence_per_final_image_current[3] == 3)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 0) &&
		       (reflection_sequence_per_final_image_current[1] == 2) &&
		       (reflection_sequence_per_final_image_current[2] == 3) &&
		       (reflection_sequence_per_final_image_current[3] == 1)) |
		      (reflection_sequence_per_final_image_current.size() == 0))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }  else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");

		  unique_image_counter++;
		}
	      }


	      // CHECKING IF ANY OF THE UNIQUE IMAGES ARE WITHIN THE DOMAIN
	      int sign = 1;
	      for (unsigned i=1; i<unique_images.size(); ++i) {
		double d1 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[0][0],
						  lines[0][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d2 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[1][0],
						  lines[1][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d3 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[2][0],
						  lines[2][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d4 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[3][0],
						  lines[3][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());

		std::vector<double> d_from_im_to_l {d1, d2, d3, d4};

		std::vector<double>::iterator result = std::min_element(d_from_im_to_l.begin(),
									d_from_im_to_l.end());
		if (!std::signbit(*result)) {
		  sign = -1;
		  break;
		}
	      }

	      // calculating max admissible time
	      double mmax = 1.0;
	      double max_admissible_t = 1.0; //(1.0/0.9) * std::pow(distance_from_image_to_line[0][1]/6.0, 2);

	      while (mmax > 1e-12) {
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

		    for (unsigned i=0; i<unique_images.size(); ++i) { // iterating over unique images
		      const gsl_vector* current_image = unique_images[i].get_position();
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

	      for (unsigned current_image_counter=0;
		   current_image_counter<unique_images.size();
		   ++current_image_counter) {
		gsl_vector* current_image = gsl_vector_alloc(2);
		// C = alpha*op(A)*x + beta*C
		gsl_blas_dgemv(CblasNoTrans, //op(A) = A
			       1.0, //alpha=1
			       &Rotation_matrix_inv_view.matrix, //A
			       unique_images[current_image_counter].get_position(), //x
			       0.0, //beta=0
			       current_image); //C

		unique_images[current_image_counter] =
		  BivariateImageWithTime(current_image,
					 sign*max_admissible_t,
					 unique_images[current_image_counter].get_mult_factor(),
					 unique_images[current_image_counter].get_reflection_sequence());
		gsl_vector_free(current_image);
	      }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(0,1), y=c(0,0)); ");
		printf("lines(x=c(1,1), y=c(0,1)); ");
		printf("lines(x=c(1,0), y=c(1,1)); ");
		printf("lines(x=c(0,0), y=c(1,0)); \n");
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 2) &&
		       (reflection_sequence_per_final_image_current[1] == 0) &&
		       (reflection_sequence_per_final_image_current[2] == 1) &&
		       (reflection_sequence_per_final_image_current[3] == 3)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 2) &&
		       (reflection_sequence_per_final_image_current[1] == 0) &&
		       (reflection_sequence_per_final_image_current[2] == 3) &&
		       (reflection_sequence_per_final_image_current[3] == 1)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 0) &&
		       (reflection_sequence_per_final_image_current[1] == 2) &&
		       (reflection_sequence_per_final_image_current[2] == 1) &&
		       (reflection_sequence_per_final_image_current[3] == 3)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 0) &&
		       (reflection_sequence_per_final_image_current[1] == 2) &&
		       (reflection_sequence_per_final_image_current[2] == 3) &&
		       (reflection_sequence_per_final_image_current[3] == 1)) |
		      (reflection_sequence_per_final_image_current.size() == 0))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }  else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");
		}
	      }

	      image_counter = image_counter + 1;
	    }
	  }
	}
      }
    }
  }

  return unique_images;
}

std::vector<BivariateImageWithTime> BivariateSolver::
small_t_image_positions_type_2(bool PRINT) const
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

  unsigned number_images = 64;
  double images_array [number_images*2];
  for (unsigned i=0; i<number_images; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+number_images] = get_y_0_2();
  }
  double images_transformed_array [number_images*2];

  gsl_matrix_view images_view =
    gsl_matrix_view_array(images_array, 2, number_images);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, number_images);

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

  std::vector<gsl_vector_view> images_vector (number_images);
  std::vector<BivariateImageWithTime> all_images (number_images);

  for (unsigned i=0; i<number_images; ++i) {
    images_vector[i] =
      gsl_matrix_column(&images_transformed_view.matrix,i);
  }

  std::vector<std::vector<double>> distance_from_image_to_line (number_images, std::vector<double> (4));
  std::vector<double> max_admissible_times (1); // ONLY ONE SET OF IMAGES FOR NOW
  std::vector<BivariateImageWithTime> final_images (number_images);
  std::vector<std::vector<unsigned>> reflection_sequence_per_final_image (number_images, std::vector<unsigned> (0));
  std::vector<double> signs_vector = std::vector<double> (number_images,1.0);

  std::vector<BivariateImageWithTime> unique_images (0);
  unsigned image_counter = 0; // iterating over the number_images

  std::vector<unsigned> p_indeces {2};

  for (unsigned p : p_indeces) {
    std::vector<unsigned> o_indeces {0};

    for (unsigned o : o_indeces) {
      std::vector<unsigned> n_indeces {2};

      for (unsigned n : n_indeces) {
  	std::vector<unsigned> m_indeces {3};

  	for (unsigned m : m_indeces) {
	  std::vector<unsigned> q_indeces {1};

	  for (unsigned q : q_indeces) {
	    std::vector<unsigned> r_indeces {3};

	    for (unsigned r : r_indeces) {
	      // std::cout << "## (p=" << p
	      // 		<< "o=" << o
	      // 		<< ",n=" << n
	      // 		<< ",m=" << m
	      // 		<< ",q=" << q
	      // 		<< ",r=" << r
	      // 		<< ")" << std::endl;


	      gsl_blas_dgemm(CblasNoTrans, //op(A) = A
			     CblasNoTrans, //op(B) = B
			     1.0, //alpha=1
			     small_t_solution_->get_rotation_matrix(), //A
			     &images_view.matrix, //B
			     0.0, //beta=0
			     &images_transformed_view.matrix); //C

	      signs_vector = std::vector<double> (number_images,1.0);
	      unsigned counter = 0;
	      for (unsigned l=0; l<2; ++l) {
		for (unsigned k=0; k<2; ++k) {
		  for (unsigned j=0; j<2; ++j) {
		    for (unsigned i=0; i<2; ++i) {
		      for (unsigned h=0; h<2; ++h) {
			for (unsigned g=0; g<2; ++g) {

			  reflection_sequence_per_final_image[counter] =
			    std::vector<unsigned> (0);

			  gsl_vector* current_image = &images_vector[counter].vector;
			  if (i==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[p][0],
								 lines[p][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(p);
			  }
			  if (j==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[o][0],
								 lines[o][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(o);
			  }
			  if (k==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[n][0],
								 lines[n][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(n);
			  }
			  if (l==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[m][0],
								 lines[m][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(m);
			  }
			  if (h==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[q][0],
								 lines[q][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(q);
			  }
			  if (g==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[r][0],
								 lines[r][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(r);
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

			  distance_from_image_to_line[counter][0] = d1;
			  distance_from_image_to_line[counter][1] = d2;
			  distance_from_image_to_line[counter][2] = d3;
			  distance_from_image_to_line[counter][3] = d4;

			  all_images[counter] = BivariateImageWithTime(current_image,
								       1.0,
								       signs_vector[counter],
								       reflection_sequence_per_final_image[counter]);



			  counter = counter + 1;
			}
		      }
		    }
		  }
		}
	      }

	      // GENERATING UNIQUE IMAGES FROM THE SET
	      unsigned unique_image_counter = 0;
	      unique_images = std::vector<BivariateImageWithTime> (0);
	      for (const BivariateImageWithTime& current_image : all_images) {
		if (unique_images.empty()) {
		  unique_images.push_back(current_image);
		} else {
		  unsigned n=0;
		  std::vector<bool> distances_to_current_image (unique_images.size());

		  std::generate(distances_to_current_image.begin(),
				distances_to_current_image.end(),
				[&current_image, &n, &unique_images] () {
				  double distance =
				    std::sqrt(
					      std::pow(gsl_vector_get(unique_images[n].get_position(),0)-
						       gsl_vector_get(current_image.get_position(),0),
						       2) +
					      std::pow(gsl_vector_get(unique_images[n].get_position(),1)-
						       gsl_vector_get(current_image.get_position(),1),
						       2));

				  bool out = false;
				  if ((distance <= 1e-13) && //10*std::numeric_limits<double>::epsilon()) &&
				      (current_image.get_mult_factor() == unique_images[n].get_mult_factor())) {
				    out = true;
				  }

				  n++;
				  return out;
				});

		  std::vector<bool>::iterator it = std::find(distances_to_current_image.begin(), distances_to_current_image.end(), true);
		  if (it == std::end(distances_to_current_image)) {
		    unique_images.push_back(current_image);
		  }
		}
	      }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[0][0], 0),
		       gsl_vector_get(lines[0][1], 0),
		       gsl_vector_get(lines[0][0], 1),
		       gsl_vector_get(lines[0][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[1][0], 0),
		       gsl_vector_get(lines[1][1], 0),
		       gsl_vector_get(lines[1][0], 1),
		       gsl_vector_get(lines[1][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[2][0], 0),
		       gsl_vector_get(lines[2][1], 0),
		       gsl_vector_get(lines[2][0], 1),
		       gsl_vector_get(lines[2][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); \n",
		       gsl_vector_get(lines[3][0], 0),
		       gsl_vector_get(lines[3][1], 0),
		       gsl_vector_get(lines[3][0], 1),
		       gsl_vector_get(lines[3][1], 1));
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 2) &&
		       (reflection_sequence_per_final_image_current[1] == 0) &&
		       (reflection_sequence_per_final_image_current[2] == 1) &&
		       (reflection_sequence_per_final_image_current[3] == 3)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 2) &&
		       (reflection_sequence_per_final_image_current[1] == 0) &&
		       (reflection_sequence_per_final_image_current[2] == 3) &&
		       (reflection_sequence_per_final_image_current[3] == 1)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 0) &&
		       (reflection_sequence_per_final_image_current[1] == 2) &&
		       (reflection_sequence_per_final_image_current[2] == 1) &&
		       (reflection_sequence_per_final_image_current[3] == 3)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 0) &&
		       (reflection_sequence_per_final_image_current[1] == 2) &&
		       (reflection_sequence_per_final_image_current[2] == 3) &&
		       (reflection_sequence_per_final_image_current[3] == 1)) |
		      (reflection_sequence_per_final_image_current.size() == 0))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }  else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");

		  unique_image_counter++;
		}
	      }


	      // CHECKING IF ANY OF THE UNIQUE IMAGES ARE WITHIN THE DOMAIN
	      int sign = 1;
	      for (unsigned i=1; i<unique_images.size(); ++i) {
		double d1 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[0][0],
						  lines[0][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d2 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[1][0],
						  lines[1][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d3 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[2][0],
						  lines[2][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d4 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[3][0],
						  lines[3][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());

		std::vector<double> d_from_im_to_l {d1, d2, d3, d4};

		std::vector<double>::iterator result = std::min_element(d_from_im_to_l.begin(),
									d_from_im_to_l.end());
		if (!std::signbit(*result)) {
		  sign = -1;
		  break;
		}
	      }

	      // calculating max admissible time
	      double mmax = 1.0;
	      double max_admissible_t = 1.0; //(1.0/0.9) * std::pow(distance_from_image_to_line[0][1]/6.0, 2);

	      while (mmax > 1e-12) {
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

		    for (unsigned i=0; i<unique_images.size(); ++i) { // iterating over unique images
		      const gsl_vector* current_image = unique_images[i].get_position();
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

	      for (unsigned current_image_counter=0;
		   current_image_counter<unique_images.size();
		   ++current_image_counter) {
		gsl_vector* current_image = gsl_vector_alloc(2);
		// C = alpha*op(A)*x + beta*C
		gsl_blas_dgemv(CblasNoTrans, //op(A) = A
			       1.0, //alpha=1
			       &Rotation_matrix_inv_view.matrix, //A
			       unique_images[current_image_counter].get_position(), //x
			       0.0, //beta=0
			       current_image); //C

		unique_images[current_image_counter] =
		  BivariateImageWithTime(current_image,
					 sign*max_admissible_t,
					 unique_images[current_image_counter].get_mult_factor(),
					 unique_images[current_image_counter].get_reflection_sequence());
		gsl_vector_free(current_image);
	      }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(0,1), y=c(0,0)); ");
		printf("lines(x=c(1,1), y=c(0,1)); ");
		printf("lines(x=c(1,0), y=c(1,1)); ");
		printf("lines(x=c(0,0), y=c(1,0)); \n");
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 2) &&
		       (reflection_sequence_per_final_image_current[1] == 0) &&
		       (reflection_sequence_per_final_image_current[2] == 1) &&
		       (reflection_sequence_per_final_image_current[3] == 3)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 2) &&
		       (reflection_sequence_per_final_image_current[1] == 0) &&
		       (reflection_sequence_per_final_image_current[2] == 3) &&
		       (reflection_sequence_per_final_image_current[3] == 1)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 0) &&
		       (reflection_sequence_per_final_image_current[1] == 2) &&
		       (reflection_sequence_per_final_image_current[2] == 1) &&
		       (reflection_sequence_per_final_image_current[3] == 3)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 0) &&
		       (reflection_sequence_per_final_image_current[1] == 2) &&
		       (reflection_sequence_per_final_image_current[2] == 3) &&
		       (reflection_sequence_per_final_image_current[3] == 1)) |
		      (reflection_sequence_per_final_image_current.size() == 0))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }  else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");
		}
	      }

	      image_counter = image_counter + 1;
	    }
	  }
	}
      }
    }
  }

  return unique_images;
}

std::vector<BivariateImageWithTime> BivariateSolver::
small_t_image_positions_type_3(bool PRINT) const
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

  unsigned number_images = 64;
  double images_array [number_images*2];
  for (unsigned i=0; i<number_images; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+number_images] = get_y_0_2();
  }
  double images_transformed_array [number_images*2];

  gsl_matrix_view images_view =
    gsl_matrix_view_array(images_array, 2, number_images);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, number_images);

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

  std::vector<gsl_vector_view> images_vector (number_images);
  std::vector<BivariateImageWithTime> all_images (number_images);

  for (unsigned i=0; i<number_images; ++i) {
    images_vector[i] =
      gsl_matrix_column(&images_transformed_view.matrix,i);
  }

  std::vector<std::vector<double>> distance_from_image_to_line (number_images, std::vector<double> (4));
  std::vector<double> max_admissible_times (1); // ONLY ONE SET OF IMAGES FOR NOW
  std::vector<BivariateImageWithTime> final_images (number_images);
  std::vector<std::vector<unsigned>> reflection_sequence_per_final_image (number_images, std::vector<unsigned> (0));
  std::vector<double> signs_vector = std::vector<double> (number_images,1.0);

  std::vector<BivariateImageWithTime> unique_images (0);
  unsigned image_counter = 0; // iterating over the number_images

  std::vector<unsigned> p_indeces {1};

  for (unsigned p : p_indeces) {
    std::vector<unsigned> o_indeces {3};

    for (unsigned o : o_indeces) {
      std::vector<unsigned> n_indeces {1};

      for (unsigned n : n_indeces) {
  	std::vector<unsigned> m_indeces {0};

  	for (unsigned m : m_indeces) {
	  std::vector<unsigned> q_indeces {2};

	  for (unsigned q : q_indeces) {
	    std::vector<unsigned> r_indeces {0};

	    for (unsigned r : r_indeces) {
	      // std::cout << "## (p=" << p
	      // 		<< "o=" << o
	      // 		<< ",n=" << n
	      // 		<< ",m=" << m
	      // 		<< ",q=" << q
	      // 		<< ",r=" << r
	      // 		<< ")" << std::endl;


	      gsl_blas_dgemm(CblasNoTrans, //op(A) = A
			     CblasNoTrans, //op(B) = B
			     1.0, //alpha=1
			     small_t_solution_->get_rotation_matrix(), //A
			     &images_view.matrix, //B
			     0.0, //beta=0
			     &images_transformed_view.matrix); //C

	      signs_vector = std::vector<double> (number_images,1.0);
	      unsigned counter = 0;
	      for (unsigned l=0; l<2; ++l) {
		for (unsigned k=0; k<2; ++k) {
		  for (unsigned j=0; j<2; ++j) {
		    for (unsigned i=0; i<2; ++i) {
		      for (unsigned h=0; h<2; ++h) {
			for (unsigned g=0; g<2; ++g) {

			  reflection_sequence_per_final_image[counter] =
			    std::vector<unsigned> (0);

			  gsl_vector* current_image = &images_vector[counter].vector;
			  if (i==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[p][0],
								 lines[p][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(p);
			  }
			  if (j==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[o][0],
								 lines[o][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(o);
			  }
			  if (k==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[n][0],
								 lines[n][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(n);
			  }
			  if (l==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[m][0],
								 lines[m][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(m);
			  }
			  if (h==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[q][0],
								 lines[q][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(q);
			  }
			  if (g==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[r][0],
								 lines[r][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(r);
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

			  distance_from_image_to_line[counter][0] = d1;
			  distance_from_image_to_line[counter][1] = d2;
			  distance_from_image_to_line[counter][2] = d3;
			  distance_from_image_to_line[counter][3] = d4;

			  all_images[counter] = BivariateImageWithTime(current_image,
								       1.0,
								       signs_vector[counter],
								       reflection_sequence_per_final_image[counter]);



			  counter = counter + 1;
			}
		      }
		    }
		  }
		}
	      }

	      // GENERATING UNIQUE IMAGES FROM THE SET
	      unsigned unique_image_counter = 0;
	      unique_images = std::vector<BivariateImageWithTime> (0);
	      for (const BivariateImageWithTime& current_image : all_images) {
		if (unique_images.empty()) {
		  unique_images.push_back(current_image);
		} else {
		  unsigned n=0;
		  std::vector<bool> distances_to_current_image (unique_images.size());

		  std::generate(distances_to_current_image.begin(),
				distances_to_current_image.end(),
				[&current_image, &n, &unique_images] () {
				  double distance =
				    std::sqrt(
					      std::pow(gsl_vector_get(unique_images[n].get_position(),0)-
						       gsl_vector_get(current_image.get_position(),0),
						       2) +
					      std::pow(gsl_vector_get(unique_images[n].get_position(),1)-
						       gsl_vector_get(current_image.get_position(),1),
						       2));

				  bool out = false;
				  if ((distance <= 1e-13) && //10*std::numeric_limits<double>::epsilon()) &&
				      (current_image.get_mult_factor() == unique_images[n].get_mult_factor())) {
				    out = true;
				  }

				  n++;
				  return out;
				});

		  std::vector<bool>::iterator it = std::find(distances_to_current_image.begin(), distances_to_current_image.end(), true);
		  if (it == std::end(distances_to_current_image)) {
		    unique_images.push_back(current_image);
		  }
		}
	      }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[0][0], 0),
		       gsl_vector_get(lines[0][1], 0),
		       gsl_vector_get(lines[0][0], 1),
		       gsl_vector_get(lines[0][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[1][0], 0),
		       gsl_vector_get(lines[1][1], 0),
		       gsl_vector_get(lines[1][0], 1),
		       gsl_vector_get(lines[1][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[2][0], 0),
		       gsl_vector_get(lines[2][1], 0),
		       gsl_vector_get(lines[2][0], 1),
		       gsl_vector_get(lines[2][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); \n",
		       gsl_vector_get(lines[3][0], 0),
		       gsl_vector_get(lines[3][1], 0),
		       gsl_vector_get(lines[3][0], 1),
		       gsl_vector_get(lines[3][1], 1));
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      (reflection_sequence_per_final_image_current.size() == 0))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }  else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");

		  unique_image_counter++;
		}
	      }


	      // CHECKING IF ANY OF THE UNIQUE IMAGES ARE WITHIN THE DOMAIN
	      int sign = 1;
	      for (unsigned i=1; i<unique_images.size(); ++i) {
		double d1 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[0][0],
						  lines[0][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d2 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[1][0],
						  lines[1][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d3 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[2][0],
						  lines[2][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d4 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[3][0],
						  lines[3][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());

		std::vector<double> d_from_im_to_l {d1, d2, d3, d4};

		std::vector<double>::iterator result = std::min_element(d_from_im_to_l.begin(),
									d_from_im_to_l.end());
		if (!std::signbit(*result)) {
		  sign = -1;
		  break;
		}
	      }

	      // calculating max admissible time
	      double mmax = 1.0;
	      double max_admissible_t = 1.0; //(1.0/0.9) * std::pow(distance_from_image_to_line[0][1]/6.0, 2);

	      while (mmax > 1e-12) {
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

		    for (unsigned i=0; i<unique_images.size(); ++i) { // iterating over unique images
		      const gsl_vector* current_image = unique_images[i].get_position();
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

	      for (unsigned current_image_counter=0;
		   current_image_counter<unique_images.size();
		   ++current_image_counter) {
		gsl_vector* current_image = gsl_vector_alloc(2);
		// C = alpha*op(A)*x + beta*C
		gsl_blas_dgemv(CblasNoTrans, //op(A) = A
			       1.0, //alpha=1
			       &Rotation_matrix_inv_view.matrix, //A
			       unique_images[current_image_counter].get_position(), //x
			       0.0, //beta=0
			       current_image); //C

		unique_images[current_image_counter] =
		  BivariateImageWithTime(current_image,
					 sign*max_admissible_t,
					 unique_images[current_image_counter].get_mult_factor(),
					 unique_images[current_image_counter].get_reflection_sequence());
		gsl_vector_free(current_image);
	      }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(0,1), y=c(0,0)); ");
		printf("lines(x=c(1,1), y=c(0,1)); ");
		printf("lines(x=c(1,0), y=c(1,1)); ");
		printf("lines(x=c(0,0), y=c(1,0)); \n");
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      (reflection_sequence_per_final_image_current.size() == 0))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }  else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");
		}
	      }

	      image_counter = image_counter + 1;
	    }
	  }
	}
      }
    }
  }

  return unique_images;
}

std::vector<BivariateImageWithTime> BivariateSolver::
small_t_image_positions_type_4(bool PRINT) const
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

  unsigned number_images = 64;
  double images_array [number_images*2];
  for (unsigned i=0; i<number_images; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+number_images] = get_y_0_2();
  }
  double images_transformed_array [number_images*2];

  gsl_matrix_view images_view =
    gsl_matrix_view_array(images_array, 2, number_images);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, number_images);

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

  std::vector<gsl_vector_view> images_vector (number_images);
  std::vector<BivariateImageWithTime> all_images (number_images);

  for (unsigned i=0; i<number_images; ++i) {
    images_vector[i] =
      gsl_matrix_column(&images_transformed_view.matrix,i);
  }

  std::vector<std::vector<double>> distance_from_image_to_line (number_images, std::vector<double> (4));
  std::vector<double> max_admissible_times (1); // ONLY ONE SET OF IMAGES FOR NOW
  std::vector<BivariateImageWithTime> final_images (number_images);
  std::vector<std::vector<unsigned>> reflection_sequence_per_final_image (number_images, std::vector<unsigned> (0));
  std::vector<double> signs_vector = std::vector<double> (number_images,1.0);

  std::vector<BivariateImageWithTime> unique_images (0);
  unsigned image_counter = 0; // iterating over the number_images

  std::vector<unsigned> p_indeces {1};

  for (unsigned p : p_indeces) {
    std::vector<unsigned> o_indeces {3};

    for (unsigned o : o_indeces) {
      std::vector<unsigned> n_indeces {1};

      for (unsigned n : n_indeces) {
  	std::vector<unsigned> m_indeces {2};

  	for (unsigned m : m_indeces) {
	  std::vector<unsigned> q_indeces {0};

	  for (unsigned q : q_indeces) {
	    std::vector<unsigned> r_indeces {2};

	    for (unsigned r : r_indeces) {
	      // std::cout << "## (p=" << p
	      // 		<< "o=" << o
	      // 		<< ",n=" << n
	      // 		<< ",m=" << m
	      // 		<< ",q=" << q
	      // 		<< ",r=" << r
	      // 		<< ")" << std::endl;


	      gsl_blas_dgemm(CblasNoTrans, //op(A) = A
			     CblasNoTrans, //op(B) = B
			     1.0, //alpha=1
			     small_t_solution_->get_rotation_matrix(), //A
			     &images_view.matrix, //B
			     0.0, //beta=0
			     &images_transformed_view.matrix); //C

	      signs_vector = std::vector<double> (number_images,1.0);
	      unsigned counter = 0;
	      for (unsigned l=0; l<2; ++l) {
		for (unsigned k=0; k<2; ++k) {
		  for (unsigned j=0; j<2; ++j) {
		    for (unsigned i=0; i<2; ++i) {
		      for (unsigned h=0; h<2; ++h) {
			for (unsigned g=0; g<2; ++g) {

			  reflection_sequence_per_final_image[counter] =
			    std::vector<unsigned> (0);

			  gsl_vector* current_image = &images_vector[counter].vector;
			  if (i==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[p][0],
								 lines[p][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(p);
			  }
			  if (j==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[o][0],
								 lines[o][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(o);
			  }
			  if (k==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[n][0],
								 lines[n][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(n);
			  }
			  if (l==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[m][0],
								 lines[m][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(m);
			  }
			  if (h==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[q][0],
								 lines[q][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(q);
			  }
			  if (g==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[r][0],
								 lines[r][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(r);
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

			  distance_from_image_to_line[counter][0] = d1;
			  distance_from_image_to_line[counter][1] = d2;
			  distance_from_image_to_line[counter][2] = d3;
			  distance_from_image_to_line[counter][3] = d4;

			  all_images[counter] = BivariateImageWithTime(current_image,
								       1.0,
								       signs_vector[counter],
								       reflection_sequence_per_final_image[counter]);



			  counter = counter + 1;
			}
		      }
		    }
		  }
		}
	      }

	      // GENERATING UNIQUE IMAGES FROM THE SET
	      unsigned unique_image_counter = 0;
	      unique_images = std::vector<BivariateImageWithTime> (0);
	      for (const BivariateImageWithTime& current_image : all_images) {
		if (unique_images.empty()) {
		  unique_images.push_back(current_image);
		} else {
		  unsigned n=0;
		  std::vector<bool> distances_to_current_image (unique_images.size());

		  std::generate(distances_to_current_image.begin(),
				distances_to_current_image.end(),
				[&current_image, &n, &unique_images] () {
				  double distance =
				    std::sqrt(
					      std::pow(gsl_vector_get(unique_images[n].get_position(),0)-
						       gsl_vector_get(current_image.get_position(),0),
						       2) +
					      std::pow(gsl_vector_get(unique_images[n].get_position(),1)-
						       gsl_vector_get(current_image.get_position(),1),
						       2));

				  bool out = false;
				  if ((distance <= 1e-13) && //10*std::numeric_limits<double>::epsilon()) &&
				      (current_image.get_mult_factor() == unique_images[n].get_mult_factor())) {
				    out = true;
				  }

				  n++;
				  return out;
				});

		  std::vector<bool>::iterator it = std::find(distances_to_current_image.begin(), distances_to_current_image.end(), true);
		  if (it == std::end(distances_to_current_image)) {
		    unique_images.push_back(current_image);
		  }
		}
	      }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[0][0], 0),
		       gsl_vector_get(lines[0][1], 0),
		       gsl_vector_get(lines[0][0], 1),
		       gsl_vector_get(lines[0][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[1][0], 0),
		       gsl_vector_get(lines[1][1], 0),
		       gsl_vector_get(lines[1][0], 1),
		       gsl_vector_get(lines[1][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[2][0], 0),
		       gsl_vector_get(lines[2][1], 0),
		       gsl_vector_get(lines[2][0], 1),
		       gsl_vector_get(lines[2][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); \n",
		       gsl_vector_get(lines[3][0], 0),
		       gsl_vector_get(lines[3][1], 0),
		       gsl_vector_get(lines[3][0], 1),
		       gsl_vector_get(lines[3][1], 1));
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      (reflection_sequence_per_final_image_current.size() == 0))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }  else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");

		  unique_image_counter++;
		}
	      }


	      // CHECKING IF ANY OF THE UNIQUE IMAGES ARE WITHIN THE DOMAIN
	      int sign = 1;
	      for (unsigned i=1; i<unique_images.size(); ++i) {
		double d1 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[0][0],
						  lines[0][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d2 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[1][0],
						  lines[1][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d3 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[2][0],
						  lines[2][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d4 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[3][0],
						  lines[3][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());

		std::vector<double> d_from_im_to_l {d1, d2, d3, d4};

		std::vector<double>::iterator result = std::min_element(d_from_im_to_l.begin(),
									d_from_im_to_l.end());
		if (!std::signbit(*result)) {
		  sign = -1;
		  break;
		}
	      }

	      // calculating max admissible time
	      double mmax = 1.0;
	      double max_admissible_t = 1.0; //(1.0/0.9) * std::pow(distance_from_image_to_line[0][1]/6.0, 2);

	      while (mmax > 1e-12) {
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

		    for (unsigned i=0; i<unique_images.size(); ++i) { // iterating over unique images
		      const gsl_vector* current_image = unique_images[i].get_position();
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

	      for (unsigned current_image_counter=0;
		   current_image_counter<unique_images.size();
		   ++current_image_counter) {
		gsl_vector* current_image = gsl_vector_alloc(2);
		// C = alpha*op(A)*x + beta*C
		gsl_blas_dgemv(CblasNoTrans, //op(A) = A
			       1.0, //alpha=1
			       &Rotation_matrix_inv_view.matrix, //A
			       unique_images[current_image_counter].get_position(), //x
			       0.0, //beta=0
			       current_image); //C

		unique_images[current_image_counter] =
		  BivariateImageWithTime(current_image,
					 sign*max_admissible_t,
					 unique_images[current_image_counter].get_mult_factor(),
					 unique_images[current_image_counter].get_reflection_sequence());
		gsl_vector_free(current_image);
	      }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(0,1), y=c(0,0)); ");
		printf("lines(x=c(1,1), y=c(0,1)); ");
		printf("lines(x=c(1,0), y=c(1,1)); ");
		printf("lines(x=c(0,0), y=c(1,0)); \n");
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      (reflection_sequence_per_final_image_current.size() == 0))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }  else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");
		}
	      }

	      image_counter = image_counter + 1;
	    }
	  }
	}
      }
    }
  }

  std::vector<BivariateImageWithTime> differentiable_images (0);
  for (const BivariateImageWithTime& unique_image : unique_images) {
    std::vector<unsigned> reflection_sequence_per_final_image_current =
      unique_image.get_reflection_sequence();
    if (((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 1) &&
	 (reflection_sequence_per_final_image_current[1] == 3) &&
	 (reflection_sequence_per_final_image_current[2] == 0) &&
	 (reflection_sequence_per_final_image_current[3] == 2)) |
	((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 1) &&
	 (reflection_sequence_per_final_image_current[1] == 3) &&
	 (reflection_sequence_per_final_image_current[2] == 2) &&
	 (reflection_sequence_per_final_image_current[3] == 0)) |
	((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 3) &&
	 (reflection_sequence_per_final_image_current[1] == 1) &&
	 (reflection_sequence_per_final_image_current[2] == 0) &&
	 (reflection_sequence_per_final_image_current[3] == 2)) |
	((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 3) &&
	 (reflection_sequence_per_final_image_current[1] == 1) &&
	 (reflection_sequence_per_final_image_current[2] == 2) &&
	 (reflection_sequence_per_final_image_current[3] == 0)) |
	(reflection_sequence_per_final_image_current.size() > 4))
      {
	differentiable_images.push_back(unique_image);
      }
  }

  return differentiable_images;
}

std::vector<BivariateImageWithTime> BivariateSolver::
small_t_image_positions_type_41(bool PRINT) const
{
  gsl_matrix* Rotation_matrix = gsl_matrix_alloc(2,2);
  double cc = std::sin(M_PI/4.0);

  gsl_matrix_set(Rotation_matrix, 0, 0, cc / (sigma_x_2_*std::sqrt(1.0-rho_)));
  gsl_matrix_set(Rotation_matrix, 1, 0, cc / (sigma_x_2_*std::sqrt(1.0+rho_)));
  gsl_matrix_set(Rotation_matrix, 0, 1, -1.0*cc / (sigma_y_2_*std::sqrt(1-rho_)));
  gsl_matrix_set(Rotation_matrix, 1, 1, cc / (sigma_y_2_*std::sqrt(1+rho_)));

  double Rotation_matrix_inv [4];
  gsl_matrix_view Rotation_matrix_inv_view =
    gsl_matrix_view_array(Rotation_matrix_inv, 2, 2);

  gsl_matrix_memcpy(&Rotation_matrix_inv_view.matrix,
  		    Rotation_matrix);

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

  unsigned number_images = 128;
  double images_array [number_images*2];
  for (unsigned i=0; i<number_images; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+number_images] = get_y_0_2();
  }
  double images_transformed_array [number_images*2];

  gsl_matrix_view images_view =
    gsl_matrix_view_array(images_array, 2, number_images);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, number_images);

  // C = alpha*op(A)*op(B) + beta*C
  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  		 CblasNoTrans, //op(B) = B
  		 1.0, //alpha=1
  		 Rotation_matrix, //A
  		 &corner_points_view.matrix, //B
  		 0.0, //beta=0
  		 &corner_points_transformed_view.matrix); //C

  // C = alpha*op(A)*op(B) + beta*C
  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  		 CblasNoTrans, //op(B) = B
  		 1.0, //alpha=1
  		 Rotation_matrix, //A
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

  std::vector<gsl_vector_view> images_vector (number_images);
  std::vector<BivariateImageWithTime> all_images (number_images);

  for (unsigned i=0; i<number_images; ++i) {
    images_vector[i] =
      gsl_matrix_column(&images_transformed_view.matrix,i);
  }

  std::vector<std::vector<double>> distance_from_image_to_line (number_images, std::vector<double> (4));
  std::vector<double> max_admissible_times (1); // ONLY ONE SET OF IMAGES FOR NOW
  std::vector<BivariateImageWithTime> final_images (number_images);
  std::vector<std::vector<unsigned>> reflection_sequence_per_final_image (number_images, std::vector<unsigned> (0));
  std::vector<double> signs_vector = std::vector<double> (number_images,1.0);

  std::vector<BivariateImageWithTime> unique_images (0);
  unsigned image_counter = 0; // iterating over the number_images

  std::vector<unsigned> p_indeces {3};

  for (unsigned p : p_indeces) {
    std::vector<unsigned> o_indeces {1};

    for (unsigned o : o_indeces) {
      std::vector<unsigned> n_indeces {3};

      for (unsigned n : n_indeces) {
  	std::vector<unsigned> m_indeces {2};

  	for (unsigned m : m_indeces) {
	  std::vector<unsigned> q_indeces {0};

	  for (unsigned q : q_indeces) {
	    std::vector<unsigned> r_indeces {2};

	    for (unsigned r : r_indeces) {
	      std::vector<unsigned> s_indeces {0};

	      for (unsigned s : s_indeces) {
	      // std::cout << "## (p=" << p
	      // 		<< "o=" << o
	      // 		<< ",n=" << n
	      // 		<< ",m=" << m
	      // 		<< ",q=" << q
	      // 		<< ",r=" << r
	      // 		<< ")" << std::endl;


	      gsl_blas_dgemm(CblasNoTrans, //op(A) = A
			     CblasNoTrans, //op(B) = B
			     1.0, //alpha=1
			     Rotation_matrix, //A
			     &images_view.matrix, //B
			     0.0, //beta=0
			     &images_transformed_view.matrix); //C

	      signs_vector = std::vector<double> (number_images,1.0);
	      unsigned counter = 0;
	      for (unsigned l=0; l<2; ++l) {
		for (unsigned k=0; k<2; ++k) {
		  for (unsigned j=0; j<2; ++j) {
		    for (unsigned i=0; i<2; ++i) {
		      for (unsigned h=0; h<2; ++h) {
			for (unsigned g=0; g<2; ++g) {
			  for (unsigned f=0; f<2; ++f) {

			  reflection_sequence_per_final_image[counter] =
			    std::vector<unsigned> (0);

			  gsl_vector* current_image = &images_vector[counter].vector;
			  if (i==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[p][0],
								 lines[p][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(p);
			  }
			  if (j==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[o][0],
								 lines[o][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(o);
			  }
			  if (k==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[n][0],
								 lines[n][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(n);
			  }
			  if (l==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[m][0],
								 lines[m][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(m);
			  }
			  if (h==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[q][0],
								 lines[q][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(q);
			  }
			  if (g==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[r][0],
								 lines[r][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(r);
			  }
			  if (f==1) { //
			    small_t_solution_->reflect_point_raw(lines[s][0],
								 lines[s][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(s);
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

			  distance_from_image_to_line[counter][0] = d1;
			  distance_from_image_to_line[counter][1] = d2;
			  distance_from_image_to_line[counter][2] = d3;
			  distance_from_image_to_line[counter][3] = d4;

			  all_images[counter] = BivariateImageWithTime(current_image,
								       1.0,
								       signs_vector[counter],
								       reflection_sequence_per_final_image[counter]);



			  counter = counter + 1;
			  }
			}
		      }
		    }
		  }
		}
	      }

	      // GENERATING UNIQUE IMAGES FROM THE SET
	      unsigned unique_image_counter = 0;
	      unique_images = std::vector<BivariateImageWithTime> (0);
	      for (const BivariateImageWithTime& current_image : all_images) {
		if (unique_images.empty()) {
		  unique_images.push_back(current_image);
		} else {
		  unsigned n=0;
		  std::vector<bool> distances_to_current_image (unique_images.size());

		  std::generate(distances_to_current_image.begin(),
				distances_to_current_image.end(),
				[&current_image, &n, &unique_images] () {
				  double distance =
				    std::sqrt(
					      std::pow(gsl_vector_get(unique_images[n].get_position(),0)-
						       gsl_vector_get(current_image.get_position(),0),
						       2) +
					      std::pow(gsl_vector_get(unique_images[n].get_position(),1)-
						       gsl_vector_get(current_image.get_position(),1),
						       2));

				  bool out = false;
				  if ((distance <= 1e-12) && //10*std::numeric_limits<double>::epsilon()) &&
				      (current_image.get_mult_factor() == unique_images[n].get_mult_factor())) {
				    out = true;
				  }

				  n++;
				  return out;
				});

		  std::vector<bool>::iterator it = std::find(distances_to_current_image.begin(),
							     distances_to_current_image.end(), true);

		  if (it == std::end(distances_to_current_image)) {
		    unique_images.push_back(current_image);
		  }
		}
	      }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[0][0], 0),
		       gsl_vector_get(lines[0][1], 0),
		       gsl_vector_get(lines[0][0], 1),
		       gsl_vector_get(lines[0][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[1][0], 0),
		       gsl_vector_get(lines[1][1], 0),
		       gsl_vector_get(lines[1][0], 1),
		       gsl_vector_get(lines[1][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[2][0], 0),
		       gsl_vector_get(lines[2][1], 0),
		       gsl_vector_get(lines[2][0], 1),
		       gsl_vector_get(lines[2][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); \n",
		       gsl_vector_get(lines[3][0], 0),
		       gsl_vector_get(lines[3][1], 0),
		       gsl_vector_get(lines[3][0], 1),
		       gsl_vector_get(lines[3][1], 1));
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      (reflection_sequence_per_final_image_current.size() == 0))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }  else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");

		  unique_image_counter++;
		}
	      }


	      // CHECKING IF ANY OF THE UNIQUE IMAGES ARE WITHIN THE DOMAIN
	      int sign = 1;
	      for (unsigned i=1; i<unique_images.size(); ++i) {
		double d1 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[0][0],
						  lines[0][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d2 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[1][0],
						  lines[1][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d3 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[2][0],
						  lines[2][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d4 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[3][0],
						  lines[3][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());

		std::vector<double> d_from_im_to_l {d1, d2, d3, d4};

		std::vector<double>::iterator result = std::min_element(d_from_im_to_l.begin(),
									d_from_im_to_l.end());
		if (!std::signbit(*result)) {
		  sign = -1;
		  break;
		}
	      }

	      // calculating max admissible time
	      double mmax = 1.0;
	      double max_admissible_t = 1.0; //(1.0/0.9) * std::pow(distance_from_image_to_line[0][1]/6.0, 2);

	      while (mmax > 1e-12) {
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

		    for (unsigned i=0; i<unique_images.size(); ++i) { // iterating over unique images
		      const gsl_vector* current_image = unique_images[i].get_position();
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

	      for (unsigned current_image_counter=0;
		   current_image_counter<unique_images.size();
		   ++current_image_counter) {
		gsl_vector* current_image = gsl_vector_alloc(2);
		// C = alpha*op(A)*x + beta*C
		gsl_blas_dgemv(CblasNoTrans, //op(A) = A
			       1.0, //alpha=1
			       &Rotation_matrix_inv_view.matrix, //A
			       unique_images[current_image_counter].get_position(), //x
			       0.0, //beta=0
			       current_image); //C

		unique_images[current_image_counter] =
		  BivariateImageWithTime(current_image,
					 sign*max_admissible_t,
					 unique_images[current_image_counter].get_mult_factor(),
					 unique_images[current_image_counter].get_reflection_sequence());
		gsl_vector_free(current_image);
	      }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(0,1), y=c(0,0)); ");
		printf("lines(x=c(1,1), y=c(0,1)); ");
		printf("lines(x=c(1,0), y=c(1,1)); ");
		printf("lines(x=c(0,0), y=c(1,0)); \n");
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      (reflection_sequence_per_final_image_current.size() == 0))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }  else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");
		}
	      }

	      image_counter = image_counter + 1;
	      }
	    }
	  }
	}
      }
    }
  }

  std::vector<BivariateImageWithTime> differentiable_images (0);
  for (const BivariateImageWithTime& unique_image : unique_images) {
    std::vector<unsigned> reflection_sequence_per_final_image_current =
      unique_image.get_reflection_sequence();
    if (((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 1) &&
	 (reflection_sequence_per_final_image_current[1] == 3) &&
	 (reflection_sequence_per_final_image_current[2] == 0) &&
	 (reflection_sequence_per_final_image_current[3] == 2)) |
	((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 1) &&
	 (reflection_sequence_per_final_image_current[1] == 3) &&
	 (reflection_sequence_per_final_image_current[2] == 2) &&
	 (reflection_sequence_per_final_image_current[3] == 0)) |
	((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 3) &&
	 (reflection_sequence_per_final_image_current[1] == 1) &&
	 (reflection_sequence_per_final_image_current[2] == 0) &&
	 (reflection_sequence_per_final_image_current[3] == 2)) |
	((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 3) &&
	 (reflection_sequence_per_final_image_current[1] == 1) &&
	 (reflection_sequence_per_final_image_current[2] == 2) &&
	 (reflection_sequence_per_final_image_current[3] == 0)) |
	((reflection_sequence_per_final_image_current.size() > 4) &&
	 (reflection_sequence_per_final_image_current.size() <= 6)))
      {
	differentiable_images.push_back(unique_image);
      }
  }

  gsl_matrix_free(Rotation_matrix);
  return differentiable_images;
}

std::vector<BivariateImageWithTime> BivariateSolver::
small_t_image_positions_type_31(bool PRINT) const
{
  gsl_matrix* Rotation_matrix = gsl_matrix_alloc(2,2);
  double cc = std::sin(M_PI/4.0);

  gsl_matrix_set(Rotation_matrix, 0, 0, cc / (sigma_x_2_*std::sqrt(1.0-rho_)));
  gsl_matrix_set(Rotation_matrix, 1, 0, cc / (sigma_x_2_*std::sqrt(1.0+rho_)));
  gsl_matrix_set(Rotation_matrix, 0, 1, -1.0*cc / (sigma_y_2_*std::sqrt(1-rho_)));
  gsl_matrix_set(Rotation_matrix, 1, 1, cc / (sigma_y_2_*std::sqrt(1+rho_)));

  double Rotation_matrix_inv [4];
  gsl_matrix_view Rotation_matrix_inv_view =
    gsl_matrix_view_array(Rotation_matrix_inv, 2, 2);

  gsl_matrix_memcpy(&Rotation_matrix_inv_view.matrix,
  		    Rotation_matrix);

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

  unsigned number_images = 128;
  double images_array [number_images*2];
  for (unsigned i=0; i<number_images; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+number_images] = get_y_0_2();
  }
  double images_transformed_array [number_images*2];

  gsl_matrix_view images_view =
    gsl_matrix_view_array(images_array, 2, number_images);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, number_images);

  // C = alpha*op(A)*op(B) + beta*C
  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  		 CblasNoTrans, //op(B) = B
  		 1.0, //alpha=1
  		 Rotation_matrix, //A
  		 &corner_points_view.matrix, //B
  		 0.0, //beta=0
  		 &corner_points_transformed_view.matrix); //C

  // C = alpha*op(A)*op(B) + beta*C
  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  		 CblasNoTrans, //op(B) = B
  		 1.0, //alpha=1
  		 Rotation_matrix, //A
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

  std::vector<gsl_vector_view> images_vector (number_images);
  std::vector<BivariateImageWithTime> all_images (number_images);

  for (unsigned i=0; i<number_images; ++i) {
    images_vector[i] =
      gsl_matrix_column(&images_transformed_view.matrix,i);
  }

  std::vector<std::vector<double>> distance_from_image_to_line (number_images, std::vector<double> (4));
  std::vector<double> max_admissible_times (1); // ONLY ONE SET OF IMAGES FOR NOW
  std::vector<BivariateImageWithTime> final_images (number_images);
  std::vector<std::vector<unsigned>> reflection_sequence_per_final_image (number_images, std::vector<unsigned> (0));
  std::vector<double> signs_vector = std::vector<double> (number_images,1.0);

  std::vector<BivariateImageWithTime> unique_images (0);
  unsigned image_counter = 0; // iterating over the number_images

  std::vector<unsigned> p_indeces {1};

  for (unsigned p : p_indeces) {
    std::vector<unsigned> o_indeces {3};

    for (unsigned o : o_indeces) {
      std::vector<unsigned> n_indeces {1};

      for (unsigned n : n_indeces) {
  	std::vector<unsigned> m_indeces {3};

  	for (unsigned m : m_indeces) {
	  std::vector<unsigned> q_indeces {0};

	  for (unsigned q : q_indeces) {
	    std::vector<unsigned> r_indeces {2};

	    for (unsigned r : r_indeces) {
	      std::vector<unsigned> s_indeces {0};

	      for (unsigned s : s_indeces) {
	      // std::cout << "## (p=" << p
	      // 		<< "o=" << o
	      // 		<< ",n=" << n
	      // 		<< ",m=" << m
	      // 		<< ",q=" << q
	      // 		<< ",r=" << r
	      // 		<< ")" << std::endl;


	      gsl_blas_dgemm(CblasNoTrans, //op(A) = A
			     CblasNoTrans, //op(B) = B
			     1.0, //alpha=1
			     Rotation_matrix, //A
			     &images_view.matrix, //B
			     0.0, //beta=0
			     &images_transformed_view.matrix); //C

	      signs_vector = std::vector<double> (number_images,1.0);
	      unsigned counter = 0;
	      for (unsigned l=0; l<2; ++l) {
		for (unsigned k=0; k<2; ++k) {
		  for (unsigned j=0; j<2; ++j) {
		    for (unsigned i=0; i<2; ++i) {
		      for (unsigned h=0; h<2; ++h) {
			for (unsigned g=0; g<2; ++g) {
			  for (unsigned f=0; f<2; ++f) {

			  reflection_sequence_per_final_image[counter] =
			    std::vector<unsigned> (0);

			  gsl_vector* current_image = &images_vector[counter].vector;
			  if (i==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[p][0],
								 lines[p][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(p);
			  }
			  if (j==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[o][0],
								 lines[o][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(o);
			  }
			  if (k==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[n][0],
								 lines[n][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(n);
			  }
			  if (l==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[m][0],
								 lines[m][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(m);
			  }
			  if (h==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[q][0],
								 lines[q][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(q);
			  }
			  if (g==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[r][0],
								 lines[r][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(r);
			  }
			  if (f==1) { //
			    small_t_solution_->reflect_point_raw(lines[s][0],
								 lines[s][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(s);
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

			  distance_from_image_to_line[counter][0] = d1;
			  distance_from_image_to_line[counter][1] = d2;
			  distance_from_image_to_line[counter][2] = d3;
			  distance_from_image_to_line[counter][3] = d4;

			  all_images[counter] = BivariateImageWithTime(current_image,
								       1.0,
								       signs_vector[counter],
								       reflection_sequence_per_final_image[counter]);



			  counter = counter + 1;
			  }
			}
		      }
		    }
		  }
		}
	      }

	      // GENERATING UNIQUE IMAGES FROM THE SET
	      unsigned unique_image_counter = 0;
	      unique_images = std::vector<BivariateImageWithTime> (0);
	      for (const BivariateImageWithTime& current_image : all_images) {
		if (unique_images.empty()) {
		  unique_images.push_back(current_image);
		} else if (current_image.get_reflection_sequence().size() <= 6) {
		  unsigned n=0;
		  std::vector<bool> distances_to_current_image (unique_images.size());

		  std::generate(distances_to_current_image.begin(),
				distances_to_current_image.end(),
				[&current_image, &n, &unique_images] () {
				  double distance =
				    std::sqrt(
					      std::pow(gsl_vector_get(unique_images[n].get_position(),0)-
						       gsl_vector_get(current_image.get_position(),0),
						       2) +
					      std::pow(gsl_vector_get(unique_images[n].get_position(),1)-
						       gsl_vector_get(current_image.get_position(),1),
						       2));

				  bool out = false;
				  if ((distance <= 1e-12) && //10*std::numeric_limits<double>::epsilon()) &&
				      (current_image.get_mult_factor() == unique_images[n].get_mult_factor())) {
				    out = true;
				  }

				  n++;
				  return out;
				});

		  std::vector<bool>::iterator it = std::find(distances_to_current_image.begin(),
							     distances_to_current_image.end(), true);

		  if (it == std::end(distances_to_current_image)) {
		    unique_images.push_back(current_image);
		  }
		}
	      }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[0][0], 0),
		       gsl_vector_get(lines[0][1], 0),
		       gsl_vector_get(lines[0][0], 1),
		       gsl_vector_get(lines[0][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[1][0], 0),
		       gsl_vector_get(lines[1][1], 0),
		       gsl_vector_get(lines[1][0], 1),
		       gsl_vector_get(lines[1][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[2][0], 0),
		       gsl_vector_get(lines[2][1], 0),
		       gsl_vector_get(lines[2][0], 1),
		       gsl_vector_get(lines[2][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); \n",
		       gsl_vector_get(lines[3][0], 0),
		       gsl_vector_get(lines[3][1], 0),
		       gsl_vector_get(lines[3][0], 1),
		       gsl_vector_get(lines[3][1], 1));
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      (reflection_sequence_per_final_image_current.size() == 0))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }  else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");

		  unique_image_counter++;
		}
	      }


	      // CHECKING IF ANY OF THE UNIQUE IMAGES ARE WITHIN THE DOMAIN
	      int sign = 1;
	      for (unsigned i=1; i<unique_images.size(); ++i) {
		double d1 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[0][0],
						  lines[0][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d2 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[1][0],
						  lines[1][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d3 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[2][0],
						  lines[2][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d4 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[3][0],
						  lines[3][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());

		std::vector<double> d_from_im_to_l {d1, d2, d3, d4};

		std::vector<double>::iterator result = std::min_element(d_from_im_to_l.begin(),
									d_from_im_to_l.end());
		if (!std::signbit(*result)) {
		  sign = -1;
		  break;
		}
	      }

	      // calculating max admissible time
	      double mmax = 1.0;
	      double max_admissible_t = 1.0; //(1.0/0.9) * std::pow(distance_from_image_to_line[0][1]/6.0, 2);

	      while (mmax > 1e-12) {
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

		    for (unsigned i=0; i<unique_images.size(); ++i) { // iterating over unique images
		      const gsl_vector* current_image = unique_images[i].get_position();
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

	      for (unsigned current_image_counter=0;
		   current_image_counter<unique_images.size();
		   ++current_image_counter) {
		gsl_vector* current_image = gsl_vector_alloc(2);
		// C = alpha*op(A)*x + beta*C
		gsl_blas_dgemv(CblasNoTrans, //op(A) = A
			       1.0, //alpha=1
			       &Rotation_matrix_inv_view.matrix, //A
			       unique_images[current_image_counter].get_position(), //x
			       0.0, //beta=0
			       current_image); //C

		unique_images[current_image_counter] =
		  BivariateImageWithTime(current_image,
					 sign*max_admissible_t,
					 unique_images[current_image_counter].get_mult_factor(),
					 unique_images[current_image_counter].get_reflection_sequence());
		gsl_vector_free(current_image);
	      }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(0,1), y=c(0,0)); ");
		printf("lines(x=c(1,1), y=c(0,1)); ");
		printf("lines(x=c(1,0), y=c(1,1)); ");
		printf("lines(x=c(0,0), y=c(1,0)); \n");
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      (reflection_sequence_per_final_image_current.size() == 0))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }  else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");
		}
	      }

	      image_counter = image_counter + 1;
	      }
	    }
	  }
	}
      }
    }
  }

  std::vector<BivariateImageWithTime> differentiable_images (0);
  for (const BivariateImageWithTime& unique_image : unique_images) {
    std::vector<unsigned> reflection_sequence_per_final_image_current =
      unique_image.get_reflection_sequence();
    if (((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 1) &&
	 (reflection_sequence_per_final_image_current[1] == 3) &&
	 (reflection_sequence_per_final_image_current[2] == 0) &&
	 (reflection_sequence_per_final_image_current[3] == 2)) |
	((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 1) &&
	 (reflection_sequence_per_final_image_current[1] == 3) &&
	 (reflection_sequence_per_final_image_current[2] == 2) &&
	 (reflection_sequence_per_final_image_current[3] == 0)) |
	((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 3) &&
	 (reflection_sequence_per_final_image_current[1] == 1) &&
	 (reflection_sequence_per_final_image_current[2] == 0) &&
	 (reflection_sequence_per_final_image_current[3] == 2)) |
	((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 3) &&
	 (reflection_sequence_per_final_image_current[1] == 1) &&
	 (reflection_sequence_per_final_image_current[2] == 2) &&
	 (reflection_sequence_per_final_image_current[3] == 0)) |
	((reflection_sequence_per_final_image_current.size() > 4) &&
	 (reflection_sequence_per_final_image_current.size() <= 6)))
      {
	differentiable_images.push_back(unique_image);
      }
  }

  gsl_matrix_free(Rotation_matrix);
  return differentiable_images;
}

std::vector<BivariateImageWithTime> BivariateSolver::
small_t_image_positions_type_41_symmetric(bool PRINT) const
{
  gsl_matrix* Rotation_matrix = gsl_matrix_alloc(2,2);
  double cc = std::sin(M_PI/4.0);

  gsl_matrix_set(Rotation_matrix, 0, 0, cc / (sigma_x_2_*std::sqrt(1.0-rho_)));
  gsl_matrix_set(Rotation_matrix, 1, 0, cc / (sigma_x_2_*std::sqrt(1.0+rho_)));
  gsl_matrix_set(Rotation_matrix, 0, 1, -1.0*cc / (sigma_y_2_*std::sqrt(1-rho_)));
  gsl_matrix_set(Rotation_matrix, 1, 1, cc / (sigma_y_2_*std::sqrt(1+rho_)));

  double Rotation_matrix_inv [4];
  gsl_matrix_view Rotation_matrix_inv_view =
    gsl_matrix_view_array(Rotation_matrix_inv, 2, 2);

  gsl_matrix_memcpy(&Rotation_matrix_inv_view.matrix,
  		    Rotation_matrix);

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

  unsigned number_images = 64;
  double images_array [number_images*2];
  for (unsigned i=0; i<number_images; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+number_images] = get_y_0_2();
  }
  double images_transformed_array [number_images*2];

  gsl_matrix_view images_view =
    gsl_matrix_view_array(images_array, 2, number_images);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, number_images);

  // C = alpha*op(A)*op(B) + beta*C
  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  		 CblasNoTrans, //op(B) = B
  		 1.0, //alpha=1
  		 Rotation_matrix, //A
  		 &corner_points_view.matrix, //B
  		 0.0, //beta=0
  		 &corner_points_transformed_view.matrix); //C

  // C = alpha*op(A)*op(B) + beta*C
  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  		 CblasNoTrans, //op(B) = B
  		 1.0, //alpha=1
  		 Rotation_matrix, //A
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

  std::vector<gsl_vector_view> images_vector (number_images);
  std::vector<BivariateImageWithTime> all_images (number_images);

  for (unsigned i=0; i<number_images; ++i) {
    images_vector[i] =
      gsl_matrix_column(&images_transformed_view.matrix,i);
  }

  std::vector<std::vector<double>> distance_from_image_to_line (number_images,
								std::vector<double> (4));
  std::vector<double> max_admissible_times (1); // ONLY ONE SET OF IMAGES FOR NOW
  std::vector<BivariateImageWithTime> final_images (number_images);
  std::vector<std::vector<unsigned>> reflection_sequence_per_final_image (number_images, std::vector<unsigned> (0));
  std::vector<double> signs_vector = std::vector<double> (number_images,1.0);

  std::vector<BivariateImageWithTime> unique_images (0);
  unsigned image_counter = 0; // iterating over the number_images

  std::vector<unsigned> p_indeces {3};

  for (unsigned p : p_indeces) {
    std::vector<unsigned> o_indeces {1};

    for (unsigned o : o_indeces) {
      std::vector<unsigned> n_indeces {3};

      for (unsigned n : n_indeces) {
  	std::vector<unsigned> m_indeces {0};

  	for (unsigned m : m_indeces) {
	  std::vector<unsigned> q_indeces {2};

	  for (unsigned q : q_indeces) {
	    std::vector<unsigned> r_indeces {0};

	    for (unsigned r : r_indeces) {
	      // std::cout << "## (p=" << p
	      // 		<< "o=" << o
	      // 		<< ",n=" << n
	      // 		<< ",m=" << m
	      // 		<< ",q=" << q
	      // 		<< ",r=" << r
	      // 		<< ")" << std::endl;


	      gsl_blas_dgemm(CblasNoTrans, //op(A) = A
			     CblasNoTrans, //op(B) = B
			     1.0, //alpha=1
			     Rotation_matrix, //A
			     &images_view.matrix, //B
			     0.0, //beta=0
			     &images_transformed_view.matrix); //C

	      signs_vector = std::vector<double> (number_images,1.0);
	      unsigned counter = 0;
	      for (unsigned l=0; l<2; ++l) {
		for (unsigned k=0; k<2; ++k) {
		  for (unsigned j=0; j<2; ++j) {
		    for (unsigned i=0; i<2; ++i) {
		      for (unsigned h=0; h<2; ++h) {
			for (unsigned g=0; g<2; ++g) {

			  reflection_sequence_per_final_image[counter] =
			    std::vector<unsigned> (0);

			  gsl_vector* current_image = &images_vector[counter].vector;
			  if (i==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[p][0],
								 lines[p][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(p);
			  }
			  if (j==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[o][0],
								 lines[o][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(o);
			  }
			  if (k==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[n][0],
								 lines[n][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(n);
			  }
			  if (l==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[m][0],
								 lines[m][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(m);
			  }
			  if (h==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[q][0],
								 lines[q][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(q);
			  }
			  if (g==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[r][0],
								 lines[r][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(r);
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

			  distance_from_image_to_line[counter][0] = d1;
			  distance_from_image_to_line[counter][1] = d2;
			  distance_from_image_to_line[counter][2] = d3;
			  distance_from_image_to_line[counter][3] = d4;

			  all_images[counter] = BivariateImageWithTime(current_image,
								       1.0,
								       signs_vector[counter],
								       reflection_sequence_per_final_image[counter]);
			  counter = counter + 1;
			}
		      }
		    }
		  }
		}
	      }

	      // GENERATING UNIQUE, SYMMETRIC IMAGES FROM THE SET
	      unsigned unique_image_counter = 0;
	      unique_images = std::vector<BivariateImageWithTime> (0);
	      for (const BivariateImageWithTime& current_image : all_images) {
		if (unique_images.empty()) {
		  unique_images.push_back(current_image);
		} else if (current_image.get_reflection_sequence().size() <= 4) {
		  const std::vector<unsigned>& reflection_sequence_per_current_image = current_image.get_reflection_sequence();
		  unsigned n=0;
		  std::vector<bool> distances_to_current_image (unique_images.size());

		  std::generate(distances_to_current_image.begin(),
				distances_to_current_image.end(),
				[&current_image, &n, &unique_images] () {
				  double distance =
				    std::sqrt(
					      std::pow(gsl_vector_get(unique_images[n].get_position(),0)-
						       gsl_vector_get(current_image.get_position(),0),
						       2) +
					      std::pow(gsl_vector_get(unique_images[n].get_position(),1)-
						       gsl_vector_get(current_image.get_position(),1),
						       2));

				  bool out = false;
				  if ((distance <= 1e-12) && //10*std::numeric_limits<double>::epsilon()) &&
				      (current_image.get_mult_factor() == unique_images[n].get_mult_factor())) {
				    out = true;
				  }

				  n++;
				  return out;
				});

		  std::vector<bool>::iterator it = std::find(distances_to_current_image.begin(),
							     distances_to_current_image.end(), true);

		  if ((it == std::end(distances_to_current_image)) &&
		      ((reflection_sequence_per_current_image.size() < 4) |
		      ((reflection_sequence_per_current_image.size() == 4) &&
		       (reflection_sequence_per_current_image[0] == 1) &&
		       (reflection_sequence_per_current_image[1] == 3) &&
		       (reflection_sequence_per_current_image[2] == 0) &&
		       (reflection_sequence_per_current_image[3] == 2)) |
		      ((reflection_sequence_per_current_image.size() == 4) &&
		       (reflection_sequence_per_current_image[0] == 1) &&
		       (reflection_sequence_per_current_image[1] == 3) &&
		       (reflection_sequence_per_current_image[2] == 2) &&
		       (reflection_sequence_per_current_image[3] == 0)) |
		      ((reflection_sequence_per_current_image.size() == 4) &&
		       (reflection_sequence_per_current_image[0] == 3) &&
		       (reflection_sequence_per_current_image[1] == 1) &&
		       (reflection_sequence_per_current_image[2] == 0) &&
		       (reflection_sequence_per_current_image[3] == 2)) |
		      ((reflection_sequence_per_current_image.size() == 4) &&
		       (reflection_sequence_per_current_image[0] == 3) &&
		       (reflection_sequence_per_current_image[1] == 1) &&
		       (reflection_sequence_per_current_image[2] == 2) &&
		       (reflection_sequence_per_current_image[3] == 0)))) {
		    unique_images.push_back(current_image);
		  }
		}
	      }

	      // ////////////////////////////////////////////////////////////////////
	      // std::vector<std::vector<unsigned>> reflections =
	      // 	{std::vector<unsigned> {1,3,0,2,1},
	      // 	 std::vector<unsigned> {1,3,2,0,1},
	      // 	 std::vector<unsigned> {3,1,0,2,3},
	      // 	 std::vector<unsigned> {3,1,2,0,3}};

	      // for (const std::vector<unsigned>& reflection_set : reflections) {
	      // 	gsl_blas_dgemm(CblasNoTrans, //op(A) = A
	      // 		       CblasNoTrans, //op(B) = B
	      // 		       1.0, //alpha=1
	      // 		       Rotation_matrix, //A
	      // 		       &images_view.matrix, //B
	      // 		       0.0, //beta=0
	      // 		       &images_transformed_view.matrix); //C
	      // 	gsl_vector* current_image_position = &images_vector[0].vector;
	      // 	BivariateImageWithTime current_image = BivariateImageWithTime();


	      // 	current_image.set_reflection_sequence(reflection_set);
	      // 	current_image.set_mult_factor(-1.0);

	      // 	for (const unsigned& reflection : reflection_set) {
	      // 	  small_t_solution_->reflect_point_raw(lines[reflection][0],
	      // 					       lines[reflection][1],
	      // 					       current_image_position);
	      // 	}
	      // 	current_image.set_position(current_image_position);
	      // 	unique_images.push_back(current_image);
	      // }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[0][0], 0),
		       gsl_vector_get(lines[0][1], 0),
		       gsl_vector_get(lines[0][0], 1),
		       gsl_vector_get(lines[0][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[1][0], 0),
		       gsl_vector_get(lines[1][1], 0),
		       gsl_vector_get(lines[1][0], 1),
		       gsl_vector_get(lines[1][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[2][0], 0),
		       gsl_vector_get(lines[2][1], 0),
		       gsl_vector_get(lines[2][0], 1),
		       gsl_vector_get(lines[2][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); \n",
		       gsl_vector_get(lines[3][0], 0),
		       gsl_vector_get(lines[3][1], 0),
		       gsl_vector_get(lines[3][0], 1),
		       gsl_vector_get(lines[3][1], 1));
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      (reflection_sequence_per_final_image_current.size() == 0))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }  else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");

		  unique_image_counter++;
		}
	      }

	      // CHECKING IF ANY OF THE UNIQUE IMAGES ARE WITHIN THE DOMAIN
	      int sign = 1;
	      for (unsigned i=1; i<unique_images.size(); ++i) {
		double d1 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[0][0],
						  lines[0][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d2 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[1][0],
						  lines[1][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d3 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[2][0],
						  lines[2][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d4 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[3][0],
						  lines[3][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());

		std::vector<double> d_from_im_to_l {d1, d2, d3, d4};

		std::vector<double>::iterator result = std::min_element(d_from_im_to_l.begin(),
									d_from_im_to_l.end());
		if (!std::signbit(*result)) {
		  sign = -1;
		  break;
		}
	      }

	      // calculating max admissible time
	      double mmax = 1.0;
	      double max_admissible_t = 1.0; //(1.0/0.9) * std::pow(distance_from_image_to_line[0][1]/6.0, 2);

	      while (mmax > 1e-10) {
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

		    for (unsigned i=0; i<unique_images.size(); ++i) { // iterating over unique images
		      const gsl_vector* current_image = unique_images[i].get_position();
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

	      for (unsigned current_image_counter=0;
		   current_image_counter<unique_images.size();
		   ++current_image_counter) {
		gsl_vector* current_image = gsl_vector_alloc(2);
		// C = alpha*op(A)*x + beta*C
		gsl_blas_dgemv(CblasNoTrans, //op(A) = A
			       1.0, //alpha=1
			       &Rotation_matrix_inv_view.matrix, //A
			       unique_images[current_image_counter].get_position(), //x
			       0.0, //beta=0
			       current_image); //C

		unique_images[current_image_counter] =
		  BivariateImageWithTime(current_image,
					 sign*max_admissible_t,
					 unique_images[current_image_counter].get_mult_factor(),
					 unique_images[current_image_counter].get_reflection_sequence());
		gsl_vector_free(current_image);
	      }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(0,1), y=c(0,0)); ");
		printf("lines(x=c(1,1), y=c(0,1)); ");
		printf("lines(x=c(1,0), y=c(1,1)); ");
		printf("lines(x=c(0,0), y=c(1,0)); \n");
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      (reflection_sequence_per_final_image_current.size() == 0))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }  else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");
		}
	      }

	      image_counter = image_counter + 1;
	    }
	  }
	}
      }
    }
  }

  std::vector<BivariateImageWithTime> differentiable_images (0);
  for (const BivariateImageWithTime& unique_image : unique_images) {
    std::vector<unsigned> reflection_sequence_per_final_image_current =
      unique_image.get_reflection_sequence();
    if (((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 1) &&
	 (reflection_sequence_per_final_image_current[1] == 3) &&
	 (reflection_sequence_per_final_image_current[2] == 0) &&
	 (reflection_sequence_per_final_image_current[3] == 2)) |
	((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 1) &&
	 (reflection_sequence_per_final_image_current[1] == 3) &&
	 (reflection_sequence_per_final_image_current[2] == 2) &&
	 (reflection_sequence_per_final_image_current[3] == 0)) |
	((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 3) &&
	 (reflection_sequence_per_final_image_current[1] == 1) &&
	 (reflection_sequence_per_final_image_current[2] == 0) &&
	 (reflection_sequence_per_final_image_current[3] == 2)) |
	((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 3) &&
	 (reflection_sequence_per_final_image_current[1] == 1) &&
	 (reflection_sequence_per_final_image_current[2] == 2) &&
	 (reflection_sequence_per_final_image_current[3] == 0)) |
	(reflection_sequence_per_final_image_current.size() > 4))
      {
	differentiable_images.push_back(unique_image);
      }
  }

  gsl_matrix_free(Rotation_matrix);
  return differentiable_images;
}

std::vector<BivariateImageWithTime> BivariateSolver::
small_t_image_positions_1_3(bool PRINT) const
{
  gsl_matrix* Rotation_matrix = gsl_matrix_alloc(2,2);
  double cc = std::sin(M_PI/4.0);

  gsl_matrix_set(Rotation_matrix, 0, 0, cc / (sigma_x_2_*std::sqrt(1.0-rho_)));
  gsl_matrix_set(Rotation_matrix, 1, 0, cc / (sigma_x_2_*std::sqrt(1.0+rho_)));
  gsl_matrix_set(Rotation_matrix, 0, 1, -1.0*cc / (sigma_y_2_*std::sqrt(1-rho_)));
  gsl_matrix_set(Rotation_matrix, 1, 1, cc / (sigma_y_2_*std::sqrt(1+rho_)));

  double Rotation_matrix_inv [4];
  gsl_matrix_view Rotation_matrix_inv_view =
    gsl_matrix_view_array(Rotation_matrix_inv, 2, 2);

  gsl_matrix_memcpy(&Rotation_matrix_inv_view.matrix,
  		    Rotation_matrix);

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

  gsl_matrix_view corner_points_view =
    gsl_matrix_view_array(corner_points_array,
			  2, 5);

  double corner_points_transformed_array [10];
  gsl_matrix_view corner_points_transformed_view =
    gsl_matrix_view_array(corner_points_transformed_array, 2, 5);

  std::vector<std::vector<unsigned>> reflections
    {std::vector<unsigned> {},
	std::vector<unsigned> {1},
	  std::vector<unsigned> {3}};


  unsigned type_1 = 1;
  unsigned type_2 = 3;

  for (unsigned ii=0; ii<2; ++ii) {
    std::vector<unsigned> current_reflection_set_11 = std::vector<unsigned> {};
    std::vector<unsigned> current_reflection_set_12 = std::vector<unsigned> {};
    std::vector<unsigned> current_reflection_set_21 = std::vector<unsigned> {};
    std::vector<unsigned> current_reflection_set_22 = std::vector<unsigned> {};

    // std::vector<unsigned> current_reflection_set_1 = std::vector<unsigned> {};
    // std::vector<unsigned> current_reflection_set_2 = std::vector<unsigned> {};

    // std::vector<unsigned> current_reflection_set_1_1 = std::vector<unsigned> {};
    // std::vector<unsigned> current_reflection_set_2_1 = std::vector<unsigned> {};
    // std::vector<unsigned> current_reflection_set_1_2 = std::vector<unsigned> {};
    // std::vector<unsigned> current_reflection_set_2_2 = std::vector<unsigned> {};

    std::vector<unsigned> end_1 = std::vector<unsigned> {2,0};
    std::vector<unsigned> end_2 = std::vector<unsigned> {0,2};

    for (unsigned jj=0; jj<=ii; ++jj) {
      current_reflection_set_11.push_back(type_1);
      current_reflection_set_12.push_back(type_1);
      current_reflection_set_21.push_back(type_2);
      current_reflection_set_22.push_back(type_2);

      // current_reflection_set_1.push_back(type_1);
      // current_reflection_set_2.push_back(type_2);

      // current_reflection_set_1_1.push_back(type_1);
      // current_reflection_set_2_1.push_back(type_2);
      // current_reflection_set_1_2.push_back(type_1);
      // current_reflection_set_2_2.push_back(type_2);

      if (type_1 == 1) {
	type_1 = 3;
      } else if (type_1 == 3) {
	type_1 = 1;
      }

      if (type_2 == 1) {
	type_2 = 3;
      } else if (type_2 == 3){
	type_2 = 1;
      }
    }

    // current_reflection_set_1_1.push_back(0);
    // current_reflection_set_2_1.push_back(0);
    // current_reflection_set_1_2.push_back(2);
    // current_reflection_set_2_2.push_back(2);

    current_reflection_set_11.insert( current_reflection_set_11.end(),
				      end_1.begin(),
				      end_1.end() );
    current_reflection_set_12.insert( current_reflection_set_12.end(),
				      end_2.begin(),
				      end_2.end() );
    current_reflection_set_21.insert( current_reflection_set_21.end(),
				      end_1.begin(),
				      end_1.end() );
    current_reflection_set_22.insert( current_reflection_set_22.end(),
				      end_2.begin(),
				      end_2.end() );

    // reflections.push_back(current_reflection_set_1);
    // reflections.push_back(current_reflection_set_2);
    reflections.push_back(current_reflection_set_11);
    reflections.push_back(current_reflection_set_12);
    reflections.push_back(current_reflection_set_21);
    reflections.push_back(current_reflection_set_22);

    // reflections.push_back(current_reflection_set_1_1);
    // reflections.push_back(current_reflection_set_1_2);
    // reflections.push_back(current_reflection_set_2_1);
    // reflections.push_back(current_reflection_set_2_2);

    type_1 = 1;
    type_2 = 3;
  }


  if (PRINT) {
    for (std::vector<unsigned> reflection_set : reflections) {
      for (unsigned ref : reflection_set) {
	std::cout << ref << " ";
      }
      std::cout << std::endl;
    }
  }

  unsigned number_images = reflections.size();
  double images_array [number_images*2];
  for (unsigned i=0; i<number_images; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+number_images] = get_y_0_2();
  }
  double images_transformed_array [number_images*2];

  gsl_matrix_view images_view =
    gsl_matrix_view_array(images_array, 2, number_images);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, number_images);

  // C = alpha*op(A)*op(B) + beta*C
  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  		 CblasNoTrans, //op(B) = B
  		 1.0, //alpha=1
  		 Rotation_matrix, //A
  		 &corner_points_view.matrix, //B
  		 0.0, //beta=0
  		 &corner_points_transformed_view.matrix); //C

  // C = alpha*op(A)*op(B) + beta*C
  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  		 CblasNoTrans, //op(B) = B
  		 1.0, //alpha=1
  		 Rotation_matrix, //A
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
  				      &initial_condition_transformed_view.vector)
      };

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

  std::vector<gsl_vector_view> images_vector (number_images);
  std::vector<BivariateImageWithTime> all_images (0);

  for (unsigned i=0; i<number_images; ++i) {
    images_vector[i] =
      gsl_matrix_column(&images_transformed_view.matrix,i);
  }

  std::vector<std::vector<double>> distance_from_image_to_line (number_images, std::vector<double> (4));
  std::vector<double> max_admissible_times (1); // ONLY ONE SET OF IMAGES FOR NOW
  std::vector<BivariateImageWithTime> final_images (number_images);
  std::vector<std::vector<unsigned>> reflection_sequence_per_final_image (number_images, std::vector<unsigned> (0));
  std::vector<double> signs_vector = std::vector<double> (number_images,1.0);

  std::vector<BivariateImageWithTime> unique_images (0);
  unsigned image_counter = 0; // iterating over the number_images

  for (const std::vector<unsigned>& reflection_set : reflections) {
    gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  		   CblasNoTrans, //op(B) = B
  		   1.0, //alpha=1
  		   Rotation_matrix, //A
  		   &images_view.matrix, //B
  		   0.0, //beta=0
  		   &images_transformed_view.matrix); //C
    gsl_vector* current_image_position = &images_vector[0].vector;
    BivariateImageWithTime current_image = BivariateImageWithTime();

    current_image.set_reflection_sequence(reflection_set);
    current_image.set_mult_factor(1.0);

    if (reflection_set.size() > 0) {
      for (const unsigned& reflection : reflection_set) {
    	small_t_solution_->reflect_point_raw(lines[reflection][0],
    					     lines[reflection][1],
    					     current_image_position);
    	current_image.set_mult_factor(current_image.get_mult_factor()*-1);
      }
    }

    current_image.set_position(current_image_position);
    all_images.push_back(current_image);
  }

  // GENERATING UNIQUE IMAGES FROM THE SET AND NON-SYMMETRIC IMAGES
  unsigned unique_image_counter = 0;
  unique_images = std::vector<BivariateImageWithTime> (0);
  for (const BivariateImageWithTime& current_image : all_images) {
    if (unique_images.empty()) {
      unique_images.push_back(current_image);
    } else {
      unsigned n=0;
      std::vector<bool> distances_to_current_image (unique_images.size());

      std::generate(distances_to_current_image.begin(),
  		    distances_to_current_image.end(),
  		    [&current_image, &n, &unique_images] () {
  		      double distance =
  			std::sqrt(
  				  std::pow(gsl_vector_get(unique_images[n].get_position(),0)-
  					   gsl_vector_get(current_image.get_position(),0),
  					   2) +
  				  std::pow(gsl_vector_get(unique_images[n].get_position(),1)-
  					   gsl_vector_get(current_image.get_position(),1),
  					   2));

  		      bool out = false;
  		      if ((distance <= 1e-12) && //10*std::numeric_limits<double>::epsilon()) &&
  			  (current_image.get_mult_factor() == unique_images[n].get_mult_factor())) {
  			out = true;
  		      }

  		      n++;
  		      return out;
  		    });

      std::vector<bool>::iterator it = std::find(distances_to_current_image.begin(),
  						 distances_to_current_image.end(), true);

      if (it == std::end(distances_to_current_image)) {
  	unique_images.push_back(current_image);
      }
    }
  }

  if (PRINT) {
    printf("plot(x=c(-1000,1000),y=c(-1000,1000),type=\"n\");\n");
    printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
  	   gsl_vector_get(lines[0][0], 0),
  	   gsl_vector_get(lines[0][1], 0),
  	   gsl_vector_get(lines[0][0], 1),
  	   gsl_vector_get(lines[0][1], 1));
    printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
  	   gsl_vector_get(lines[1][0], 0),
  	   gsl_vector_get(lines[1][1], 0),
  	   gsl_vector_get(lines[1][0], 1),
  	   gsl_vector_get(lines[1][1], 1));
    printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
  	   gsl_vector_get(lines[2][0], 0),
  	   gsl_vector_get(lines[2][1], 0),
  	   gsl_vector_get(lines[2][0], 1),
  	   gsl_vector_get(lines[2][1], 1));
    printf("lines(x=c(%g,%g), y=c(%g,%g)); \n",
  	   gsl_vector_get(lines[3][0], 0),
  	   gsl_vector_get(lines[3][1], 0),
  	   gsl_vector_get(lines[3][0], 1),
  	   gsl_vector_get(lines[3][1], 1));

    // SLOPE AND INTERCEPT OF LINE PARALLEL TO 2
    // slope
    double delta_x =
      gsl_vector_get(lines[2][0],0)-
      gsl_vector_get(lines[2][1],0);
    double delta_y =
      gsl_vector_get(lines[2][0],1)-
      gsl_vector_get(lines[2][1],1);
    double slope = delta_y/delta_x;
    // intercept => b = y-mx
    double intercept = gsl_vector_get(lines[2][1],1) -
      slope*gsl_vector_get(lines[2][1],0);
    printf("abline(a=%g, b=%g, lty=3);\n ",
	   intercept, slope);

    // SLOPE AND INTERCEPT OF LINE PARALLEL TO 0
    // slope
    delta_x =
      gsl_vector_get(lines[0][0],0)-
      gsl_vector_get(lines[0][1],0);
    delta_y =
      gsl_vector_get(lines[0][0],1)-
      gsl_vector_get(lines[0][1],1);
    slope = delta_y/delta_x;
    // intercept => b = y-mx
    intercept = gsl_vector_get(lines[0][1],1) -
      slope*gsl_vector_get(lines[0][1],0);
    printf("abline(a=%g, b=%g, lty=3);\n ",
	   intercept, slope);

    // SLOPE AND INTERCEPT OF LINE PERPENDICULAR TO LINES 1 AND 3
    // slope
    BivariateImageWithTime IC = unique_images[0];
    BivariateImageWithTime IC_accross_2 = unique_images[1];
    delta_x =
      gsl_vector_get(IC_accross_2.get_position(),0)-
      gsl_vector_get(IC.get_position(),0);
    delta_y =
      gsl_vector_get(IC_accross_2.get_position(),1)-
      gsl_vector_get(IC.get_position(),1);
    slope = delta_y/delta_x;
    // intercept
    // y = mx + b => b = y-mx
    intercept = gsl_vector_get(IC.get_position(),1) -
      slope*gsl_vector_get(IC.get_position(),0);
    // LINE OF REFLECTIONS ABOUT 2 AND 4
    printf("abline(a=%g, b=%g, lty=2);\n ",
	   intercept, slope);

    // SLOPE AND INTERCEPT OF LINE PERPENDICULAR TO LINES 1 AND 3
    // AFTER REFLECTION ABOUT 0 THEN 2
    // slope
    IC = unique_images[4];
    IC_accross_2 = unique_images[6];
    delta_x =
      gsl_vector_get(IC_accross_2.get_position(),0)-
      gsl_vector_get(IC.get_position(),0);
    delta_y =
      gsl_vector_get(IC_accross_2.get_position(),1)-
      gsl_vector_get(IC.get_position(),1);
    slope = delta_y/delta_x;
    // intercept
    // y = mx + b => b = y-mx
    intercept = gsl_vector_get(IC.get_position(),1) -
      slope*gsl_vector_get(IC.get_position(),0);
    // LINE OF REFLECTIONS ABOUT 2 AND 4
    printf("abline(a=%g, b=%g, lty=2);\n ",
	   intercept, slope);

    // SLOPE AND INTERCEPT OF LINE PERPENDICULAR TO LINES 1 AND 3
    // AFTER REFLECTION ABOUT 2 THEN 0
    // slope
    IC = unique_images[3];
    IC_accross_2 = unique_images[5];
    delta_x =
      gsl_vector_get(IC_accross_2.get_position(),0)-
      gsl_vector_get(IC.get_position(),0);
    delta_y =
      gsl_vector_get(IC_accross_2.get_position(),1)-
      gsl_vector_get(IC.get_position(),1);
    slope = delta_y/delta_x;
    // intercept
    // y = mx + b => b = y-mx
    intercept = gsl_vector_get(IC.get_position(),1) -
      slope*gsl_vector_get(IC.get_position(),0);
    // LINE OF REFLECTIONS ABOUT 2 AND 4
    printf("abline(a=%g, b=%g, lty=2);\n ",
	   intercept, slope);

    // SLOPE AND INTERCEPT OF LINE PERPENDICULAR TO LINES 1 AND 3
    // AFTER REFLECTION ABOUT 0
    // slope
    IC = unique_images[7];
    IC_accross_2 = unique_images[9];
    delta_x =
      gsl_vector_get(IC_accross_2.get_position(),0)-
      gsl_vector_get(IC.get_position(),0);
    delta_y =
      gsl_vector_get(IC_accross_2.get_position(),1)-
      gsl_vector_get(IC.get_position(),1);
    slope = delta_y/delta_x;
    // intercept
    // y = mx + b => b = y-mx
    intercept = gsl_vector_get(IC.get_position(),1) -
      slope*gsl_vector_get(IC.get_position(),0);
    // LINE OF REFLECTIONS ABOUT 2 AND 4
    printf("abline(a=%g, b=%g, lty=3);\n ",
	   intercept, slope);

    // SLOPE AND INTERCEPT OF LINE PERPENDICULAR TO LINES 1 AND 3
    // AFTER REFLECTION ABOUT 2
    // slope
    IC = unique_images[8];
    IC_accross_2 = unique_images[10];
    delta_x =
      gsl_vector_get(IC_accross_2.get_position(),0)-
      gsl_vector_get(IC.get_position(),0);
    delta_y =
      gsl_vector_get(IC_accross_2.get_position(),1)-
      gsl_vector_get(IC.get_position(),1);
    slope = delta_y/delta_x;
    // intercept
    // y = mx + b => b = y-mx
    intercept = gsl_vector_get(IC.get_position(),1) -
      slope*gsl_vector_get(IC.get_position(),0);
    // LINE OF REFLECTIONS ABOUT 2 AND 4
    printf("abline(a=%g, b=%g, lty=3);\n ",
	   intercept, slope);


    for (const BivariateImageWithTime& unique_image : unique_images) {
      std::vector<unsigned> reflection_sequence_per_final_image_current =
  	unique_image.get_reflection_sequence();

      if (((reflection_sequence_per_final_image_current.size() == 4) &&
  	   (reflection_sequence_per_final_image_current[0] == 1) &&
  	   (reflection_sequence_per_final_image_current[1] == 3) &&
  	   (reflection_sequence_per_final_image_current[2] == 0) &&
  	   (reflection_sequence_per_final_image_current[3] == 2)) |
  	  ((reflection_sequence_per_final_image_current.size() == 4) &&
  	   (reflection_sequence_per_final_image_current[0] == 1) &&
  	   (reflection_sequence_per_final_image_current[1] == 3) &&
  	   (reflection_sequence_per_final_image_current[2] == 2) &&
  	   (reflection_sequence_per_final_image_current[3] == 0)) |
  	  ((reflection_sequence_per_final_image_current.size() == 4) &&
  	   (reflection_sequence_per_final_image_current[0] == 3) &&
  	   (reflection_sequence_per_final_image_current[1] == 1) &&
  	   (reflection_sequence_per_final_image_current[2] == 0) &&
  	   (reflection_sequence_per_final_image_current[3] == 2)) |
  	  ((reflection_sequence_per_final_image_current.size() == 4) &&
  	   (reflection_sequence_per_final_image_current[0] == 3) &&
  	   (reflection_sequence_per_final_image_current[1] == 1) &&
  	   (reflection_sequence_per_final_image_current[2] == 2) &&
  	   (reflection_sequence_per_final_image_current[3] == 0)) |
  	  (reflection_sequence_per_final_image_current.size() == 0))
  	{
  	  printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
  		 unique_image_counter,
  		 gsl_vector_get(unique_image.get_position(),0),
  		 gsl_vector_get(unique_image.get_position(),1),
  		 unique_image_counter,
  		 unique_image.get_mult_factor(),
  		 gsl_vector_get(unique_image.get_position(),0),
  		 gsl_vector_get(unique_image.get_position(),1));
  	}  else if (reflection_sequence_per_final_image_current.size() > 4) {
  	printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
  	       unique_image_counter,
  	       gsl_vector_get(unique_image.get_position(),0),
  	       gsl_vector_get(unique_image.get_position(),1),
  	       unique_image_counter,
  	       unique_image.get_mult_factor(),
  	       gsl_vector_get(unique_image.get_position(),0),
  	       gsl_vector_get(unique_image.get_position(),1));
      } else
  	{
  	  printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
  		 unique_image_counter,
  		 gsl_vector_get(unique_image.get_position(),0),
  		 gsl_vector_get(unique_image.get_position(),1),
  		 unique_image_counter,
  		 unique_image.get_mult_factor(),
  		 gsl_vector_get(unique_image.get_position(),0),
  		 gsl_vector_get(unique_image.get_position(),1));
  	}

      for (const unsigned& refl : reflection_sequence_per_final_image_current) {
  	printf("%i ", refl);
      }

      printf("\n");

      unique_image_counter++;
    }
  }


  // CHECKING IF ANY OF THE UNIQUE IMAGES ARE WITHIN THE DOMAIN
  int sign = 1;
  for (unsigned i=1; i<unique_images.size(); ++i) {
    double d1 = small_t_solution_->
      distance_from_point_to_axis_raw(lines[0][0],
  				      lines[0][1],
  				      &initial_condition_transformed_view.vector,
  				      unique_images[i].get_position());
    double d2 = small_t_solution_->
      distance_from_point_to_axis_raw(lines[1][0],
  				      lines[1][1],
  				      &initial_condition_transformed_view.vector,
  				      unique_images[i].get_position());
    double d3 = small_t_solution_->
      distance_from_point_to_axis_raw(lines[2][0],
  				      lines[2][1],
  				      &initial_condition_transformed_view.vector,
  				      unique_images[i].get_position());
    double d4 = small_t_solution_->
      distance_from_point_to_axis_raw(lines[3][0],
  				      lines[3][1],
  				      &initial_condition_transformed_view.vector,
  				      unique_images[i].get_position());

    std::vector<double> d_from_im_to_l {d1, d2, d3, d4};

    std::vector<double>::iterator result = std::min_element(d_from_im_to_l.begin(),
  							    d_from_im_to_l.end());
    if (!std::signbit(*result)) {
      sign = -1;
      break;
    }
  }

  // calculating max admissible time
  double mmax = 1.0;
  double max_admissible_t = 1.0; //(1.0/0.9) * std::pow(distance_from_image_to_line[0][1]/6.0, 2);

  while (mmax > 1e-12) {
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

  	for (unsigned i=0; i<unique_images.size(); ++i) { // iterating over unique images
  	  const gsl_vector* current_image = unique_images[i].get_position();
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
  max_admissible_times[0] = sign*max_admissible_t;

  for (unsigned current_image_counter=0;
       current_image_counter<unique_images.size();
       ++current_image_counter) {

    gsl_vector* current_image = gsl_vector_alloc(2);
    // C = alpha*op(A)*x + beta*C
    gsl_blas_dgemv(CblasNoTrans, //op(A) = A
  		   1.0, //alpha=1
  		   &Rotation_matrix_inv_view.matrix, //A
  		   unique_images[current_image_counter].get_position(), //x
  		   0.0, //beta=0
  		   current_image); //C

    unique_images[current_image_counter] =
      BivariateImageWithTime(current_image,
  			     sign*max_admissible_t,
  			     unique_images[current_image_counter].get_mult_factor(),
  			     unique_images[current_image_counter].get_reflection_sequence());
    gsl_vector_free(current_image);
  }

  if (PRINT) {
    printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
    printf("lines(x=c(0,1), y=c(0,0)); ");
    printf("lines(x=c(1,1), y=c(0,1)); ");
    printf("lines(x=c(1,0), y=c(1,1)); ");
    printf("lines(x=c(0,0), y=c(1,0)); \n");
    for (const BivariateImageWithTime& unique_image : unique_images) {
      std::vector<unsigned> reflection_sequence_per_final_image_current =
  	unique_image.get_reflection_sequence();

      if (((reflection_sequence_per_final_image_current.size() == 4) &&
  	   (reflection_sequence_per_final_image_current[0] == 1) &&
  	   (reflection_sequence_per_final_image_current[1] == 3) &&
  	   (reflection_sequence_per_final_image_current[2] == 0) &&
  	   (reflection_sequence_per_final_image_current[3] == 2)) |
  	  ((reflection_sequence_per_final_image_current.size() == 4) &&
  	   (reflection_sequence_per_final_image_current[0] == 1) &&
  	   (reflection_sequence_per_final_image_current[1] == 3) &&
  	   (reflection_sequence_per_final_image_current[2] == 2) &&
  	   (reflection_sequence_per_final_image_current[3] == 0)) |
  	  ((reflection_sequence_per_final_image_current.size() == 4) &&
  	   (reflection_sequence_per_final_image_current[0] == 3) &&
  	   (reflection_sequence_per_final_image_current[1] == 1) &&
  	   (reflection_sequence_per_final_image_current[2] == 0) &&
  	   (reflection_sequence_per_final_image_current[3] == 2)) |
  	  ((reflection_sequence_per_final_image_current.size() == 4) &&
  	   (reflection_sequence_per_final_image_current[0] == 3) &&
  	   (reflection_sequence_per_final_image_current[1] == 1) &&
  	   (reflection_sequence_per_final_image_current[2] == 2) &&
  	   (reflection_sequence_per_final_image_current[3] == 0)) |
  	  (reflection_sequence_per_final_image_current.size() == 0))
  	{
  	  printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); time=%g; ## LOOK HERE ",
  		 unique_image_counter,
  		 gsl_vector_get(unique_image.get_position(),0),
  		 gsl_vector_get(unique_image.get_position(),1),
  		 unique_image_counter,
  		 unique_image.get_mult_factor(),
  		 gsl_vector_get(unique_image.get_position(),0),
  		 gsl_vector_get(unique_image.get_position(),1),
		 unique_image.get_t());
  	}  else if (reflection_sequence_per_final_image_current.size() > 4) {
  	printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
  	       unique_image_counter,
  	       gsl_vector_get(unique_image.get_position(),0),
  	       gsl_vector_get(unique_image.get_position(),1),
  	       unique_image_counter,
  	       unique_image.get_mult_factor(),
  	       gsl_vector_get(unique_image.get_position(),0),
  	       gsl_vector_get(unique_image.get_position(),1));
      } else
  	{
  	  printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
  		 unique_image_counter,
  		 gsl_vector_get(unique_image.get_position(),0),
  		 gsl_vector_get(unique_image.get_position(),1),
  		 unique_image_counter,
  		 unique_image.get_mult_factor(),
  		 gsl_vector_get(unique_image.get_position(),0),
  		 gsl_vector_get(unique_image.get_position(),1));
  	}

      for (const unsigned& refl : reflection_sequence_per_final_image_current) {
  	printf("%i ", refl);
      }

      printf("\n");
    }
  }

  std::vector<BivariateImageWithTime> differentiable_images (0);
  for (const BivariateImageWithTime& unique_image : unique_images) {
    std::vector<unsigned> reflection_sequence_per_final_image_current =
      unique_image.get_reflection_sequence();
    if (reflection_sequence_per_final_image_current.size() >= 4)
      {
  	differentiable_images.push_back(unique_image);
      }
  }

  gsl_matrix_free(Rotation_matrix);
  return differentiable_images;
}

std::vector<BivariateImageWithTime> BivariateSolver::
small_t_image_positions_type_41_all(bool PRINT) const
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

  unsigned number_images = 64;
  double images_array [number_images*2];
  for (unsigned i=0; i<number_images; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+number_images] = get_y_0_2();
  }
  double images_transformed_array [number_images*2];

  gsl_matrix_view images_view =
    gsl_matrix_view_array(images_array, 2, number_images);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, number_images);

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

  std::vector<gsl_vector_view> images_vector (number_images);
  std::vector<BivariateImageWithTime> all_images (number_images);

  for (unsigned i=0; i<number_images; ++i) {
    images_vector[i] =
      gsl_matrix_column(&images_transformed_view.matrix,i);
  }

  std::vector<std::vector<double>> distance_from_image_to_line (number_images, std::vector<double> (4));
  std::vector<double> max_admissible_times (1); // ONLY ONE SET OF IMAGES FOR NOW
  std::vector<BivariateImageWithTime> final_images (number_images);
  std::vector<std::vector<unsigned>> reflection_sequence_per_final_image (number_images, std::vector<unsigned> (0));
  std::vector<double> signs_vector = std::vector<double> (number_images,1.0);

  std::vector<BivariateImageWithTime> unique_images (0);
  unsigned image_counter = 0; // iterating over the number_images

  std::vector<unsigned> p_indeces {3};

  for (unsigned p : p_indeces) {
    std::vector<unsigned> o_indeces {1};

    for (unsigned o : o_indeces) {
      std::vector<unsigned> n_indeces {3};

      for (unsigned n : n_indeces) {
  	std::vector<unsigned> m_indeces {2};

  	for (unsigned m : m_indeces) {
	  std::vector<unsigned> q_indeces {0};

	  for (unsigned q : q_indeces) {
	    std::vector<unsigned> r_indeces {2};

	    for (unsigned r : r_indeces) {
	      // std::cout << "## (p=" << p
	      // 		<< "o=" << o
	      // 		<< ",n=" << n
	      // 		<< ",m=" << m
	      // 		<< ",q=" << q
	      // 		<< ",r=" << r
	      // 		<< ")" << std::endl;


	      gsl_blas_dgemm(CblasNoTrans, //op(A) = A
			     CblasNoTrans, //op(B) = B
			     1.0, //alpha=1
			     small_t_solution_->get_rotation_matrix(), //A
			     &images_view.matrix, //B
			     0.0, //beta=0
			     &images_transformed_view.matrix); //C

	      signs_vector = std::vector<double> (number_images,1.0);
	      unsigned counter = 0;
	      for (unsigned l=0; l<2; ++l) {
		for (unsigned k=0; k<2; ++k) {
		  for (unsigned j=0; j<2; ++j) {
		    for (unsigned i=0; i<2; ++i) {
		      for (unsigned h=0; h<2; ++h) {
			for (unsigned g=0; g<2; ++g) {

			  reflection_sequence_per_final_image[counter] =
			    std::vector<unsigned> (0);

			  gsl_vector* current_image = &images_vector[counter].vector;
			  if (i==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[p][0],
								 lines[p][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(p);
			  }
			  if (j==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[o][0],
								 lines[o][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(o);
			  }
			  if (k==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[n][0],
								 lines[n][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(n);
			  }
			  if (l==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[m][0],
								 lines[m][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(m);
			  }
			  if (h==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[q][0],
								 lines[q][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(q);
			  }
			  if (g==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
			    small_t_solution_->reflect_point_raw(lines[r][0],
								 lines[r][1],
								 current_image);
			    signs_vector[counter] = signs_vector[counter]*(-1.0);
			    reflection_sequence_per_final_image[counter].push_back(r);
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

			  distance_from_image_to_line[counter][0] = d1;
			  distance_from_image_to_line[counter][1] = d2;
			  distance_from_image_to_line[counter][2] = d3;
			  distance_from_image_to_line[counter][3] = d4;

			  all_images[counter] = BivariateImageWithTime(current_image,
								       1.0,
								       signs_vector[counter],
								       reflection_sequence_per_final_image[counter]);



			  counter = counter + 1;
			}
		      }
		    }
		  }
		}
	      }

	      // GENERATING UNIQUE IMAGES FROM THE SET
	      unsigned unique_image_counter = 0;
	      unique_images = std::vector<BivariateImageWithTime> (0);
	      for (const BivariateImageWithTime& current_image : all_images) {
		if (unique_images.empty()) {
		  unique_images.push_back(current_image);
		} else {
		  unsigned n=0;
		  std::vector<bool> distances_to_current_image (unique_images.size());

		  std::generate(distances_to_current_image.begin(),
				distances_to_current_image.end(),
				[&current_image, &n, &unique_images] () {
				  double distance =
				    std::sqrt(
					      std::pow(gsl_vector_get(unique_images[n].get_position(),0)-
						       gsl_vector_get(current_image.get_position(),0),
						       2) +
					      std::pow(gsl_vector_get(unique_images[n].get_position(),1)-
						       gsl_vector_get(current_image.get_position(),1),
						       2));

				  bool out = false;
				  if ((distance <= 1e-13) && //10*std::numeric_limits<double>::epsilon()) &&
				      (current_image.get_mult_factor() == unique_images[n].get_mult_factor())) {
				    out = true;
				  }

				  n++;
				  return out;
				});

		  std::vector<bool>::iterator it = std::find(distances_to_current_image.begin(), distances_to_current_image.end(), true);
		  if (it == std::end(distances_to_current_image)) {
		    unique_images.push_back(current_image);
		  }
		}
	      }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[0][0], 0),
		       gsl_vector_get(lines[0][1], 0),
		       gsl_vector_get(lines[0][0], 1),
		       gsl_vector_get(lines[0][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[1][0], 0),
		       gsl_vector_get(lines[1][1], 0),
		       gsl_vector_get(lines[1][0], 1),
		       gsl_vector_get(lines[1][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); ",
		       gsl_vector_get(lines[2][0], 0),
		       gsl_vector_get(lines[2][1], 0),
		       gsl_vector_get(lines[2][0], 1),
		       gsl_vector_get(lines[2][1], 1));
		printf("lines(x=c(%g,%g), y=c(%g,%g)); \n",
		       gsl_vector_get(lines[3][0], 0),
		       gsl_vector_get(lines[3][1], 0),
		       gsl_vector_get(lines[3][0], 1),
		       gsl_vector_get(lines[3][1], 1));
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      (reflection_sequence_per_final_image_current.size() == 0))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }  else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");

		  unique_image_counter++;
		}
	      }


	      // CHECKING IF ANY OF THE UNIQUE IMAGES ARE WITHIN THE DOMAIN
	      int sign = 1;
	      for (unsigned i=1; i<unique_images.size(); ++i) {
		double d1 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[0][0],
						  lines[0][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d2 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[1][0],
						  lines[1][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d3 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[2][0],
						  lines[2][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());
		double d4 = small_t_solution_->
		  distance_from_point_to_axis_raw(lines[3][0],
						  lines[3][1],
						  &initial_condition_transformed_view.vector,
						  unique_images[i].get_position());

		std::vector<double> d_from_im_to_l {d1, d2, d3, d4};

		std::vector<double>::iterator result = std::min_element(d_from_im_to_l.begin(),
									d_from_im_to_l.end());
		if (!std::signbit(*result)) {
		  sign = -1;
		  break;
		}
	      }

	      // calculating max admissible time
	      double mmax = 1.0;
	      double max_admissible_t = 1.0; //(1.0/0.9) * std::pow(distance_from_image_to_line[0][1]/6.0, 2);

	      while (mmax > 1e-12) {
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

		    for (unsigned i=0; i<unique_images.size(); ++i) { // iterating over unique images
		      const gsl_vector* current_image = unique_images[i].get_position();
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

	      for (unsigned current_image_counter=0;
		   current_image_counter<unique_images.size();
		   ++current_image_counter) {
		gsl_vector* current_image = gsl_vector_alloc(2);
		// C = alpha*op(A)*x + beta*C
		gsl_blas_dgemv(CblasNoTrans, //op(A) = A
			       1.0, //alpha=1
			       &Rotation_matrix_inv_view.matrix, //A
			       unique_images[current_image_counter].get_position(), //x
			       0.0, //beta=0
			       current_image); //C

		unique_images[current_image_counter] =
		  BivariateImageWithTime(current_image,
					 sign*max_admissible_t,
					 unique_images[current_image_counter].get_mult_factor(),
					 unique_images[current_image_counter].get_reflection_sequence());
		gsl_vector_free(current_image);
	      }

	      if (PRINT) {
		printf("plot(x=c(-10,10),y=c(-10,10),type=\"n\");\n");
		printf("lines(x=c(0,1), y=c(0,0)); ");
		printf("lines(x=c(1,1), y=c(0,1)); ");
		printf("lines(x=c(1,0), y=c(1,1)); ");
		printf("lines(x=c(0,0), y=c(1,0)); \n");
		for (const BivariateImageWithTime& unique_image : unique_images) {
		  std::vector<unsigned> reflection_sequence_per_final_image_current =
		    unique_image.get_reflection_sequence();

		  if (((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 1) &&
		       (reflection_sequence_per_final_image_current[1] == 3) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 0) &&
		       (reflection_sequence_per_final_image_current[3] == 2)) |
		      ((reflection_sequence_per_final_image_current.size() == 4) &&
		       (reflection_sequence_per_final_image_current[0] == 3) &&
		       (reflection_sequence_per_final_image_current[1] == 1) &&
		       (reflection_sequence_per_final_image_current[2] == 2) &&
		       (reflection_sequence_per_final_image_current[3] == 0)))
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    } else if (reflection_sequence_per_final_image_current.size() == 0)
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"red\"); ## LOOK HERE ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		  } else if (reflection_sequence_per_final_image_current.size() > 4) {
		    printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
			   unique_image_counter,
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1),
			   unique_image_counter,
			   unique_image.get_mult_factor(),
			   gsl_vector_get(unique_image.get_position(),0),
			   gsl_vector_get(unique_image.get_position(),1));
		  } else
		    {
		      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
			     unique_image_counter,
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1),
			     unique_image_counter,
			     unique_image.get_mult_factor(),
			     gsl_vector_get(unique_image.get_position(),0),
			     gsl_vector_get(unique_image.get_position(),1));
		    }

		  for (const unsigned& refl : reflection_sequence_per_final_image_current) {
		    printf("%i ", refl);
		  }

		  printf("\n");
		}
	      }

	      image_counter = image_counter + 1;
	    }
	  }
	}
      }
    }
  }

  std::vector<BivariateImageWithTime> differentiable_images (0);
  for (const BivariateImageWithTime& unique_image : unique_images) {
    std::vector<unsigned> reflection_sequence_per_final_image_current =
      unique_image.get_reflection_sequence();

    differentiable_images.push_back(unique_image);
  }

  return differentiable_images;
}

double BivariateSolver::numerical_solution_small_t(const gsl_vector* input) const
{
  gsl_matrix* Rotation_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_memcpy(Rotation_matrix, small_t_solution_->get_rotation_matrix());

  double Rotation_matrix_inv [4];
  gsl_matrix_view Rotation_matrix_inv_view =
    gsl_matrix_view_array(Rotation_matrix_inv, 2, 2);

  gsl_matrix_memcpy(&Rotation_matrix_inv_view.matrix,
  		    small_t_solution_->get_rotation_matrix());

  int s = 0;
  gsl_permutation * permutation = gsl_permutation_alloc(2);
  gsl_linalg_LU_decomp(Rotation_matrix, permutation, &s);
  gsl_linalg_LU_invert(Rotation_matrix, permutation, &Rotation_matrix_inv_view.matrix);
  gsl_permutation_free(permutation);
  gsl_matrix_free(Rotation_matrix);

  double product [4];
  gsl_matrix_view product_view =
    gsl_matrix_view_array(product, 2,2);
  // C = alpha*op(A)*op(B) + beta*C
  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
  		 CblasNoTrans, //op(B) = B
  		 1.0, //alpha=1
  		 small_t_solution_->get_rotation_matrix(), //A
  		 &Rotation_matrix_inv_view.matrix, //B
  		 0.0, //beta=0
  		 &product_view.matrix); //C
  std::cout << gsl_matrix_get(&product_view.matrix,0,0) << " "
  	    << gsl_matrix_get(&product_view.matrix,0,1) << "\n"
  	    << gsl_matrix_get(&product_view.matrix,1,0) << " "
    	    << gsl_matrix_get(&product_view.matrix,1,1) << std::endl;

  // double cc = std::sin(M_PI/4.0);
  // gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 0,0,
  // 		 0.5/cc * sigma_x_2_*std::sqrt(1-rho_));
  // gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 0,1,
  // 		  0.5/cc * sigma_x_2_*std::sqrt(1+rho_));
  // gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 1,0,
  // 		 -0.5/cc * sigma_y_2_*std::sqrt(1-rho_));
  // gsl_matrix_set(&Rotation_matrix_inv_view.matrix, 1,1,
  // 		 0.5/cc * sigma_y_2_*std::sqrt(1+rho_));

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
  std::vector<double> max_admissible_times (1);
  std::vector<BivariateImageWithTime> final_images (16);
  std::vector<double> signs_vector = std::vector<double> (16,1.0);

  unsigned image_counter = 0;
  unsigned p = 0;
  unsigned o = 1;
  n = 2;
  unsigned m = 3;

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
  max_admissible_times[image_counter] = sign*max_admissible_t;

  gsl_matrix* cov_matrix = gsl_matrix_alloc(2,2);
  gsl_matrix_set(cov_matrix, 0,0,
  		 sigma_x_*sigma_x_*t_);
  gsl_matrix_set(cov_matrix, 0,1,
  		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,0,
  		 rho_*sigma_x_*sigma_y_*t_);
  gsl_matrix_set(cov_matrix, 1,1,
  		 sigma_y_*sigma_y_*t_);
  MultivariateNormal mvtnorm = MultivariateNormal();
  double out = 0;
  for (unsigned ii=0; ii<16; ++ii) {
    gsl_vector* current_image = gsl_vector_alloc(2);
    // C = alpha*op(A)*x + beta*C
    gsl_blas_dgemv(CblasNoTrans, //op(A) = A
		   1.0, //alpha=1
		   &Rotation_matrix_inv_view.matrix, //A
		   &images_vector[ii].vector, //x
		   0.0, //beta=0
		   current_image); //C
    final_images[ii] = BivariateImageWithTime(current_image,
					      sign*max_admissible_t,
					      signs_vector[ii]);

    out = out +
      signs_vector[ii]*mvtnorm.dmvnorm(2,
				       current_image,
				       input,
				       cov_matrix);


    printf("image.%i = c(%g,%g);\n",
	   ii,
	   gsl_vector_get(current_image, 0),
	   gsl_vector_get(current_image, 1));
    printf("points(image.%i[1],image.%i[2], pch=20,lwd=2);\n", ii,ii);
    gsl_vector_free(current_image);
  }
  gsl_matrix_free(cov_matrix);

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

  return out;
}

void BivariateSolver::figure_chapter_3_proof_1() const
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

  unsigned number_images = 64;
  double images_array [number_images*2];
  for (unsigned i=0; i<number_images; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+number_images] = get_y_0_2();
  }
  double images_transformed_array [number_images*2];

  gsl_matrix_view images_view =
    gsl_matrix_view_array(images_array, 2, number_images);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, number_images);

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

  std::vector<BivariateImageWithTime> all_images =
    std::vector<BivariateImageWithTime> (number_images);

  std::vector<gsl_vector_view> images_vector (number_images);
  for (unsigned i=0; i<number_images; ++i) {
    images_vector[i] =
      gsl_matrix_column(&images_transformed_view.matrix,i);
  }

  // REFLECTIONS
  unsigned p=3, o=1, n=3, m=2, q=0, r=2;

  std::vector<double> signs_vector (number_images,1.0);
  unsigned counter = 0;
  for (unsigned l=0; l<2; ++l) {
    for (unsigned k=0; k<2; ++k) {
      for (unsigned j=0; j<2; ++j) {
	for (unsigned i=0; i<2; ++i) {
	  for (unsigned h=0; h<2; ++h) {
	    for (unsigned g=0; g<2; ++g) {

	      std::vector<unsigned> reflection_sequence =
		std::vector<unsigned> (0);
	      gsl_vector* current_image = &images_vector[counter].vector;

	      if (i==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
		small_t_solution_->reflect_point_raw(lines[p][0],
						     lines[p][1],
						     current_image);
		signs_vector[counter] = signs_vector[counter]*(-1.0);
		reflection_sequence.push_back(p);
	      }
	      if (j==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
		small_t_solution_->reflect_point_raw(lines[o][0],
						     lines[o][1],
						     current_image);
		signs_vector[counter] = signs_vector[counter]*(-1.0);
		reflection_sequence.push_back(o);
	      }
	      if (k==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
		small_t_solution_->reflect_point_raw(lines[n][0],
						     lines[n][1],
						     current_image);
		signs_vector[counter] = signs_vector[counter]*(-1.0);
		reflection_sequence.push_back(n);
	      }
	      if (l==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
		small_t_solution_->reflect_point_raw(lines[m][0],
						     lines[m][1],
						     current_image);
		signs_vector[counter] = signs_vector[counter]*(-1.0);
		reflection_sequence.push_back(m);
	      }
	      if (h==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
		small_t_solution_->reflect_point_raw(lines[q][0],
						     lines[q][1],
						     current_image);
		signs_vector[counter] = signs_vector[counter]*(-1.0);
		reflection_sequence.push_back(q);
	      }
	      if (g==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
		small_t_solution_->reflect_point_raw(lines[r][0],
						     lines[r][1],
						     current_image);
		signs_vector[counter] = signs_vector[counter]*(-1.0);
		reflection_sequence.push_back(r);
	      }

	      all_images[counter] = BivariateImageWithTime(current_image,
							   1.0,
							   signs_vector[counter],
							   reflection_sequence);

	      counter = counter + 1;
	    }
	  }
	}
      }
    }
  }

  for (unsigned ii=0; ii<all_images.size(); ++ii) {
    std::cout << "image." << ii << ": ";
    for (unsigned reflection : all_images[ii].get_reflection_sequence()) {
      std::cout << reflection+1 << " ";
    }
    std::cout << std::endl;
  }

  printf("pdf(\"./src/kernel-expansion/documentation/chapter-3/chapter-3-figure-proof-1.pdf\", 6,6)\n");
  printf("plot(x=c(-8,5),y=c(-5,8),xlab=\"\",ylab=\"\",type=\"n\");\n");
  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
	 gsl_vector_get(lines[0][0], 0),
	 gsl_vector_get(lines[0][1], 0),
	 gsl_vector_get(lines[0][0], 1),
	 gsl_vector_get(lines[0][1], 1));
  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
	 gsl_vector_get(lines[1][0], 0),
	 gsl_vector_get(lines[1][1], 0),
	 gsl_vector_get(lines[1][0], 1),
	 gsl_vector_get(lines[1][1], 1));
  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
	 gsl_vector_get(lines[2][0], 0),
	 gsl_vector_get(lines[2][1], 0),
	 gsl_vector_get(lines[2][0], 1),
	 gsl_vector_get(lines[2][1], 1));
  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
	 gsl_vector_get(lines[3][0], 0),
	 gsl_vector_get(lines[3][1], 0),
	 gsl_vector_get(lines[3][0], 1),
	 gsl_vector_get(lines[3][1], 1));
  printf("points(x=%g, y=%g, pch=20, col=\"red\");\n ",
	 gsl_vector_get(&initial_condition_transformed_view.vector,0),
	 gsl_vector_get(&initial_condition_transformed_view.vector,1));
  printf("text(%g,%g-0.5,\"a)\");\n ",
	 gsl_vector_get(&lower_left_transformed_view.vector,0),
	 gsl_vector_get(&lower_left_transformed_view.vector,1));
  printf("text(%g + 0.5, %g,\"b)\");\n ",
	 gsl_vector_get(&lower_right_transformed_view.vector,0),
	 gsl_vector_get(&lower_right_transformed_view.vector,1));
  printf("text(%g, %g + 0.5,\"c)\");\n ",
	 gsl_vector_get(&upper_right_transformed_view.vector,0),
	 gsl_vector_get(&upper_right_transformed_view.vector,1));
  printf("text(%g - 0.5, %g,\"d)\");\n ",
	 gsl_vector_get(&upper_left_transformed_view.vector,0),
	 gsl_vector_get(&upper_left_transformed_view.vector,1));

  // half way between a) and b)
  printf("text(%g, %g-0.5,\"1\");\n ",
	 gsl_vector_get(&lower_left_transformed_view.vector,0) +
	 0.5*(gsl_vector_get(&lower_right_transformed_view.vector,0)-
	      gsl_vector_get(&lower_left_transformed_view.vector,0)),
	 gsl_vector_get(&lower_left_transformed_view.vector,1) +
	 0.5*(gsl_vector_get(&lower_right_transformed_view.vector,1)-
	      gsl_vector_get(&lower_left_transformed_view.vector,1)));

  // half way between b) and c)
  printf("text(%g, %g+0.5,\"2\");\n ",
	 gsl_vector_get(&lower_right_transformed_view.vector,0) +
	 0.5*(gsl_vector_get(&upper_right_transformed_view.vector,0)-
	      gsl_vector_get(&lower_right_transformed_view.vector,0)),
	 gsl_vector_get(&lower_right_transformed_view.vector,1) +
	 0.5*(gsl_vector_get(&upper_right_transformed_view.vector,1)-
	      gsl_vector_get(&lower_right_transformed_view.vector,1)));

  // half way between c) and d)
  printf("text(%g, %g+0.5,\"3\");\n ",
	 gsl_vector_get(&upper_right_transformed_view.vector,0) +
	 0.5*(gsl_vector_get(&upper_left_transformed_view.vector,0)-
	      gsl_vector_get(&upper_right_transformed_view.vector,0)),
	 gsl_vector_get(&upper_right_transformed_view.vector,1) +
	 0.5*(gsl_vector_get(&upper_left_transformed_view.vector,1)-
	      gsl_vector_get(&upper_right_transformed_view.vector,1)));

  // half way between d) and a)
  printf("text(%g, %g-0.5,\"4\");\n ",
	 gsl_vector_get(&upper_left_transformed_view.vector,0) +
	 0.5*(gsl_vector_get(&lower_left_transformed_view.vector,0)-
	      gsl_vector_get(&upper_left_transformed_view.vector,0)),
	 gsl_vector_get(&upper_left_transformed_view.vector,1) +
	 0.5*(gsl_vector_get(&lower_left_transformed_view.vector,1)-
	      gsl_vector_get(&upper_left_transformed_view.vector,1)));

  // SLOPE AND INTERCEPT OF LINE PERPENDICULAR TO LINES 2 AND 4
  // slope
  const BivariateImageWithTime& IC = all_images[0];
  const BivariateImageWithTime& IC_accross_2 = all_images[8];
  double delta_x =
    gsl_vector_get(IC_accross_2.get_position(),0)-
    gsl_vector_get(IC.get_position(),0);
  double delta_y =
    gsl_vector_get(IC_accross_2.get_position(),1)-
    gsl_vector_get(IC.get_position(),1);
  double slope = delta_y/delta_x;
  // intercept
  // y = mx + b => b = y-mx
  double intercept = gsl_vector_get(IC.get_position(),1) -
    slope*gsl_vector_get(IC.get_position(),0);
  // LINE OF REFLECTIONS ABOUT 2 AND 4
  printf("abline(a=%g, b=%g, lty=2);\n ",
	 intercept, slope);

  // IMAGES ALONG DASHED LINE
  for (unsigned ii : std::vector<unsigned> {0, 4,8,12,24,28}) {
    const BivariateImageWithTime& image = all_images[ii];
    if (std::signbit(image.get_mult_factor())) {
      printf("points(x=%g,y=%g,pch=20, col=\"black\");\n ",
	     gsl_vector_get(image.get_position(), 0),
	     gsl_vector_get(image.get_position(), 1));
    } else {
      printf("points(x=%g,y=%g,pch=20, col=\"red\");\n ",
	     gsl_vector_get(image.get_position(), 0),
	     gsl_vector_get(image.get_position(), 1));
    }
  }

  // SLOPE AND INTERCEPT OF LINE PARALLEL TO 1
  // slope
  delta_x =
    gsl_vector_get(lines[0][0],0)-
    gsl_vector_get(lines[0][1],0);
  delta_y =
    gsl_vector_get(lines[0][0],1)-
    gsl_vector_get(lines[0][1],1);
  slope = delta_y/delta_x;
  // intercept => b = y-mx
  intercept = gsl_vector_get(lines[0][1],1) -
    slope*gsl_vector_get(lines[0][1],0);
  printf("abline(a=%g, b=%g, lty=3);\n ",
	 intercept, slope);

  // SLOPE AND INTERCEPT OF LINE PARALLEL TO 3
  // slope
  delta_x =
    gsl_vector_get(lines[2][0],0)-
    gsl_vector_get(lines[2][1],0);
  delta_y =
    gsl_vector_get(lines[2][0],1)-
    gsl_vector_get(lines[2][1],1);
  slope = delta_y/delta_x;
  // intercept => b = y-mx
  intercept = gsl_vector_get(lines[2][1],1) -
    slope*gsl_vector_get(lines[2][1],0);
  printf("abline(a=%g, b=%g, lty=3);\n ",
	 intercept, slope);

  // INTERCEPT OF LINE 3 AND REFLECTION SEGMENT
  double c = std::sqrt(2)/2;

  // through intercept
  double x_intercept =
    (-2.0*x_0_*rho_ + 2*y_0_/sigma_y_ - 2/sigma_y_*(1-rho_))*c/
    (-2*rho_*std::sqrt(1-rho_));

  double y_intercept =

   std::sqrt(1 - rho_)/std::sqrt(1 + rho_)*x_intercept +
    2*c/(sigma_y_*std::sqrt(1+rho_));

  printf("points(x=%g, y=%g, col=\"blue\", pch=20);\n ",
	 x_intercept,
	 y_intercept);

  printf("dev.off();\n ");
}

void BivariateSolver::figure_chapter_3_proof_2() const
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

  unsigned number_images = 64;
  double images_array [number_images*2];
  for (unsigned i=0; i<number_images; ++i) {
    images_array[i] = get_x_0_2();
    images_array[i+number_images] = get_y_0_2();
  }
  double images_transformed_array [number_images*2];

  gsl_matrix_view images_view =
    gsl_matrix_view_array(images_array, 2, number_images);
  gsl_matrix_view images_transformed_view =
    gsl_matrix_view_array(images_transformed_array, 2, number_images);

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

  std::vector<BivariateImageWithTime> all_images =
    std::vector<BivariateImageWithTime> (number_images);

  std::vector<gsl_vector_view> images_vector (number_images);
  for (unsigned i=0; i<number_images; ++i) {
    images_vector[i] =
      gsl_matrix_column(&images_transformed_view.matrix,i);
  }

  // REFLECTIONS
  unsigned p=3, o=1, n=3, m=2, q=0, r=2;

  std::vector<double> signs_vector (number_images,1.0);
  unsigned counter = 0;
  for (unsigned l=0; l<2; ++l) {
    for (unsigned k=0; k<2; ++k) {
      for (unsigned j=0; j<2; ++j) {
	for (unsigned i=0; i<2; ++i) {
	  for (unsigned h=0; h<2; ++h) {
	    for (unsigned g=0; g<2; ++g) {

	      std::vector<unsigned> reflection_sequence =
		std::vector<unsigned> (0);
	      gsl_vector* current_image = &images_vector[counter].vector;

	      if (i==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
		small_t_solution_->reflect_point_raw(lines[p][0],
						     lines[p][1],
						     current_image);
		signs_vector[counter] = signs_vector[counter]*(-1.0);
		reflection_sequence.push_back(p);
	      }
	      if (j==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
		small_t_solution_->reflect_point_raw(lines[o][0],
						     lines[o][1],
						     current_image);
		signs_vector[counter] = signs_vector[counter]*(-1.0);
		reflection_sequence.push_back(o);
	      }
	      if (k==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
		small_t_solution_->reflect_point_raw(lines[n][0],
						     lines[n][1],
						     current_image);
		signs_vector[counter] = signs_vector[counter]*(-1.0);
		reflection_sequence.push_back(n);
	      }
	      if (l==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
		small_t_solution_->reflect_point_raw(lines[m][0],
						     lines[m][1],
						     current_image);
		signs_vector[counter] = signs_vector[counter]*(-1.0);
		reflection_sequence.push_back(m);
	      }
	      if (h==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
		small_t_solution_->reflect_point_raw(lines[q][0],
						     lines[q][1],
						     current_image);
		signs_vector[counter] = signs_vector[counter]*(-1.0);
		reflection_sequence.push_back(q);
	      }
	      if (g==1) { // & reflection_sequence_per_final_image[counter].size()<4) {
		small_t_solution_->reflect_point_raw(lines[r][0],
						     lines[r][1],
						     current_image);
		signs_vector[counter] = signs_vector[counter]*(-1.0);
		reflection_sequence.push_back(r);
	      }

	      all_images[counter] = BivariateImageWithTime(current_image,
							   1.0,
							   signs_vector[counter],
							   reflection_sequence);

	      counter = counter + 1;
	    }
	  }
	}
      }
    }
  }

  for (unsigned ii=0; ii<all_images.size(); ++ii) {
    std::cout << "image." << ii << ": ";
    for (unsigned reflection : all_images[ii].get_reflection_sequence()) {
      std::cout << reflection+1 << " ";
    }
    std::cout << std::endl;
  }

  printf("pdf(\"./src/kernel-expansion/documentation/chapter-3/chapter-3-figure-proof-2.pdf\", 6,6)\n");
  printf("plot(x=c(-32,5),y=c(-5,16),xlab=\"\",ylab=\"\",type=\"n\");\n");
  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
	 gsl_vector_get(lines[0][0], 0),
	 gsl_vector_get(lines[0][1], 0),
	 gsl_vector_get(lines[0][0], 1),
	 gsl_vector_get(lines[0][1], 1));
  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
	 gsl_vector_get(lines[1][0], 0),
	 gsl_vector_get(lines[1][1], 0),
	 gsl_vector_get(lines[1][0], 1),
	 gsl_vector_get(lines[1][1], 1));
  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
	 gsl_vector_get(lines[2][0], 0),
	 gsl_vector_get(lines[2][1], 0),
	 gsl_vector_get(lines[2][0], 1),
	 gsl_vector_get(lines[2][1], 1));
  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
	 gsl_vector_get(lines[3][0], 0),
	 gsl_vector_get(lines[3][1], 0),
	 gsl_vector_get(lines[3][0], 1),
	 gsl_vector_get(lines[3][1], 1));
  printf("points(x=%g, y=%g, pch=20, col=\"red\");\n ",
	 gsl_vector_get(&initial_condition_transformed_view.vector,0),
	 gsl_vector_get(&initial_condition_transformed_view.vector,1));
  printf("text(%g,%g-0.5,\"a)\");\n ",
	 gsl_vector_get(&lower_left_transformed_view.vector,0),
	 gsl_vector_get(&lower_left_transformed_view.vector,1));
  printf("text(%g + 0.5, %g,\"b)\");\n ",
	 gsl_vector_get(&lower_right_transformed_view.vector,0),
	 gsl_vector_get(&lower_right_transformed_view.vector,1));
  printf("text(%g, %g + 0.5,\"c)\");\n ",
	 gsl_vector_get(&upper_right_transformed_view.vector,0),
	 gsl_vector_get(&upper_right_transformed_view.vector,1));
  printf("text(%g - 0.5, %g,\"d)\");\n ",
	 gsl_vector_get(&upper_left_transformed_view.vector,0),
	 gsl_vector_get(&upper_left_transformed_view.vector,1));

  // half way between a) and b)
  printf("text(%g, %g-0.5,\"1\");\n ",
	 gsl_vector_get(&lower_left_transformed_view.vector,0) +
	 0.5*(gsl_vector_get(&lower_right_transformed_view.vector,0)-
	      gsl_vector_get(&lower_left_transformed_view.vector,0)),
	 gsl_vector_get(&lower_left_transformed_view.vector,1) +
	 0.5*(gsl_vector_get(&lower_right_transformed_view.vector,1)-
	      gsl_vector_get(&lower_left_transformed_view.vector,1)));

  // half way between b) and c)
  printf("text(%g, %g+0.5,\"2\");\n ",
	 gsl_vector_get(&lower_right_transformed_view.vector,0) +
	 0.5*(gsl_vector_get(&upper_right_transformed_view.vector,0)-
	      gsl_vector_get(&lower_right_transformed_view.vector,0)),
	 gsl_vector_get(&lower_right_transformed_view.vector,1) +
	 0.5*(gsl_vector_get(&upper_right_transformed_view.vector,1)-
	      gsl_vector_get(&lower_right_transformed_view.vector,1)));

  // half way between c) and d)
  printf("text(%g, %g+0.5,\"3\");\n ",
	 gsl_vector_get(&upper_right_transformed_view.vector,0) +
	 0.5*(gsl_vector_get(&upper_left_transformed_view.vector,0)-
	      gsl_vector_get(&upper_right_transformed_view.vector,0)),
	 gsl_vector_get(&upper_right_transformed_view.vector,1) +
	 0.5*(gsl_vector_get(&upper_left_transformed_view.vector,1)-
	      gsl_vector_get(&upper_right_transformed_view.vector,1)));

  // half way between d) and a)
  printf("text(%g, %g-0.5,\"4\");\n ",
	 gsl_vector_get(&upper_left_transformed_view.vector,0) +
	 0.5*(gsl_vector_get(&lower_left_transformed_view.vector,0)-
	      gsl_vector_get(&upper_left_transformed_view.vector,0)),
	 gsl_vector_get(&upper_left_transformed_view.vector,1) +
	 0.5*(gsl_vector_get(&lower_left_transformed_view.vector,1)-
	      gsl_vector_get(&upper_left_transformed_view.vector,1)));

  // SLOPE AND INTERCEPT OF LINE PERPENDICULAR TO LINES 2 AND 4
  // slope
  const BivariateImageWithTime& IC = all_images[0];
  const BivariateImageWithTime& IC_accross_2 = all_images[8];
  double delta_x =
    gsl_vector_get(IC_accross_2.get_position(),0)-
    gsl_vector_get(IC.get_position(),0);
  double delta_y =
    gsl_vector_get(IC_accross_2.get_position(),1)-
    gsl_vector_get(IC.get_position(),1);
  double slope = delta_y/delta_x;
  // intercept
  // y = mx + b => b = y-mx
  double intercept = gsl_vector_get(IC.get_position(),1) -
    slope*gsl_vector_get(IC.get_position(),0);
  // LINE OF REFLECTIONS ABOUT 2 AND 4
  printf("abline(a=%g, b=%g, lty=2);\n ",
	 intercept, slope);

  // IMAGES ALONG DASHED LINE
  for (unsigned ii : std::vector<unsigned> {0, 4,8,12,24,28}) {
    const BivariateImageWithTime& image = all_images[ii];
    if (std::signbit(image.get_mult_factor())) {
      printf("points(x=%g,y=%g,pch=20, col=\"black\");\n ",
	     gsl_vector_get(image.get_position(), 0),
	     gsl_vector_get(image.get_position(), 1));
    } else {
      printf("points(x=%g,y=%g,pch=20, col=\"red\");\n ",
	     gsl_vector_get(image.get_position(), 0),
	     gsl_vector_get(image.get_position(), 1));
    }
  }

  // SLOPE AND INTERCEPT OF LINE PARALLEL TO 1
  // slope
  delta_x =
    gsl_vector_get(lines[0][0],0)-
    gsl_vector_get(lines[0][1],0);
  delta_y =
    gsl_vector_get(lines[0][0],1)-
    gsl_vector_get(lines[0][1],1);
  slope = delta_y/delta_x;
  // intercept => b = y-mx
  intercept = gsl_vector_get(lines[0][1],1) -
    slope*gsl_vector_get(lines[0][1],0);
  printf("abline(a=%g, b=%g, lty=3);\n ",
	 intercept, slope);

  // SLOPE AND INTERCEPT OF LINE PARALLEL TO 3
  // slope
  delta_x =
    gsl_vector_get(lines[2][0],0)-
    gsl_vector_get(lines[2][1],0);
  delta_y =
    gsl_vector_get(lines[2][0],1)-
    gsl_vector_get(lines[2][1],1);
  slope = delta_y/delta_x;
  // intercept => b = y-mx
  intercept = gsl_vector_get(lines[2][1],1) -
    slope*gsl_vector_get(lines[2][1],0);
  printf("abline(a=%g, b=%g, lty=3);\n ",
	 intercept, slope);

  // INTERCEPT OF LINE 3 AND REFLECTION SEGMENT
  double c = std::sqrt(2)/2;

  // through intercept
  double x_intercept =
    (-2.0*x_0_*rho_ + 2*y_0_/sigma_y_ - 2/sigma_y_*(1-rho_))*c/
    (-2*rho_*std::sqrt(1-rho_));

  double y_intercept =

   std::sqrt(1 - rho_)/std::sqrt(1 + rho_)*x_intercept +
    2*c/(sigma_y_*std::sqrt(1+rho_));

  printf("points(x=%g, y=%g, col=\"blue\", pch=20);\n ",
	 x_intercept,
	 y_intercept);

  printf("dev.off();\n ");
}


void BivariateSolver::figure_chapter_3_illustration_1() const
{

  std::vector<BivariateImageWithTime> all_images =
    small_t_image_positions_type_41_all(false);
  printf("pdf(\"./src/kernel-expansion/documentation/chapter-3/chapter-3-figure-illustration-1.pdf\", 6,6)\n");
  printf("plot(x=c(-4,4),y=c(-3,5),xlab=\"\",ylab=\"\",type=\"n\");\n");
  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
	 0.0, 1.0, 0.0, 0.0);
  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
	 1.0, 1.0, 0.0, 1.0);
  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
	 1.0, 0.0, 1.0, 1.0);
  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
	 0.0, 0.0, 1.0, 0.0);

  for (unsigned ii=0; ii<all_images.size(); ++ii) {
    std::vector<unsigned> reflection_sequence_per_final_image_current =
      all_images[ii].get_reflection_sequence();

    if (((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 1) &&
	 (reflection_sequence_per_final_image_current[1] == 3) &&
	 (reflection_sequence_per_final_image_current[2] == 0) &&
	 (reflection_sequence_per_final_image_current[3] == 2)) |
	((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 1) &&
	 (reflection_sequence_per_final_image_current[1] == 3) &&
	 (reflection_sequence_per_final_image_current[2] == 2) &&
	 (reflection_sequence_per_final_image_current[3] == 0)) |
	((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 3) &&
	 (reflection_sequence_per_final_image_current[1] == 1) &&
	 (reflection_sequence_per_final_image_current[2] == 0) &&
	 (reflection_sequence_per_final_image_current[3] == 2)) |
	((reflection_sequence_per_final_image_current.size() == 4) &&
	 (reflection_sequence_per_final_image_current[0] == 3) &&
	 (reflection_sequence_per_final_image_current[1] == 1) &&
	 (reflection_sequence_per_final_image_current[2] == 2) &&
	 (reflection_sequence_per_final_image_current[3] == 0)))
      {
	printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## ",
	       ii,
	       gsl_vector_get(all_images[ii].get_position(),0),
	       gsl_vector_get(all_images[ii].get_position(),1),
	       ii,
	       all_images[ii].get_mult_factor(),
	       gsl_vector_get(all_images[ii].get_position(),0),
	       gsl_vector_get(all_images[ii].get_position(),1));
      }  else if (reflection_sequence_per_final_image_current.size() > 4) {
      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
	     ii,
	     gsl_vector_get(all_images[ii].get_position(),0),
	     gsl_vector_get(all_images[ii].get_position(),1),
	     ii,
	     all_images[ii].get_mult_factor(),
	     gsl_vector_get(all_images[ii].get_position(),0),
	     gsl_vector_get(all_images[ii].get_position(),1));
    } else if (reflection_sequence_per_final_image_current.size() == 0) {
      	printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"red\"); ## ",
	       ii,
	       gsl_vector_get(all_images[ii].get_position(),0),
	       gsl_vector_get(all_images[ii].get_position(),1),
	       ii,
	       all_images[ii].get_mult_factor(),
	       gsl_vector_get(all_images[ii].get_position(),0),
	       gsl_vector_get(all_images[ii].get_position(),1));
    } else {
	printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
	       ii,
	       gsl_vector_get(all_images[ii].get_position(),0),
	       gsl_vector_get(all_images[ii].get_position(),1),
	       ii,
	       all_images[ii].get_mult_factor(),
	       gsl_vector_get(all_images[ii].get_position(),0),
	       gsl_vector_get(all_images[ii].get_position(),1));
      }

    for (const unsigned& refl : reflection_sequence_per_final_image_current) {
      printf("%i ", refl);
    }

    printf("\n");
  }
  printf("dev.off();\n ");
}

double BivariateSolver::wrapper(const std::vector<double> &x,
				std::vector<double> &grad,
				void * data)
{
  std::cout << "trying t=" << x[0] << " ";

  double t = x[0];

  BivariateSolver * solver = reinterpret_cast<BivariateSolver*>(data);
  double raw_input [2] = {solver->get_x_t_2(),
			  solver->get_y_t_2()};
  gsl_vector_view raw_input_view = gsl_vector_view_array(raw_input, 2);

  double sigma_y_current = solver->get_sigma_y();
  double rho_current = solver->get_rho();
  double t_2_current = solver->get_t_2();

  double x_0_2_current = solver->get_x_0_2();
  double y_0_2_current = solver->get_y_0_2();

  solver->set_diffusion_parameters_and_data_small_t(1.0,
						    sigma_y_current,
						    rho_current,
						    t,
						    0.0,
						    x_0_2_current,
						    1.0,
						    0.0,
						    y_0_2_current,
						    1.0);

  double out = solver->analytic_likelihood(&raw_input_view.vector,1000);

  solver->set_diffusion_parameters_and_data_small_t(1.0,
						    sigma_y_current,
						    rho_current,
						    t_2_current,
						    0.0,
						    x_0_2_current,
						    1.0,
						    0.0,
						    y_0_2_current,
						    1.0);
  std::cout << "out=" << out << std::endl;
  return out;
}

double BivariateSolver::wrapper_small_t(const std::vector<double> &x,
					std::vector<double> &grad,
					void * data)
{
  std::cout << "trying t=" << x[0] << " ";

  double t = x[0];

  BivariateSolver * solver = reinterpret_cast<BivariateSolver*>(data);
  double raw_input [2] = {solver->get_x_t_2(),
			  solver->get_y_t_2()};
  gsl_vector_view raw_input_view = gsl_vector_view_array(raw_input, 2);

  double sigma_y_current = solver->get_sigma_y();
  double rho_current = solver->get_rho();
  double t_2_current = solver->get_t_2();

  double x_0_2_current = solver->get_x_0_2();
  double y_0_2_current = solver->get_y_0_2();

  solver->set_diffusion_parameters_and_data_small_t(1.0,
						    sigma_y_current,
						    rho_current,
						    t,
						    0.0,
						    x_0_2_current,
						    1.0,
						    0.0,
						    y_0_2_current,
						    1.0);

  double out =
    solver->numerical_likelihood_first_order_small_t_ax_bx(&raw_input_view.vector,
							   1e-5);

  solver->set_diffusion_parameters_and_data_small_t(1.0,
						    sigma_y_current,
						    rho_current,
						    t_2_current,
						    0.0,
						    x_0_2_current,
						    1.0,
						    0.0,
						    y_0_2_current,
						    1.0);
  std::cout << "out=" << out << std::endl;
  return out;
}
