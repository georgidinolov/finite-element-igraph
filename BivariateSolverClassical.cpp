#include <algorithm>
#include "BasisElementTypes.hpp"
#include <chrono>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_randist.h>
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

  double ss_1 = atan(-1.0*sqrt(1.0+rho_)/sqrt(1-rho_));

  double C_1 = (-1.0*eta_ic + xi_ic*sqrt(1.0-rho_)/sqrt(1.0+rho_))/
    (sin(ss_1) - cos(ss_1)*sqrt(1.0-rho_)/sqrt(1.0+rho_));


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

  // std::cout << "ss_1 = " << ss_1 << "; C_1 = " << C_1 << std::endl;
  // std::cout << "ss_2 = " << ss_2 << "; C_2 = " << C_2 << std::endl;
  // std::cout << "ss_3 = " << ss_3 << "; C_3 = " << C_3 << std::endl;
  // std::cout << "ss_4 = " << ss_4 << "; C_4 = " << C_4 << std::endl;

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

  tt_ = std::pow(Cs[Cs_indeces[1]]/5.0, 2.0);
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

BivariateSolverClassical::BivariateSolverClassical(const BivariateSolverClassical& solver)
  : sigma_x_(solver.sigma_x_),
    sigma_y_(solver.sigma_y_),
    rho_(solver.rho_),
    x_0_(solver.x_0_),
    y_0_(solver.y_0_),
    mvtnorm_(MultivariateNormal()),
    xi_eta_input_(gsl_vector_alloc(solver.xi_eta_input_->size)),
    initial_condition_xi_eta_(gsl_vector_alloc(solver.initial_condition_xi_eta_->size)),
    Rotation_matrix_(gsl_matrix_alloc(solver.Rotation_matrix_->size1,
				      solver.Rotation_matrix_->size2)),
    tt_(solver.tt_),
    Variance_(gsl_matrix_alloc(solver.Variance_->size1,
			       solver.Variance_->size2)),
    initial_condition_xi_eta_reflected_(gsl_vector_alloc(solver.
							 initial_condition_xi_eta_reflected_->size)),
    function_grid_(gsl_matrix_alloc(solver.function_grid_->size1,
				    solver.function_grid_->size2))
{
  gsl_vector_memcpy(xi_eta_input_,
		    solver.xi_eta_input_);

  gsl_vector_memcpy(initial_condition_xi_eta_,
		    solver.initial_condition_xi_eta_);

  gsl_matrix_memcpy(Rotation_matrix_,
		    solver.Rotation_matrix_);

  gsl_matrix_memcpy(Variance_,
		    solver.Variance_);

  gsl_vector_memcpy(initial_condition_xi_eta_reflected_,
		    solver.initial_condition_xi_eta_reflected_);

  gsl_matrix_memcpy(function_grid_,
		    solver.function_grid_);
}

BivariateSolverClassical& BivariateSolverClassical::
operator=(const BivariateSolverClassical& rhs)
{
  sigma_x_ = rhs.sigma_x_;
  sigma_y_ = rhs.sigma_y_;
  rho_ = rhs.rho_;
  x_0_ = rhs.x_0_;
  y_0_ = rhs.y_0_;
  mvtnorm_ = MultivariateNormal();

  gsl_vector_free(xi_eta_input_);
  xi_eta_input_ = gsl_vector_alloc(rhs.xi_eta_input_->size);
  gsl_vector_memcpy(xi_eta_input_, rhs.xi_eta_input_);

  gsl_vector_free(initial_condition_xi_eta_);
  initial_condition_xi_eta_ = gsl_vector_alloc(rhs.initial_condition_xi_eta_->size);
  gsl_vector_memcpy(initial_condition_xi_eta_,
		    rhs.initial_condition_xi_eta_);

  gsl_matrix_free(Rotation_matrix_);
  Rotation_matrix_ = gsl_matrix_alloc(rhs.Rotation_matrix_->size1,
				      rhs.Rotation_matrix_->size2);
  gsl_matrix_memcpy(Rotation_matrix_, rhs.Rotation_matrix_);

  tt_ = rhs.tt_;

  gsl_matrix_free(Variance_);
  Variance_ = gsl_matrix_alloc(rhs.Variance_->size1,
			       rhs.Variance_->size2);
  gsl_matrix_memcpy(Variance_, rhs.Variance_);

  gsl_vector_free(initial_condition_xi_eta_reflected_);
  initial_condition_xi_eta_reflected_ =
    gsl_vector_alloc(rhs.initial_condition_xi_eta_reflected_->size);
  gsl_vector_memcpy(initial_condition_xi_eta_reflected_,
		    rhs.initial_condition_xi_eta_reflected_);

  gsl_matrix_free(function_grid_);
  function_grid_ = gsl_matrix_alloc(rhs.function_grid_->size1,
				    rhs.function_grid_->size2);
  gsl_matrix_memcpy(function_grid_, rhs.function_grid_);

  return *this;
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

  double out =
    (gsl_ran_gaussian_pdf(gsl_vector_get(xi_eta_input_,0)-
			gsl_vector_get(initial_condition_xi_eta_,0),
			sqrt(gsl_matrix_get(Variance_, 0,0))) *
    gsl_ran_gaussian_pdf(gsl_vector_get(xi_eta_input_,1)-
			gsl_vector_get(initial_condition_xi_eta_,1),
			 sqrt(gsl_matrix_get(Variance_, 1,1))) -
    // / 
    gsl_ran_gaussian_pdf(gsl_vector_get(xi_eta_input_,0)-
  			gsl_vector_get(initial_condition_xi_eta_reflected_,0),
  			sqrt(gsl_matrix_get(Variance_, 0,0))) *
    gsl_ran_gaussian_pdf(gsl_vector_get(xi_eta_input_,1)-
  			gsl_vector_get(initial_condition_xi_eta_reflected_,1),
			 sqrt(gsl_matrix_get(Variance_, 1,1)))) /
    (sigma_x_*sigma_y_*sqrt(1-rho_)*sqrt(1+rho_));

  return out;
}

double BivariateSolverClassical::
operator()(const gsl_vector* input, double tt) const
{
  gsl_blas_dgemv(CblasNoTrans, 1.0,
		 Rotation_matrix_, input, 0.0,
		 xi_eta_input_);

  // gsl_matrix* Variance = gsl_matrix_alloc(2,2);
  // gsl_matrix_set_all(Variance, 0.0);

  // for (int i=0; i<2; ++i) {
  //   gsl_matrix_set(Variance, i, i, tt);
  // }

  double sd = std::sqrt(tt);
  double out = (gsl_ran_gaussian_pdf(gsl_vector_get(xi_eta_input_,0)-
				     gsl_vector_get(initial_condition_xi_eta_,0),
				     sd)*
		gsl_ran_gaussian_pdf(gsl_vector_get(xi_eta_input_,1)-
				     gsl_vector_get(initial_condition_xi_eta_,1),
				     sd) -
		gsl_ran_gaussian_pdf(gsl_vector_get(xi_eta_input_,0)-
				     gsl_vector_get(initial_condition_xi_eta_reflected_,0),
				     sd)*
		gsl_ran_gaussian_pdf(gsl_vector_get(xi_eta_input_,1)-
				     gsl_vector_get(initial_condition_xi_eta_reflected_,1),
				     sd)) /
   		(sigma_x_*sigma_y_*sqrt(1-rho_)*sqrt(1+rho_));

  // mvtnorm_.dmvnorm(2,
  //   				 xi_eta_input_,
  //   				 initial_condition_xi_eta_,
  //   				 Variance) -
  //   		mvtnorm_.dmvnorm(2,
  //   				 xi_eta_input_,
  //   				 initial_condition_xi_eta_reflected_,
  //   				 Variance)) /
  //   		(sigma_x_*sigma_y_*sqrt(1-rho_)*sqrt(1+rho_));
  // double out = mvtnorm_.dmvnorm(2,
  // 				xi_eta_input_,
  // 				initial_condition_xi_eta_,
  // 				Variance) /
  //   (sigma_x_*sigma_y_*sqrt(1-rho_)*sqrt(1+rho_));

  // gsl_matrix_free(Variance);
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

double BivariateSolverClassical::
distance_from_point_to_axis_raw(const gsl_vector* point_1,
				const gsl_vector* point_2,
				const gsl_vector* normal_point,
				const gsl_vector* input) const
{
  gsl_vector* axis_vector = gsl_vector_alloc(2);
  gsl_vector* input_cpy = gsl_vector_alloc(2);
  gsl_vector* normal_point_cpy = gsl_vector_alloc(2);

  gsl_vector_memcpy(input_cpy, input);
  gsl_vector_memcpy(axis_vector, point_2);
  gsl_vector_memcpy(normal_point_cpy, normal_point);

  // re-centering on zero
  gsl_vector_sub(axis_vector, point_1);
  gsl_vector_sub(input_cpy, point_1);
  gsl_vector_sub(normal_point_cpy, point_1);

  double out = distance_from_point_to_axis(axis_vector,
					   normal_point_cpy,
					   input_cpy);

  gsl_vector_free(axis_vector);
  gsl_vector_free(input_cpy);
  gsl_vector_free(normal_point_cpy);

  return out;
}

void BivariateSolverClassical::reflect_point_raw(const gsl_vector* point_1,
						 const gsl_vector* point_2,
						 gsl_vector* input) const
{
  gsl_vector* axis_vector = gsl_vector_alloc(2);
  gsl_vector* input_cpy = gsl_vector_alloc(2);
  gsl_vector_memcpy(input_cpy, input);
  gsl_vector_memcpy(axis_vector, point_2);
  // re-centering on zero
  gsl_vector_sub(axis_vector, point_1);
  gsl_vector_sub(input_cpy, point_1);

  reflect_point(axis_vector, input_cpy);

  gsl_vector_add(input_cpy, point_1);
  gsl_vector_memcpy(input, input_cpy);

  gsl_vector_free(axis_vector);
  gsl_vector_free(input_cpy);
}

// both axis_vector and input are centered on (0,0)
double BivariateSolverClassical::
distance_from_point_to_axis(const gsl_vector* axis_vector,
			    const gsl_vector* normal_point,
			    const gsl_vector* input) const
{
  double input_cpy [input->size];
  gsl_vector_view input_cpy_view = gsl_vector_view_array(input_cpy,input->size);
  gsl_vector_memcpy(&input_cpy_view.vector, input);

  double axis_vector_array [axis_vector->size];
  gsl_vector_view axis_vector_view =
    gsl_vector_view_array(axis_vector_array, axis_vector->size);
  gsl_vector_memcpy(&axis_vector_view.vector, axis_vector);

  double normal_point_array [normal_point->size];
  gsl_vector_view normal_point_view =
    gsl_vector_view_array(normal_point_array, normal_point->size);
  gsl_vector_memcpy(&normal_point_view.vector, normal_point);

  // scaling the axis vector to be unit length
  double norm = 0.0;
  for (unsigned i=0; i<axis_vector_view.vector.size; ++i) {
    norm = norm +
      std::pow(axis_vector_array[i],2);
  }
  norm = std::sqrt(norm);
  gsl_vector_scale(&axis_vector_view.vector, 1.0/norm);


  double inner_product = 0;
  gsl_blas_ddot(&input_cpy_view.vector,
		&axis_vector_view.vector,
		&inner_product);
  gsl_vector_scale(&axis_vector_view.vector, inner_product);

  // normalized_axis <normalized_axis | input> + Delta = input
  // \Rightarrow Delta = input - normalized_axis <normalized_axis | input>
  gsl_vector_sub(&input_cpy_view.vector, &axis_vector_view.vector);

  // \Rightarrow Delta_normal_point = normal_point - normalized_axis<normalized_axis | normal_point>
  gsl_blas_ddot(&axis_vector_view.vector, &axis_vector_view.vector, &norm);
  norm = std::sqrt(norm);
  gsl_vector_scale(&axis_vector_view.vector, 1.0/norm);
  gsl_blas_ddot(&axis_vector_view.vector, &normal_point_view.vector, &inner_product);
  gsl_vector_scale(&axis_vector_view.vector, inner_product);
  gsl_vector_sub(&normal_point_view.vector, &axis_vector_view.vector);

  double out = 0;
  for (unsigned i=0; i<input_cpy_view.vector.size; ++i) {
    out = out + std::pow(input_cpy[i],2);
  }
  out = std::sqrt(out);

  double sign_inner_prod = 0;
  gsl_blas_ddot(&input_cpy_view.vector, &normal_point_view.vector, &sign_inner_prod);

  if (std::signbit(sign_inner_prod)) {
    out = -out;
  }

  return out;
}

// both axis_vector and input are centered on (0,0)
void BivariateSolverClassical::reflect_point(const gsl_vector* axis_vector,
					     gsl_vector* input) const
{
  double axis_vector_array [axis_vector->size];
  gsl_vector_view axis_vector_view = gsl_vector_view_array(axis_vector_array, axis_vector->size);
  gsl_vector_memcpy(&axis_vector_view.vector, axis_vector);
  //
  double norm = 0.0;
  for (unsigned i=0; i<axis_vector_view.vector.size; ++i) {
    norm = norm +
      std::pow(axis_vector_array[i],2);
  }
  norm = std::sqrt(norm);
  gsl_vector_scale(&axis_vector_view.vector, 1.0/norm);

  double inner_product = 0;
  gsl_blas_ddot(&axis_vector_view.vector, input, &inner_product);

  gsl_vector_scale(&axis_vector_view.vector, inner_product);

  // normalized_axis <normalized_axis | input> + Delta = input
  // \Rightarrow Delta = input - normalized_axis <normalized_axis | input>
  double Delta [axis_vector->size];
  gsl_vector_view Delta_view = gsl_vector_view_array(Delta, axis_vector->size);
  gsl_vector_memcpy(&Delta_view.vector, input);
  gsl_vector_sub(&Delta_view.vector, &axis_vector_view.vector);
  gsl_vector_scale(&Delta_view.vector, -1.0);

  gsl_vector_add(&Delta_view.vector, &axis_vector_view.vector);
  gsl_vector_memcpy(input, &Delta_view.vector);
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

  // auto t1 = std::chrono::high_resolution_clock::now();
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
  // auto t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "duration in Bivariate Classical Solver = "
  // 	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  //   	    << " milliseconds\n";
  gsl_vector_free(input);
}
