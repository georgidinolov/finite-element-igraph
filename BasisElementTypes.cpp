#include <algorithm>
#include <chrono>
#include "BasisElementTypes.hpp"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <iostream>

// =================== BASIS ELEMENT CLASS ===================
BasisElement::~BasisElement()
{}

// ============= BIVARIATE ELEMENT INTERFACE CLASS ===========

// ============== GAUSSIAN KERNEL ELEMENT =====================
GaussianKernelElement::GaussianKernelElement()
  : dx_(1.0),
    dimension_(1),
    exponent_power_(1.0),
    mean_vector_(gsl_vector_alloc(dimension_)),
    input_gsl_(gsl_vector_alloc(dimension_)),
    covariance_matrix_(gsl_matrix_alloc(dimension_,
					dimension_)),
    mvtnorm_(MultivariateNormal()),
    s_(0),
    ax_(0.0),
    ym_(gsl_vector_alloc(dimension_)),
    work_(gsl_matrix_alloc(dimension_,dimension_)),
    winv_(gsl_matrix_alloc(dimension_,dimension_)),
    p_(gsl_permutation_alloc(dimension_))
{}

GaussianKernelElement::
GaussianKernelElement(double dx,
		      long unsigned dimension,
		      double exponent_power,
		      const gsl_vector* mean_vector,
		      const gsl_matrix* covariance_matrix)
  : dx_(dx),
    dimension_(dimension),
    exponent_power_(exponent_power),
    mean_vector_(gsl_vector_alloc(dimension_)),
    input_gsl_(gsl_vector_alloc(dimension_)),
    covariance_matrix_(gsl_matrix_alloc(dimension_,
					dimension_)),
    mvtnorm_(MultivariateNormal()),
    s_(0),
    ax_(0.0),
    ym_(gsl_vector_alloc(dimension_)),
    work_(gsl_matrix_alloc(dimension_,dimension_)),
    winv_(gsl_matrix_alloc(dimension_,dimension_)),
    p_(gsl_permutation_alloc(dimension_))
{
  gsl_vector_memcpy(mean_vector_, mean_vector);
  gsl_matrix_memcpy(covariance_matrix_, covariance_matrix);

  gsl_matrix_memcpy( work_, get_covariance_matrix() );
  gsl_linalg_LU_decomp( work_, p_, &s_ );
  gsl_linalg_LU_invert( work_, p_, winv_ );
  ax_ = gsl_linalg_LU_det( work_, s_ );
}

GaussianKernelElement::
GaussianKernelElement(const GaussianKernelElement& element)
  : dx_(element.get_dx()),
    dimension_(element.get_dimension()),
    exponent_power_(element.get_exponent_power()),
    mean_vector_(gsl_vector_alloc(dimension_)),
    input_gsl_(gsl_vector_alloc(dimension_)),
    covariance_matrix_(gsl_matrix_alloc(dimension_,
					dimension_)),
    mvtnorm_(MultivariateNormal()),
    s_(0),
    ax_(0.0),
    ym_(gsl_vector_alloc(dimension_)),
    work_(gsl_matrix_alloc(dimension_,dimension_)),
    winv_(gsl_matrix_alloc(dimension_,dimension_)),
    p_(gsl_permutation_alloc(dimension_))
{
  if (element.dimension_ != element.mean_vector_->size) {
    std::cout << "ERROR: Dimensions do not match" << std::endl;
  }

  gsl_vector_memcpy(mean_vector_, element.get_mean_vector());
  gsl_matrix_memcpy(covariance_matrix_, element.get_covariance_matrix());
  
  gsl_matrix_memcpy( work_, get_covariance_matrix() );
  gsl_linalg_LU_decomp( work_, p_, &s_ );
  gsl_linalg_LU_invert( work_, p_, winv_ );
  ax_ = gsl_linalg_LU_det( work_, s_ );
  std::cout << "DONE with copy constructor for GaussianKernelElement" << std::endl;
}

GaussianKernelElement::~GaussianKernelElement()
{
  gsl_vector_free(mean_vector_);
  gsl_matrix_free(covariance_matrix_);
  gsl_vector_free(input_gsl_);

  gsl_matrix_free(work_);
  gsl_permutation_free(p_);
  gsl_matrix_free(winv_);
  gsl_vector_free(ym_);
}

double GaussianKernelElement::
operator()(const gsl_vector* input) const
{
  if (input->size == dimension_) {
    double mollifier = 1;
    double ay = 0.0;

    gsl_vector_memcpy(input_gsl_, input);
    for (unsigned i=0; i<dimension_; ++i) {
      mollifier = mollifier *
	pow(gsl_vector_get(input, i), get_exponent_power()) *
	pow((1-gsl_vector_get(input, i)), get_exponent_power());
    }

    gsl_vector_sub(input_gsl_, get_mean_vector());
    gsl_blas_dsymv(CblasUpper,1.0,get_winv(),input_gsl_,0.0,ym_);
    gsl_blas_ddot(input_gsl_, ym_, &ay);
    ay = exp(-0.5*ay)/sqrt( pow((2*M_PI),2)*get_ax());
    
    return ay*mollifier;
  } else {
    std::cout << "INPUT SIZE WRONG" << std::endl;
    return 0;
  }
}

GaussianKernelElement& GaussianKernelElement::
operator=(const GaussianKernelElement& rhs)
{
  dx_ = rhs.get_dx();
  dimension_ = rhs.get_dimension();
  exponent_power_ = rhs.get_exponent_power();

  gsl_vector_free(mean_vector_);
  mean_vector_ = gsl_vector_alloc(rhs.get_dimension());
  gsl_vector_memcpy(mean_vector_, rhs.get_mean_vector());

  gsl_vector_free(input_gsl_);
  input_gsl_ = gsl_vector_alloc(rhs.get_dimension());
  gsl_vector_memcpy(input_gsl_, rhs.get_mean_vector());

  gsl_matrix_free(covariance_matrix_);
  covariance_matrix_ = gsl_matrix_alloc(rhs.get_dimension(),
					rhs.get_dimension());
  gsl_matrix_memcpy(covariance_matrix_, rhs.get_covariance_matrix());
  
  mvtnorm_= MultivariateNormal();
  s_ = 0;
  ax_ = 0.0;

  gsl_matrix_free(work_);
  gsl_permutation_free(p_);
  gsl_matrix_free(winv_);
  gsl_vector_free(ym_);
  
  ym_ = gsl_vector_alloc(dimension_);
  work_ = gsl_matrix_alloc(dimension_,dimension_);
  winv_ = gsl_matrix_alloc(dimension_,dimension_);
  p_ = gsl_permutation_alloc(dimension_);

  gsl_matrix_memcpy( work_, get_covariance_matrix() );
  gsl_linalg_LU_decomp( work_, p_, &s_ );
  gsl_linalg_LU_invert( work_, p_, winv_ );
  ax_ = gsl_linalg_LU_det( work_, s_ );

  return *this;
}


double GaussianKernelElement::
first_derivative(const gsl_vector* input,
		 long int coord_index) const
{
  return first_derivative_finite_diff(input,
				      coord_index);
}

double GaussianKernelElement::
first_derivative_finite_diff(const gsl_vector* input,
			     long int coord_index) const
{
  gsl_vector* input_plus = gsl_vector_alloc(dimension_);
  gsl_vector* input_minus = gsl_vector_alloc(dimension_);;

  gsl_vector_memcpy(input_plus, input);
  gsl_vector_memcpy(input_minus, input);
  
  gsl_vector_set(input_plus, coord_index,
  		    gsl_vector_get(input, coord_index)+dx_);
  gsl_vector_set(input_minus, coord_index,
  		    gsl_vector_get(input, coord_index)-dx_);

  double out = ((*this)(input_plus) - (*this)(input_minus))/
    (2*dx_);

  gsl_vector_free(input_minus);
  gsl_vector_free(input_plus);
  
  return out;
}

double GaussianKernelElement::norm_finite_diff() const
{
  long int N = 1.0/dx_;

  double integral = 0;
  gsl_vector* input = gsl_vector_alloc(dimension_);
  double x = 0;
  double y = 0;

  for (long int i=0; i<N; ++i) {
    for (long int j=0; j<N; ++j) {
      x = i*dx_;
      y = j*dx_;
      gsl_vector_set(input, 0, x);
      gsl_vector_set(input, 1, y);

      integral = integral + std::pow((*this)(input), 2);
    }
  }
  integral = integral * std::pow(dx_, dimension_);
  
  gsl_vector_free(input);
  return std::sqrt(integral);
}

double GaussianKernelElement::norm() const
{
  return norm_finite_diff();
}

const gsl_vector* GaussianKernelElement::get_mean_vector() const
{
  return mean_vector_;
}

const gsl_matrix* GaussianKernelElement::get_covariance_matrix() const
{
  return covariance_matrix_;
}

// void GaussianKernelElement::set_norm()
// {
//   norm_ = norm_finite_diff();
// }

// void GaussianKernelElement::set_gsl_objects()
// {
//     covariance_matrix_gsl_ = gsl_matrix_alloc(dimension_,
// 					      dimension_);
//     mean_vector_gsl_ = gsl_vector_alloc(dimension_);
//     input_gsl_ = gsl_vector_alloc(dimension_);
    
//     for (unsigned i=0; i<dimension_; ++i) {
//       gsl_vector_set(mean_vector_gsl_, i, gsl_vector_e(&mean_vector_, i));
//       gsl_vector_set(input_gsl_, i, gsl_vector_e(&mean_vector_, i));
      
//       gsl_matrix_set(covariance_matrix_gsl_, i, i,
// 		     gsl_matrix_e(&covariance_matrix_,
// 				     i, i));
//       for (unsigned j=i+1; j<dimension_; ++j) {
// 	gsl_matrix_set(covariance_matrix_gsl_, i, j,
// 		       gsl_matrix_e(&covariance_matrix_,
// 				       i, j));
// 	gsl_matrix_set(covariance_matrix_gsl_, j, i,
// 		       gsl_matrix_e(&covariance_matrix_,
// 				       j, i));
//       }
//     }
// }


// ============== BIVARIATE GAUSSIAN KERNEL ELEMENT ================
BivariateGaussianKernelElement::BivariateGaussianKernelElement()
  : GaussianKernelElement(),
    function_grid_(gsl_matrix_alloc(1/get_dx(), 1/get_dx())),
    deriv_function_grid_dx_(gsl_matrix_alloc(1/get_dx(), 1/get_dx())),
    deriv_function_grid_dy_(gsl_matrix_alloc(1/get_dx(), 1/get_dx()))
{}

BivariateGaussianKernelElement::
BivariateGaussianKernelElement(double dx,
			       double exponent_power,
			       const gsl_vector* mean_vector,
			       const gsl_matrix* covariance_matrix)
  : GaussianKernelElement(dx,
			  2,
			  exponent_power,
			  mean_vector,
			  covariance_matrix),
    function_grid_(gsl_matrix_alloc(1/dx, 1/dx)),
    deriv_function_grid_dx_(gsl_matrix_alloc(1/dx, 1/dx)),
    deriv_function_grid_dy_(gsl_matrix_alloc(1/dx, 1/dx))
{
  // set_function_grid();
  set_function_grids();
}

BivariateGaussianKernelElement::
BivariateGaussianKernelElement(const BivariateGaussianKernelElement& element)
  : GaussianKernelElement(element)
{
  function_grid_ = gsl_matrix_alloc(1/element.get_dx(), 1/element.get_dx());
  deriv_function_grid_dx_ = gsl_matrix_alloc(1/element.get_dx(), 1/element.get_dx());
  deriv_function_grid_dy_ = gsl_matrix_alloc(1/element.get_dx(), 1/element.get_dx());
  
  gsl_matrix_memcpy(function_grid_, element.function_grid_);
  gsl_matrix_memcpy(deriv_function_grid_dx_, element.deriv_function_grid_dx_);
  gsl_matrix_memcpy(deriv_function_grid_dy_, element.deriv_function_grid_dy_);
}

BivariateGaussianKernelElement::
~BivariateGaussianKernelElement()
{
  gsl_matrix_free(function_grid_);
  gsl_matrix_free(deriv_function_grid_dx_);
  gsl_matrix_free(deriv_function_grid_dy_);
}

BivariateGaussianKernelElement& BivariateGaussianKernelElement::
operator=(const BivariateGaussianKernelElement& rhs)
{
  GaussianKernelElement::operator=(rhs);

  gsl_matrix_free(function_grid_);
  gsl_matrix_free(deriv_function_grid_dx_);
  gsl_matrix_free(deriv_function_grid_dy_);
  
  function_grid_ = gsl_matrix_alloc(1/rhs.get_dx(), 1/rhs.get_dx());
  deriv_function_grid_dx_ = gsl_matrix_alloc(1/rhs.get_dx(), 1/rhs.get_dx());
  deriv_function_grid_dy_ = gsl_matrix_alloc(1/rhs.get_dx(), 1/rhs.get_dx());
  
  gsl_matrix_memcpy(function_grid_, rhs.function_grid_);
  gsl_matrix_memcpy(deriv_function_grid_dx_, rhs.deriv_function_grid_dx_);
  gsl_matrix_memcpy(deriv_function_grid_dy_, rhs.deriv_function_grid_dy_);  

  return *this;
}

void BivariateGaussianKernelElement::set_function_grid()
{
  auto t1 = std::chrono::high_resolution_clock::now();
  double dx = get_dx();
  double out = 0;
  gsl_vector * input = gsl_vector_alloc(2);
  double x = 0;
  double y = 0;

  for (int i=0; i<1/dx; ++i) {
    x = i*dx;
    gsl_vector_set(input, 0, x);

    for (int j=0; j<1/dx; ++j) {
      y = j*dx;
      gsl_vector_set(input, 1, y);

      out = (*this)(input);
      gsl_matrix_set(function_grid_, i, j, out);
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "duration = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
    	    << " milliseconds\n";

  gsl_vector_free(input);
}

void BivariateGaussianKernelElement::set_function_grid_dx()
{
  auto t1 = std::chrono::high_resolution_clock::now();
  
  double dx = get_dx();
  double out = 0;
  gsl_vector * input = gsl_vector_alloc(2);
  double x = 0;
  double y = 0;

  for (int i=0; i<1/dx; ++i) {
    x = i*dx;
    gsl_vector_set(input, 0, x);

    for (int j=0; j<1/dx; ++j) {
      y = j*dx;
      gsl_vector_set(input, 1, y);

      out = first_derivative(input, 0);
      gsl_matrix_set(deriv_function_grid_dx_, i, j, out);
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "duration = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
    	    << " milliseconds\n";

  gsl_vector_free(input);
}

void BivariateGaussianKernelElement::set_function_grid_dy()
{
  auto t1 = std::chrono::high_resolution_clock::now();
  
  double dx = get_dx();
  double out = 0;
  gsl_vector * input = gsl_vector_alloc(2);
  double x = 0;
  double y = 0;

  for (int i=0; i<1/dx; ++i) {
    x = i*dx;
    gsl_vector_set(input, 0, x);

    for (int j=0; j<1/dx; ++j) {
      y = j*dx;
      gsl_vector_set(input, 1, y);

      out = first_derivative(input, 1);
      gsl_matrix_set(deriv_function_grid_dy_, i, j, out);
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "duration = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
    	    << " milliseconds\n";

  gsl_vector_free(input);
}

void BivariateGaussianKernelElement::set_function_grids()
{
  auto t1 = std::chrono::high_resolution_clock::now();
  double dx = get_dx();
  double function_val = 0;
  double function_dx = 0;
  double function_dy = 0;
  double alpha = get_exponent_power();
  
  gsl_vector * input = gsl_vector_alloc(2);
  double x = 0;
  double y = 0;
  double mollifier_x = 1.0;
  double mollifier_y = 1.0;
  double ay = 0.0;
  gsl_vector* ym = gsl_vector_alloc(2);

  for (int i=0; i<1/dx; ++i) {
    x = i*dx;

    mollifier_x = pow(x, get_exponent_power())*
      pow((1.0-x), get_exponent_power());
    
    for (int j=0; j<1/dx; ++j) {
      y = j*dx;

      gsl_vector_set(input, 0, x);
      gsl_vector_set(input, 1, y);

      mollifier_y = pow(y, get_exponent_power())*
	pow((1.0-y), get_exponent_power());

      gsl_vector_sub(input, get_mean_vector());
      gsl_blas_dsymv(CblasUpper,1.0,get_winv(),input,0.0,ym);
      gsl_blas_ddot(input, ym, &ay);
      ay = exp(-0.5*ay)/sqrt( pow((2*M_PI),2)*get_ax());
      function_val = ay * mollifier_x * mollifier_y;

      function_dx = alpha*
	(pow((1-x), alpha-1)*pow(x,alpha) + pow((1-x),alpha)*pow(x,alpha-1))*
	pow((1-y), alpha)*pow(y,alpha)*
	ay +
	pow((1-x), alpha)*pow(x,alpha)*
	pow((1-y), alpha)*pow(y,alpha)*
	ay*
	2*gsl_vector_get(ym,0);

      function_dy = alpha*
	(pow((1-y), alpha-1)*pow(y,alpha) + pow((1-y),alpha)*pow(y,alpha-1))*
	pow((1-x), alpha)*pow(x,alpha)*
	ay +
	pow((1-x), alpha)*pow(x,alpha)*
	pow((1-y), alpha)*pow(y,alpha)*
	ay*
	2*gsl_vector_get(ym,1);

      gsl_matrix_set(function_grid_, i, j, function_val);
      gsl_matrix_set(deriv_function_grid_dx_, i, j, function_dx);
      gsl_matrix_set(deriv_function_grid_dy_, i, j, function_dy);
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "duration = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
    	    << " milliseconds\n";

  gsl_vector_free(input);
  gsl_vector_free(ym);
}
