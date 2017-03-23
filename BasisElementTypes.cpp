#include <algorithm>
#include <chrono>
#include "BasisElementTypes.hpp"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <iostream>

// =================== BASIS ELEMENT CLASS ===================
BasisElement::~BasisElement()
{}

// ============== LINEAR COMBINATION ELEMENT =====================
LinearCombinationElement::
LinearCombinationElement(const std::vector<const BasisElement*> elements,
			 const std::vector<double>& coefficients)
  : elements_(elements),
    coefficients_(coefficients)
{
  if (elements_.size() != coefficients_.size()) {
    std::cout << "ERROR: elements and coefficients not of same size!"
	      << std::endl;
  }
}

LinearCombinationElement::
LinearCombinationElement(const LinearCombinationElement& element)
  : elements_(element.elements_),
    coefficients_(element.coefficients_)
{
  if (elements_.size() != coefficients_.size()) {
    std::cout << "ERROR: elements and coefficients not of same size!"
	      << std::endl;
  }
}

LinearCombinationElement::~LinearCombinationElement()
{}

double LinearCombinationElement::
operator()(const gsl_vector* input) const
{
  double out = 0;
  for (unsigned i=0; i<elements_.size(); ++i) {
    out = out + coefficients_[i]*(*elements_[i])(input);
  }
  return out;
}

double LinearCombinationElement::norm() const
{
  double integral = 0;
  for (unsigned i=0; i<elements_.size(); ++i) {
    const BasisElement* curr_element = elements_[i];
    integral = integral + coefficients_[i]*(curr_element->norm());
  }
  return integral;
}

double LinearCombinationElement::
first_derivative(const gsl_vector* input,
		 long int coord_index) const
{
  double deriv = 0;
  for (unsigned i=0; i<elements_.size(); ++i) {
    deriv = deriv + coefficients_[i]*(elements_[i]->
				      first_derivative(input,
						       coord_index));
  }
  return deriv;
}

const std::vector<const BasisElement*> LinearCombinationElement::
get_elements() const
{
  return elements_;
}

std::vector<double> LinearCombinationElement::get_coefficients() const
{
  return coefficients_;
}

double LinearCombinationElement::get_coefficient(unsigned i) const
{
  if (i < coefficients_.size()) {
    return coefficients_[i];
  } else {
    std::cout << "ERROR: coefficient out of range" << std::endl;
    return 0;
  }
}

// ============== GAUSSIAN KERNEL ELEMENT =====================
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
    mvtnorm_(MultivariateNormal())
{
  gsl_vector_memcpy(mean_vector_, mean_vector);
  gsl_matrix_memcpy(covariance_matrix_, covariance_matrix);

  //  set_norm();
  // set_gsl_objects();
}

GaussianKernelElement::
GaussianKernelElement(const GaussianKernelElement& element)
  : dx_(element.dx_),
    dimension_(element.dimension_),
    exponent_power_(element.exponent_power_),
    mean_vector_(gsl_vector_alloc(dimension_)),
    input_gsl_(gsl_vector_alloc(dimension_)),
    covariance_matrix_(gsl_matrix_alloc(dimension_,
					dimension_)),
    mvtnorm_(MultivariateNormal())
{
  if (element.dimension_ != element.mean_vector_->size) {
    std::cout << "ERROR: Dimensions do not match" << std::endl;
  }

  gsl_vector_memcpy(mean_vector_, element.mean_vector_);
  gsl_matrix_memcpy(covariance_matrix_, element.covariance_matrix_);
  
  // set_gsl_objects();
}

GaussianKernelElement::~GaussianKernelElement()
{
  gsl_vector_free(mean_vector_);
  gsl_matrix_free(covariance_matrix_);

  gsl_vector_free(input_gsl_);
}

double GaussianKernelElement::
operator()(const gsl_vector* input) const
{
  if (input->size == dimension_) {
    double mollifier = 1;

    for (unsigned i=0; i<dimension_; ++i) {
      mollifier = mollifier *
	std::pow(gsl_vector_get(input, i), exponent_power_) *
	std::pow((1-gsl_vector_get(input, i)), exponent_power_);
      
      gsl_vector_set(input_gsl_, i, gsl_vector_get(input, i));
    }
    
    double out = mvtnorm_.dmvnorm(dimension_,
				  input_gsl_,
				  mean_vector_,
				  covariance_matrix_) *
	mollifier;
    
    return out;
  } else {
    std::cout << "INPUT SIZE WRONG" << std::endl;
    return 0;
  }
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
  set_function_grid();
}

BivariateGaussianKernelElement::
~BivariateGaussianKernelElement()
{
  gsl_matrix_free(function_grid_);
  gsl_matrix_free(deriv_function_grid_dx_);
  gsl_matrix_free(deriv_function_grid_dy_);
}

void BivariateGaussianKernelElement::set_function_grid()
{
  auto t1 = std::chrono::high_resolution_clock::now();
  int s=0;
  double ax=0,ay=0;
  gsl_vector *ym = gsl_vector_alloc(2);
  gsl_matrix *work = gsl_matrix_alloc(2,2);
  gsl_matrix *winv = gsl_matrix_alloc(2,2);
  gsl_permutation *p = gsl_permutation_alloc(2);
  
  gsl_matrix_memcpy( work, get_covariance_matrix() );
  gsl_linalg_LU_decomp( work, p, &s );
  gsl_linalg_LU_invert( work, p, winv );
  ax = gsl_linalg_LU_det( work, s );
  
  double dx = get_dx();
  double out = 0;
  gsl_vector * input = gsl_vector_alloc(2);
  double x = 0;
  double y = 0;
  double mollifier_x = 1.0;
  double mollifier = 1.0;


  for (int i=0; i<1/dx; ++i) {
    x = i*dx;
    gsl_vector_set(input, 0, x);

    mollifier_x = pow(x, get_exponent_power()) *
      pow((1-x), get_exponent_power());
    
    for (int j=0; j<1/dx; ++j) {
      y = j*dx;
      gsl_vector_set(input, 1, y);

      mollifier = mollifier_x *
	pow(y, get_exponent_power()) *
	pow((1-y), get_exponent_power());

      gsl_vector_sub(input, get_mean_vector());
      gsl_blas_dsymv(CblasUpper,1.0,winv,input,0.0,ym);
      gsl_blas_ddot( input, ym, &ay);
      ay = exp(-0.5*ay)/sqrt( pow((2*M_PI),2)*ax );
	
      gsl_matrix_set(function_grid_, i, j, ay*mollifier);
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "duration = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
    	    << " milliseconds\n";

  gsl_vector_free(input);
  gsl_matrix_free( work );
  gsl_permutation_free( p );
  gsl_matrix_free( winv );
  gsl_vector_free(ym);
}

