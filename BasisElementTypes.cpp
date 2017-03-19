#include <algorithm>
#include "BasisElementTypes.hpp"
#include <gsl/gsl_blas.h>
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
  : elements_(element.get_elements()),
    coefficients_(element.get_coefficients())
{
  if (elements_.size() != coefficients_.size()) {
    std::cout << "ERROR: elements and coefficients not of same size!"
	      << std::endl;
  }
}

LinearCombinationElement::~LinearCombinationElement()
{}

double LinearCombinationElement::
operator()(const igraph_vector_t& input) const
{
  double out = 0;
  for (unsigned i=0; i<elements_.size(); ++i) {
    const BasisElement* curr_element = elements_[i];
    out = out + coefficients_[i]*(*curr_element)(input);
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
first_derivative(const igraph_vector_t& input,
		 long int coord_index) const
{
  double deriv = 0;
  for (unsigned i=0; i<elements_.size(); ++i) {
    const BasisElement* curr_element = elements_[i];
    deriv = deriv + coefficients_[i]*(curr_element->
				      first_derivative(input,
						       coord_index));
  }
  return deriv;
}

std::vector<const BasisElement*> LinearCombinationElement::
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
		      const igraph_vector_t& mean_vector,
		      const igraph_matrix_t& covariance_matrix)
  : dx_(dx),
    dimension_(dimension),
    exponent_power_(exponent_power),
    mvtnorm_(MultivariateNormal())
{
  igraph_vector_init(&mean_vector_, dimension_);
  igraph_vector_update(&mean_vector_, &mean_vector);
  
  igraph_matrix_init(&covariance_matrix_, dimension_, dimension_);
  igraph_matrix_update(&covariance_matrix_, &covariance_matrix);

  //  set_norm();
  set_gsl_objects();
}

GaussianKernelElement::
GaussianKernelElement(const GaussianKernelElement& element)
  : dimension_(element.dimension_),
    exponent_power_(element.exponent_power_),
    mvtnorm_(MultivariateNormal())
{
  igraph_vector_init(&mean_vector_, dimension_);
  igraph_vector_update(&mean_vector_, &element.mean_vector_);
  
  igraph_matrix_init(&covariance_matrix_, dimension_, dimension_);
  igraph_matrix_update(&covariance_matrix_, &element.covariance_matrix_);

  set_gsl_objects();
}

GaussianKernelElement::~GaussianKernelElement()
{
  igraph_vector_destroy(&mean_vector_);
  igraph_matrix_destroy(&covariance_matrix_);

  gsl_vector_free(mean_vector_gsl_);
  gsl_vector_free(input_gsl_);
  gsl_matrix_free(covariance_matrix_gsl_);
}

double GaussianKernelElement::
operator()(const igraph_vector_t& input) const
{
  if (igraph_vector_size(&input) == dimension_) {
    double mollifier = 1;

    for (unsigned i=0; i<dimension_; ++i) {
      mollifier = mollifier *
	std::pow(igraph_vector_e(&input, i), exponent_power_) *
	std::pow((1-igraph_vector_e(&input, i)), exponent_power_);
      
      gsl_vector_set(input_gsl_, i, igraph_vector_e(&input, i));
    }
    
    double out = mvtnorm_.dmvnorm(dimension_,
				  input_gsl_,
				  mean_vector_gsl_,
				  covariance_matrix_gsl_) *
	mollifier;
    
    return out;
  } else {
    std::cout << "INPUT SIZE WRONG" << std::endl;
    return 0;
  }
}

double GaussianKernelElement::
first_derivative(const igraph_vector_t& input,
		 long int coord_index) const
{
  return first_derivative_finite_diff(input,
				      coord_index);
}

double GaussianKernelElement::
first_derivative_finite_diff(const igraph_vector_t& input,
			     long int coord_index) const
{
  igraph_vector_t input_plus;
  igraph_vector_t input_minus;

  igraph_vector_init(&input_plus, dimension_);
  igraph_vector_init(&input_minus, dimension_);
  
  igraph_vector_update(&input_plus, &input);
  igraph_vector_update(&input_minus, &input);

  igraph_vector_set(&input_plus, coord_index,
  		    igraph_vector_e(&input, coord_index)+dx_);
  igraph_vector_set(&input_minus, coord_index,
  		    igraph_vector_e(&input, coord_index)-dx_);

  double out = ((*this)(input_plus) - (*this)(input_minus))/
    (2*dx_);

  igraph_vector_destroy(&input_minus);
  igraph_vector_destroy(&input_plus);
  
  return out;
}

double GaussianKernelElement::norm_finite_diff() const
{
  long int N = 1.0/dx_;

  double integral = 0;
  igraph_vector_t input;
  igraph_vector_init(&input, dimension_);
  double x;
  double y;

  for (long int i=0; i<N; ++i) {
    for (long int j=0; j<N; ++j) {
      x = i*dx_;
      y = j*dx_;
      igraph_vector_set(&input, 0, x);
      igraph_vector_set(&input, 1, y);

      integral = integral + std::pow((*this)(input), 2);
    }
  }
  integral = integral * std::pow(dx_, dimension_);
  
  igraph_vector_destroy(&input);
  return std::sqrt(integral);
}

double GaussianKernelElement::norm() const
{
  return norm_finite_diff();
}

const igraph_vector_t& GaussianKernelElement::get_mean_vector() const
{
  return mean_vector_;
}

const igraph_matrix_t& GaussianKernelElement::get_covariance_matrix() const
{
  return covariance_matrix_;
}

// void GaussianKernelElement::set_norm()
// {
//   norm_ = norm_finite_diff();
// }

void GaussianKernelElement::set_gsl_objects()
{
    covariance_matrix_gsl_ = gsl_matrix_alloc(dimension_,
					      dimension_);
    mean_vector_gsl_ = gsl_vector_alloc(dimension_);
    input_gsl_ = gsl_vector_alloc(dimension_);
    
    for (unsigned i=0; i<dimension_; ++i) {
      gsl_vector_set(mean_vector_gsl_, i, igraph_vector_e(&mean_vector_, i));
      gsl_vector_set(input_gsl_, i, igraph_vector_e(&mean_vector_, i));
      
      gsl_matrix_set(covariance_matrix_gsl_, i, i,
		     igraph_matrix_e(&covariance_matrix_,
				     i, i));
      for (unsigned j=i+1; j<dimension_; ++j) {
	gsl_matrix_set(covariance_matrix_gsl_, i, j,
		       igraph_matrix_e(&covariance_matrix_,
				       i, j));
	gsl_matrix_set(covariance_matrix_gsl_, j, i,
		       igraph_matrix_e(&covariance_matrix_,
				       j, i));
      }
    }
}
