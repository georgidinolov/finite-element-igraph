#include <algorithm>
#include <chrono>
#include "BasisElementTypes.hpp"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <iostream>

// ============== LINEAR COMBINATION ELEMENT =====================
BivariateLinearCombinationElement::
BivariateLinearCombinationElement(const std::vector<const BivariateElement*>& elements,
				  const std::vector<double>& coefficients)
  : elements_(elements),
    coefficients_(coefficients),
    dx_(elements[0]->get_dx()),
    function_grid_(gsl_matrix_alloc(1/elements[0]->get_dx(),
    				    1/elements[0]->get_dx())),
    deriv_function_grid_dx_(gsl_matrix_alloc(1/elements_[0]->get_dx(),
					     1/elements_[0]->get_dx())),
    deriv_function_grid_dy_(gsl_matrix_alloc(1/elements_[0]->get_dx(),
					     1/elements_[0]->get_dx()))
{
  if (elements_.size() != coefficients_.size()) {
    std::cout << "ERROR: elements and coefficients not of same size!"
	      << std::endl;
  }
  std::cout << "size_1 = " << function_grid_->size1 << std::endl;
  std::cout << "size_2 = " << function_grid_->size2 << std::endl;
  set_function_grids();
}

BivariateLinearCombinationElement::
BivariateLinearCombinationElement(const BivariateLinearCombinationElement& element)
  : elements_(element.elements_),
    coefficients_(element.coefficients_),
    dx_(element.elements_[0]->get_dx()),
    function_grid_(gsl_matrix_alloc(1/elements_[0]->get_dx(),
				    1/elements_[0]->get_dx())),
    deriv_function_grid_dx_(gsl_matrix_alloc(1/elements_[0]->get_dx(),
					     1/elements_[0]->get_dx())),
    deriv_function_grid_dy_(gsl_matrix_alloc(1/elements_[0]->get_dx(),
					     1/elements_[0]->get_dx()))
    
{
  if (elements_.size() != coefficients_.size()) {
    std::cout << "ERROR: elements and coefficients not of same size!"
	      << std::endl;
  }
  gsl_matrix_memcpy(function_grid_, element.function_grid_);
  gsl_matrix_memcpy(deriv_function_grid_dx_, element.deriv_function_grid_dx_);
  gsl_matrix_memcpy(deriv_function_grid_dy_, element.deriv_function_grid_dy_);
}

BivariateLinearCombinationElement::~BivariateLinearCombinationElement()
{
  gsl_matrix_free(function_grid_);
  gsl_matrix_free(deriv_function_grid_dx_);
  gsl_matrix_free(deriv_function_grid_dy_);
}

double BivariateLinearCombinationElement::
operator()(const gsl_vector* input) const
{
  double out = 0;
  for (unsigned i=0; i<elements_.size(); ++i) {
    out = out + coefficients_[i]*(*elements_[i])(input);
  }
  return out;
}

double BivariateLinearCombinationElement::norm() const
{
  double integral = 0;
  for (int i=0; i<1/get_dx(); ++i) {
    for (int j=0; j<1/get_dx(); ++j) {
      integral = integral + 
	std::pow(gsl_matrix_get(get_function_grid(), i,j), 2);
    }
  }

  integral = std::sqrt(integral * std::pow(get_dx(), 2));

  return integral;
}

double BivariateLinearCombinationElement::
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

const std::vector<const BivariateElement*> BivariateLinearCombinationElement::
get_elements() const
{
  return elements_;
}

const std::vector<double>& BivariateLinearCombinationElement::get_coefficients() const
{
  return coefficients_;
}

void BivariateLinearCombinationElement::
set_coefficients(const std::vector<double>& new_coefs)
{
  coefficients_ = new_coefs;
  set_function_grids();
}

double BivariateLinearCombinationElement::get_coefficient(unsigned i) const
{
  if (i < coefficients_.size()) {
    return coefficients_[i];
  } else {
    std::cout << "ERROR: coefficient out of range" << std::endl;
    return 0;
  }
}

void BivariateLinearCombinationElement::
set_function_grid(const gsl_matrix* new_function_grid) 
{
  gsl_matrix_memcpy(function_grid_, new_function_grid);
}

void BivariateLinearCombinationElement::
set_deriv_function_grid_dx(const gsl_matrix* new_deriv_function_grid_dx) 
{
  gsl_matrix_memcpy(deriv_function_grid_dx_, new_deriv_function_grid_dx);
}

void BivariateLinearCombinationElement::
set_deriv_function_grid_dy(const gsl_matrix* new_deriv_function_grid_dy) 
{
  gsl_matrix_memcpy(deriv_function_grid_dy_, new_deriv_function_grid_dy);
}


void BivariateLinearCombinationElement::set_function_grids()
{
  double dx = get_dx();
  double in = 0;
  double in_dx = 0;
  double in_dy = 0;

  gsl_matrix_memcpy(function_grid_, elements_[0]->get_function_grid());
  gsl_matrix_scale(function_grid_, coefficients_[0]);
  
  gsl_matrix_memcpy(deriv_function_grid_dx_,
  		    elements_[0]->get_deriv_function_grid_dx());
  gsl_matrix_scale(deriv_function_grid_dx_, coefficients_[0]);

  gsl_matrix_memcpy(deriv_function_grid_dy_,
  		    elements_[0]->get_deriv_function_grid_dy());
  gsl_matrix_scale(deriv_function_grid_dy_, coefficients_[0]);

  for (unsigned k=1; k<elements_.size(); ++k) {
    if ( std::abs(coefficients_[k]) > 1e-32) {
      gsl_matrix_scale(function_grid_, 1.0/coefficients_[k]);
      gsl_matrix_add(function_grid_, elements_[k]->get_function_grid());
      gsl_matrix_scale(function_grid_, coefficients_[k]);

      gsl_matrix_scale(deriv_function_grid_dx_, 1.0/coefficients_[k]);
      gsl_matrix_add(deriv_function_grid_dx_,
      		     elements_[k]->get_deriv_function_grid_dx());
      gsl_matrix_scale(deriv_function_grid_dx_, coefficients_[k]);

      gsl_matrix_scale(deriv_function_grid_dy_, 1.0/coefficients_[k]);
      gsl_matrix_add(deriv_function_grid_dy_,
      		     elements_[k]->get_deriv_function_grid_dy());
      gsl_matrix_scale(deriv_function_grid_dy_, coefficients_[k]);
    }
  }

  // for (int i=0; i<1/get_dx(); ++i) {
  //   for (int j=0; j<1/get_dx(); ++j) {
  //     in = 0;
  //     for (unsigned k=0; k<elements_.size(); ++k) {
  // 	in = in + coefficients_[k]*
  // 	  gsl_matrix_get(elements_[k]->get_function_grid(),i,j);
  // 	// in_dx = in_dx + coefficients_[k]*
  // 	//   gsl_matrix_get(elements_[k]->get_deriv_function_grid_dx(),i,j);
  // 	// in_dy = in_dy + coefficients_[k]*
  // 	//   gsl_matrix_get(elements_[k]->get_deriv_function_grid_dx(),i,j);
  //     }
  //     gsl_matrix_set(function_grid_, i,j, in);
  //     // gsl_matrix_set(deriv_function_grid_dx_, i,j, in_dx);
  //     // gsl_matrix_set(deriv_function_grid_dy_, i,j, in_dy);
  //   }
  // }

}
