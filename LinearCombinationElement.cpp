#include <algorithm>
#include <chrono>
#include "BasisElementTypes.hpp"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <iostream>

// ============== LINEAR COMBINATION ELEMENT =====================
LinearCombinationElement::
LinearCombinationElement(const std::vector<const BasisElement*> elements,
			 const std::vector<double>& coefficients)
  : elements_(elements),
    coefficients_(coefficients),
    dx_(elements[0]->get_dx()),
    function_grid_(gsl_matrix_alloc(1/elements[0]->get_dx(),
    				    1/elements[0]->get_dx()))    
{
  if (elements_.size() != coefficients_.size()) {
    std::cout << "ERROR: elements and coefficients not of same size!"
	      << std::endl;
  }
  std::cout << "size_1 = " << function_grid_->size1 << std::endl;
  std::cout << "size_2 = " << function_grid_->size2 << std::endl;
  set_function_grids();
}

LinearCombinationElement::
LinearCombinationElement(const LinearCombinationElement& element)
  : elements_(element.elements_),
    coefficients_(element.coefficients_),
    dx_(element.elements_[0]->get_dx()),
    function_grid_(gsl_matrix_alloc(1/elements_[0]->get_dx(),
				    1/elements_[0]->get_dx()))
{
  if (elements_.size() != coefficients_.size()) {
    std::cout << "ERROR: elements and coefficients not of same size!"
	      << std::endl;
  }
  gsl_matrix_memcpy(function_grid_, element.function_grid_);
}

LinearCombinationElement::~LinearCombinationElement()
{
  gsl_matrix_free(function_grid_);
}

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

void LinearCombinationElement::set_function_grids()
{

  auto t1 = std::chrono::high_resolution_clock::now();
  double dx = get_dx();
  double in = 0;
  double in_dx = 0;
  double in_dy = 0;

  std::cout << "1/dx = " << 1/get_dx() << std::endl;
  std::cout << "size_1 = "
	    << elements_[0]->get_function_grid()->size1
	    << std::endl;
  std::cout << "size_2 = "
	    << elements_[0]->get_function_grid()->size2
	    << std::endl;
  
  for (int i=0; i<1/dx; ++i) {
    for (int j=0; j<1/dx; ++j) {
      in = 0;
      for (unsigned k=0; k<elements_.size(); ++k) {
	in = in +
	  gsl_matrix_get(elements_[k]->get_function_grid(),i,j);
	// in_dx = in_dx +
	//   gsl_matrix_get(elements_[k]->get_deriv_function_grid_dx(),i,j);
	// in_dy = in_dy +
	//   gsl_matrix_get(elements_[k]->get_deriv_function_grid_dx(),i,j);
      }
      gsl_matrix_set(function_grid_, i,j, in);
      // gsl_matrix_set(deriv_function_grid_dx_, i,j, in_dx);
      // gsl_matrix_set(deriv_function_grid_dy_, i,j, in_dy);
    }
  }
  
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "duration = "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
    	    << " milliseconds\n";
}
