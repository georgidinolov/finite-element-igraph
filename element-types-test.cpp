#include "BasisElementTypes.hpp"
#include <iostream>
#include <vector>

int main() {
  double dx = 5e-3;
  long unsigned dimension = 2;
  double exponent_power = 1;
  
  gsl_vector* mean = gsl_vector_alloc(dimension);
  gsl_vector* input = gsl_vector_alloc(dimension);
  gsl_vector_set_all(mean, 0.5);
  gsl_vector_set_all(input, 0.5);


  gsl_matrix* cov = gsl_matrix_alloc(dimension, dimension);
  gsl_matrix_set_all(cov, 0.5);

  for (unsigned i=0; i<dimension; ++i) {
    gsl_matrix_set(cov, i, i, 1.0);
  }
  
  GaussianKernelElement kernel_element = GaussianKernelElement(dx,
  							       dimension,
  							       exponent_power,
  							       mean,
  							       cov);
  
  GaussianKernelElement kernel_element_2 = GaussianKernelElement(dx,
  								 dimension,
  								 exponent_power,
  								 mean,
  								 cov);

  LinearCombinationElement add =
    LinearCombinationElement(std::vector<const BasisElement*>
  			     {&kernel_element,
  				 &kernel_element_2,
  				 &kernel_element_2},
  			     std::vector<double> {100, 1, 1});
  
  std::cout << kernel_element(input)
  	    << "\n" << std::endl;

  std::cout << add(input) << std::endl;
  std::cout << kernel_element.norm() << std::endl;
  std::cout << add.norm() << std::endl;
  std::cout << kernel_element.first_derivative_finite_diff(mean, 0)
  	    << std::endl;

  LinearCombinationElement new_add = LinearCombinationElement(add);
  std::cout << "new_add.norm() = " << new_add.norm() << std::endl;

  return 0;
}
