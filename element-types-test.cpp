#include "BasisElementTypes.hpp"
#include <iostream>
#include <vector>

int main() {
  double dx = 1.0/128;
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
  
  BivariateGaussianKernelElement kernel_element = BivariateGaussianKernelElement(dx,
  							       exponent_power,
  							       mean,
  							       cov);
  
  BivariateGaussianKernelElement kernel_element_2 = BivariateGaussianKernelElement(dx,
  								 exponent_power,
  								 mean,
  								 cov);
  BivariateGaussianKernelElement kernel_element_3 =
    BivariateGaussianKernelElement(dx,
  				   exponent_power,
  				   mean,
  				   cov);

  BivariateGaussianKernelElement kernel_element_4 =
    BivariateGaussianKernelElement(kernel_element);
  BivariateGaussianKernelElement kernel_element_5 =
    BivariateGaussianKernelElement(kernel_element_3);

  std::cout << "Testing assignment operator for Gaussian kernel elem" << std::endl;
  BivariateGaussianKernelElement kernel_element_6 = kernel_element_4;
  
  std::cout << "Testing assignment operator for bivariate Gaussian kernel elem"
  	    << std::endl;
  
  BivariateGaussianKernelElement kernel_element_7 = kernel_element_5;

  std::cout << "Testing filling a vector" << std::endl;
  std::vector<BivariateGaussianKernelElement> vvec(0);
  vvec = std::vector<BivariateGaussianKernelElement>(2);
  vvec[0] = kernel_element_7;
  vvec[1] = BivariateGaussianKernelElement(dx, exponent_power+1,mean,cov);
  
  BivariateLinearCombinationElement add =
    BivariateLinearCombinationElement(std::vector<const BivariateElement*>
  				      {&kernel_element,
  					  &kernel_element_2,
  					  &kernel_element_3},
  				      std::vector<double> {100, 1, 1});
  
  std::cout << kernel_element(input)
  	    << "\n" << std::endl;

  std::cout << add(input) << std::endl;
  std::cout << kernel_element.norm() << std::endl;
  std::cout << add.norm() << std::endl;
  std::cout << kernel_element.first_derivative_finite_diff(mean, 0)
  	    << std::endl;

  BivariateLinearCombinationElement new_add = BivariateLinearCombinationElement(add);
  std::cout << "new_add.norm() = " << new_add.norm() << std::endl;

  BivariateSolverClassical small_t_solution = BivariateSolverClassical(1.0,1.0,0.0,
  								       0.5,0.5);

  small_t_solution.set_function_grid(dx);


  BivariateLinearCombinationElementFourier fft1 = 
    BivariateLinearCombinationElementFourier();

  std::cout << "fft1.get_FFT_grid()->size1 = " 
	    << fft1.get_FFT_grid()->size1 << std::endl;
  
  std::cout << "fft1.get_FFT_grid()->size2 = " 
	    << fft1.get_FFT_grid()->size2 << std::endl;
  std::cout << "1/fft1.get_dx() = " << 1/fft1.get_dx() << std::endl;

  BivariateLinearCombinationElementFourier fft2 = 
    BivariateLinearCombinationElementFourier(new_add);
  fft2.save_FFT_grid("odd-extension-fft.csv");
  
  BivariateLinearCombinationElementFourier fft3 = fft2;

  std::cout << "fft3.get_FFT_grid()->size1 = " << fft3.get_FFT_grid()->size1
	    << std::endl;
  std::cout << "fft3.get_FFT_grid()->size2 = " << fft3.get_FFT_grid()->size2
	    << std::endl;
  return 0;
}
