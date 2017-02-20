#include "BasisTypes.hpp"
#include <iostream>
#include <vector>

int main() {
  long unsigned dimension = 2;
  double exponent_power = 1;
  
  igraph_vector_t mean;
  igraph_vector_t input;
  igraph_vector_init(&mean, dimension);
  igraph_vector_init(&input, dimension);
  igraph_vector_fill(&mean, 0.5);
  igraph_vector_fill(&input, 0.5);

  igraph_matrix_t cov;
  igraph_matrix_init(&cov, dimension, dimension);
  igraph_matrix_fill(&cov, 0.5);

  for (unsigned i=0; i<dimension; ++i) {
    igraph_matrix_set(&cov, i, i, 1.0);
  }
  
  GaussianKernelElement kernel_element = GaussianKernelElement(dimension,
							       exponent_power,
							       mean,
							       cov);
  
  GaussianKernelElement kernel_element_2 = GaussianKernelElement(dimension,
								 exponent_power,
								 mean,
								 cov);

  LinearCombinationElement add = LinearCombinationElement(std::vector<const BasisElement*>
			   {&kernel_element, &kernel_element_2, &kernel_element_2},
							  std::vector<double> {100, 1, 1});
  
  std::cout << kernel_element(input)
	    << "\n" << std::endl;

  std::cout << add(input) << std::endl;
  std::cout << kernel_element.norm() << std::endl;
  std::cout << add.norm() << std::endl;

  igraph_vector_print(&kernel_element.get_mean_vector());
  return 0;
}
