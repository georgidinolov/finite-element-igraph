#include "BasisTypes.hpp"
#include <iostream>
#include <vector>

int main() {
  GaussianKernelBasis basis = GaussianKernelBasis(0.5,
						  0.3,
						  1,
						  0.5);
  
  const igraph_matrix_t mass_matrix = basis.get_mass_matrix();
  std::cout << &mass_matrix << std::endl;
  std::cout << igraph_

  igraph_vector_t first_col;
  igraph_matrix_get_col(&mass_matrix,
			&first_col,
			0);
  
  //  igraph_vector_print(&first_col);
  return 1;
}
