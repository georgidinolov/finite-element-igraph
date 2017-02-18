#include "BasisTypes.hpp"
#include <iostream>
#include <vector>

int main() {
  GaussianKernelBasis basis = GaussianKernelBasis(0.5,
						  0.3,
						  1,
						  0.5);
  
  const igraph_matrix_t& mass_matrix = basis.get_mass_matrix();
  std::cout << &mass_matrix << std::endl;
  std::cout << igraph_matrix_e(&mass_matrix, 0, 0) << " ";
  std::cout << igraph_matrix_e(&mass_matrix, 0, 1) << std::endl;
  std::cout << igraph_matrix_e(&mass_matrix, 1, 0) << " ";
  std::cout << igraph_matrix_e(&mass_matrix, 1, 1) << "\n" << std::endl;

  igraph_vector_t first_col;
  igraph_vector_init(&first_col, 1);
  igraph_matrix_get_col(&mass_matrix, &first_col, 1);
  igraph_vector_print(&first_col);

  igraph_vector_destroy(&first_col);

  return 0;
}
