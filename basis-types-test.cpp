#include "BasisTypes.hpp"
#include <iostream>
#include <vector>

int main() {
  double dx = 1e-2;
  BivariateGaussianKernelBasis basis = BivariateGaussianKernelBasis(dx,
								    0.9,
								    0.3,
								    1,
								    0.5);
  
  const LinearCombinationElement& ortho_e_1 = basis.get_orthonormal_element(0);
  const LinearCombinationElement& ortho_e_2 = basis.get_orthonormal_element(1);
  const LinearCombinationElement& ortho_e_3 = basis.get_orthonormal_element(2);

  std::cout << "<ortho_e_1 | ortho_e_1> = "
	    << basis.project(ortho_e_1, ortho_e_1) << std::endl;
  
  std::cout << "<ortho_e_1 | ortho_e_2> = "
	    << basis.project(ortho_e_1, ortho_e_2) << std::endl;

  std::cout << "<ortho_e_1 | ortho_e_3> = "
	    << basis.project(ortho_e_1, ortho_e_3) << std::endl;

  std::cout << "<ortho_e_2 | ortho_e_3> = "
	    << basis.project(ortho_e_2, ortho_e_3) << std::endl;
  
  std::cout << "<ortho_e_2 | ortho_e_2> = "
	    << basis.project(ortho_e_2, ortho_e_2) << std::endl;

  std::cout << "<ortho_e_3 | ortho_e_3> = "
	    << basis.project(ortho_e_3, ortho_e_3) << std::endl;
  
  
  // const igraph_matrix_t& mass_matrix = basis.get_mass_matrix();
  // std::cout << &mass_matrix << std::endl;
  // std::cout << igraph_matrix_e(&mass_matrix, 0, 0) << " ";
  // std::cout << igraph_matrix_e(&mass_matrix, 0, 1) << std::endl;
  // std::cout << igraph_matrix_e(&mass_matrix, 1, 0) << " ";
  // std::cout << igraph_matrix_e(&mass_matrix, 1, 1) << "\n" << std::endl;

  // igraph_vector_t first_col;
  // igraph_vector_init(&first_col, 1);
  // igraph_matrix_get_col(&mass_matrix, &first_col, 1);
  // igraph_vector_print(&first_col);

  // igraph_vector_destroy(&first_col);

  return 0;
}
