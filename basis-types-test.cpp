#include <algorithm>
#include "BasisTypes.hpp"
#include <fstream>
#include <iostream>
#include <vector>

int main() {
  double dx = 5e-3;
  BivariateGaussianKernelBasis basis = BivariateGaussianKernelBasis(dx,
								    0.9,
								    0.3,
								    1,
								    0.5);
  
  const BivariateLinearCombinationElement& ortho_e_1 =
    basis.get_orthonormal_element(0);
  const BivariateLinearCombinationElement& ortho_e_2 =
    basis.get_orthonormal_element(1);
  const BivariateLinearCombinationElement& ortho_e_3 =
    basis.get_orthonormal_element(2);

  const BivariateLinearCombinationElement& ortho_e_100 =
    basis.get_orthonormal_element(99);
  const BivariateLinearCombinationElement& ortho_e_99 =
    basis.get_orthonormal_element(98);

  const BivariateLinearCombinationElement& last_ortho_elem =
    basis.get_orthonormal_element(basis.
				  get_orthonormal_elements().
				  size()-1);

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

  std::cout << "<ortho_e_99 | ortho_e_100> = "
	    << basis.project(ortho_e_99, ortho_e_100) << std::endl;

  // OUTPUTTING FUNCTION GRID START
  const BivariateElement& printed_elem = basis.get_orthonormal_element(10);
  std::cout << "norm of printed_elem = " << printed_elem.norm() << std::endl;

  int N = 1/dx;
  double x = 0.0;
  double y = 0.0;
  double min_out = 0;
  double max_out = 0;
  
  gsl_matrix_minmax(printed_elem.get_function_grid(), &min_out, &max_out);
  std::cout << "min_out = " << min_out << "\n";
  std::cout << "max_out = " << max_out << std::endl;
  std::vector<double> minmax = std::vector<double> {std::abs(min_out),
						    std::abs(max_out)};
  double abs_max = *std::max_element(minmax.begin(),
				     minmax.end());
  std::ofstream output_file;
  output_file.open("orthonormal-element.csv");

  // header
  output_file << "x, y, function.val\n";
  
  for ( unsigned i=0; i<N; ++i) {
    x = dx*i;
    for (unsigned j=0; j<N; ++j) {
      y = dx*j;

      output_file << x << ","
		  << y << ","
		  << (gsl_matrix_get(printed_elem.get_function_grid(),
				     i, j) - min_out) / (max_out-min_out) << "\n";
    }
  }
  output_file.close();
  // OUTPUTTING FUNCTION GRID END

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
