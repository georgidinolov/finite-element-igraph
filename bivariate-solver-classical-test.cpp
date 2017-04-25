#include "BasisElementTypes.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>

int main() {
  std::cout << std::fixed << std::setprecision(32);
  double x_T = 0.69587235906731104151390354672913;
  double y_T = 0.59615714684947884727250766445650;
  double x_0 = 0.42972231989526271656032463397423;
  double y_0 = 0.00633588300194680778543165899919;
  double sigma_x = 0.88008638461644062012112499360228;
  double sigma_y = 0.94621168768833074924629045199254;
  double rho = 0.400;
  
  BivariateSolverClassical classical_solver =
    BivariateSolverClassical(sigma_x, sigma_y, rho,
			     x_0, y_0);
  double dx = 0.002;
  classical_solver.set_function_grid(dx);
  classical_solver.save_function_grid("ic.csv");

  return 0;
}


