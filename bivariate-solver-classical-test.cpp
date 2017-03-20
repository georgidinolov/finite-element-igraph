#include "BivariateSolverClassical.hpp"
#include <iostream>
#include <vector>

int main() {
  BivariateSolverClassical classical_solver =
    BivariateSolverClassical(1.0, 0.1, -0.8,
			     0.5, 0.5);
  return 0;
}
