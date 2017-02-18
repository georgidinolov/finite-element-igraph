#include "MultivariateNormal.hpp"
#include <iostream>

int main() {
  long unsigned dimension = 10;
  gsl_vector *mean = gsl_vector_alloc(dimension);
  gsl_matrix *covariance = gsl_matrix_alloc(dimension, dimension);
  
  for (unsigned i=0; i<dimension; ++i) {
    gsl_vector_set(mean, i, 1.0);
    gsl_matrix_set(covariance, i, i, 1.0);
    for (unsigned j=i+1; j<dimension; ++j) {
      gsl_matrix_set(covariance, i, j, 0.5);
      gsl_matrix_set(covariance, j, i, 0.5);
    }
  }
  
  double out = dmvnorm(dimension, mean, mean, covariance);
  
  std::cout << out << std::endl;
  std::cout << log(out) << std::endl;
  std::cout << dmvnorm_log(dimension, mean, mean, covariance) << std::endl;
									    
  gsl_vector_free(mean);
  gsl_matrix_free(covariance);
  return 0;
}
