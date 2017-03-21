// the basis needs to give us the mass matrix, the system matrices, as
// well as functions which can be evaluated. Each Basis class needs to
// supply a mass and system matrix, as well as the functions used to
// create those matrices.

extern "C" {
#include "igraph.h"
}
#include "BasisTypes.hpp"

class BivariateSolver
{
public:
  BivariateSolver(const BivariateBasis& basis,
		  double sigma_x,
		  double sigma_y,
		  double rho,
		  double x_0,
		  double y_0);
  ~BivariateSolver();
  
  // IMPLEMENT
  virtual double operator()(const igraph_vector_t& input) const;
  virtual double operator()(const gsl_vector* input) const;
  
private:
  double sigma_x_;
  double sigma_y_;
  double rho_;
  double x_0_;
  double y_0_;
  MultivariateNormal mvtnorm_;
  const BivariateBasis& basis_;

  BivariateSolverClassical small_t_solution_;

  // gsl_vector * xi_eta_input_;
  // gsl_vector * initial_condition_xi_eta_;
  // gsl_matrix * Rotation_matrix_;
  // gsl_matrix * Variance_;
  // gsl_vector * initial_condition_xi_eta_reflected_;
};
