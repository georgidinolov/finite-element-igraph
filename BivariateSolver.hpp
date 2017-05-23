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
  BivariateSolver();
  BivariateSolver(BivariateBasis* basis);
  BivariateSolver(BivariateBasis* basis,
		  double sigma_x,
		  double sigma_y,
		  double rho,
		  double a,
		  double x_0,
		  double b,
		  double c,
		  double y_0,
		  double d,
		  double t,
		  double dx);
  BivariateSolver(const BivariateSolver& solver);
  virtual BivariateSolver& operator=(const BivariateSolver& rhs);

  virtual ~BivariateSolver();
  
  inline double get_t() const { return t_; }
  inline const BivariateBasis* get_basis() const { return basis_; }
    
  // need to reset t_ AND solution_coefs_!
  void set_t(double t);
  // need to reset IC_coefs_, mass_matrix_, stiffness_matrix_, eval_,
  // evec_, solution_coefs_
  void set_diffusion_parameters(double sigma_x,
				double sigma_y,
				double rho);
  
  gsl_vector * scale_input(const gsl_vector* input) const;
  
  void set_data(double a,
		double x_0,
		double b,
		double c,
		double y_0,
		double d);
  
  const gsl_vector* get_solution_coefs() const;
  const gsl_vector* get_ic_coefs() const;
  const gsl_vector* get_evals() const;
  
  
  virtual double operator()(const gsl_vector* input) const;
  virtual double numerical_likelihood_second_order(const gsl_vector* input, double h);
  virtual double numerical_likelihood_first_order(const gsl_vector* input, double h);
  virtual double numerical_likelihood(const gsl_vector* input, double h);
  
private:
  double a_;
  double b_;
  double c_;
  double d_;
  double sigma_x_;
  double sigma_y_;
  double rho_;
  double x_0_;
  double y_0_;

  // scaled params. a_2_ and c_2_ are numerically zero.
  double a_2_;
  double b_2_;
  double c_2_;
  double d_2_;
  double sigma_x_2_;
  double sigma_y_2_;
  double x_0_2_;
  double y_0_2_;
  
  MultivariateNormal mvtnorm_;
  BivariateBasis* basis_;
  BivariateSolverClassical* small_t_solution_;
  double t_;
  double dx_;
  gsl_vector* IC_coefs_;
  gsl_matrix* mass_matrix_;
  gsl_matrix* stiffness_matrix_;
  gsl_vector* eval_;
  gsl_matrix* evec_;
  gsl_vector* solution_coefs_;

  void set_scaled_data();
  void set_IC_coefs();
  // requires sigma_x_, sigma_y_, rho_;
  void set_mass_and_stiffness_matrices();
  // requires mass_matrix_ and stiffness_matrix_;
  void set_eval_and_evec();
  // requires t_, eval_, evec_;
  void set_solution_coefs();
  
  // gsl_vector * xi_eta_input_;
  // gsl_vector * initial_condition_xi_eta_;
  // gsl_matrix * Rotation_matrix_;
  // gsl_matrix * Variance_;
  // gsl_vector * initial_condition_xi_eta_reflected_;
};
