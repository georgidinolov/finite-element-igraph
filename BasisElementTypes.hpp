// the basis needs to give us the mass matrix, the system matrix, as
// well as functions which can be evaluated. Each Basis class needs to
// supply a mass and system matrix, as well as the functions used to
// create those matrices.

extern "C" {
#include "igraph.h"
}
#include "MultivariateNormal.hpp"
#include <vector>

// NEEDS TO BE UPDATED AS MORE TYPES ARE ADDED
enum BasisType { linear_combination, gaussian, other};

// =================== BASIS ELEMENT CLASS ===================
class BasisElement
{
public:
  BasisElement() {};
  virtual ~BasisElement() =0;
  virtual double operator()(const gsl_vector* input) const =0;
  virtual double norm() const =0;
  virtual double first_derivative(const gsl_vector* input,
				  long int coord_index) const =0;
};

// ============== LINEAR COMBINATION ELEMENT =====================
class LinearCombinationElement
  : public BasisElement
{
public:
  LinearCombinationElement(const std::vector<const BasisElement*> elements,
			   const std::vector<double>& coefficients);
  LinearCombinationElement(const LinearCombinationElement& lin_comb_element);

  virtual ~LinearCombinationElement();
  virtual double operator()(const gsl_vector* input) const;
  virtual double norm() const;
  virtual double first_derivative(const gsl_vector* input,
				  long int coord_index) const;

  const std::vector<const BasisElement*> get_elements() const;
  std::vector<double> get_coefficients() const;
  double get_coefficient(unsigned i) const;
  
private:
  const std::vector<const BasisElement*> elements_;
  std::vector<double> coefficients_;
};

// ============== GAUSSIAN KERNEL ELEMENT =====================
class GaussianKernelElement
  : public BasisElement
{
public:
  GaussianKernelElement();
  GaussianKernelElement(double dx,
			long unsigned dimension,
			double exponent_power,
			const gsl_vector* mean_vector,
			const gsl_matrix* covariance_matrix);
  
  GaussianKernelElement(const GaussianKernelElement& element);

  virtual ~GaussianKernelElement();
  virtual double operator()(const gsl_vector* input) const;

  virtual double first_derivative(const gsl_vector* input,
				  long int coord_index) const;

  virtual double first_derivative_finite_diff(const gsl_vector* input,
					      long int coord_index) const;

  virtual double norm() const;
  virtual double norm_finite_diff() const;

  const gsl_vector* get_mean_vector() const;
  const gsl_matrix* get_covariance_matrix() const;
  inline double get_dx() const
  { return dx_; }
  inline double get_exponent_power() const
  { return exponent_power_; }
  
private:
  double dx_;
  long int dimension_;
  double exponent_power_;

  gsl_vector *mean_vector_;
  gsl_vector *input_gsl_;
  gsl_matrix *covariance_matrix_;
  
  MultivariateNormal mvtnorm_;
};

class BivariateGaussianKernelElement
  : public GaussianKernelElement
{
public:
  BivariateGaussianKernelElement();
  BivariateGaussianKernelElement(double dx,
				 double exponent_power,
				 const gsl_vector* mean_vector,
				 const gsl_matrix* covariance_matrix);
  
  // BivariateGaussianKernelElement(const BivariateGaussianKernelElement& element);
  virtual ~BivariateGaussianKernelElement();

  inline virtual const gsl_matrix * get_function_grid() const
  { return function_grid_; }
  inline virtual const gsl_matrix * get_deriv_function_grid_dx() const
  { return deriv_function_grid_dx_; }
  inline virtual const gsl_matrix * get_deriv_function_grid_dy() const
  { return deriv_function_grid_dy_; }

private:
  void set_function_grid();
  
  gsl_matrix * function_grid_;
  gsl_matrix * deriv_function_grid_dx_;
  gsl_matrix * deriv_function_grid_dy_;
};

// ============== BIVARIATE SOLVER ====================
class BivariateSolverClassical
  : public BasisElement
{
public:
  BivariateSolverClassical(double sigma_x,
			   double sigma_y,
			   double rho,
			   double x_0,
			   double y_0);
  ~BivariateSolverClassical();

  virtual double operator()(const gsl_vector* input) const;
  virtual double operator()(const gsl_vector* input,
			    double t) const;
  virtual double norm() const;
  virtual double first_derivative(const gsl_vector* input,
				  long int coord_index) const;
  double get_t() const;
private:
  double sigma_x_;
  double sigma_y_;
  double rho_;
  double x_0_;
  double y_0_;
  MultivariateNormal mvtnorm_;

  gsl_vector * xi_eta_input_;
  gsl_vector * initial_condition_xi_eta_;
  gsl_matrix * Rotation_matrix_;
  double tt_;
  gsl_matrix * Variance_;
  gsl_vector * initial_condition_xi_eta_reflected_;
};
