// the basis needs to give us the mass matrix, the system matrix, as
// well as functions which can be evaluated. Each Basis class needs to
// supply a mass and system matrix, as well as the functions used to
// create those matrices.

extern "C" {
#include "igraph.h"
}
#include "MultivariateNormal.hpp"
#include <vector>
#include <iostream>

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
  virtual double get_dx() const=0;
};

// ============== BIVARIATE ELEMENT INTERFACE CLASS =============

// Abstract class implementation of function calls appropriate for a
// bivariate basis element, such as:
// a matrix containing the function values at node points. 
class BivariateElement
{
public:
  BivariateElement() {};
  virtual ~BivariateElement() =0;

  virtual const gsl_matrix* get_function_grid() const=0;
}

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

    
  double get_coefficient(unsigned i) const;
  std::vector<double> get_coefficients() const;
  inline virtual double get_dx() const { return dx_; }
  const std::vector<const BasisElement*> get_elements() const;
  inline virtual const gsl_matrix* get_function_grid() const
  { return function_grid_; };
  
private:
  void set_function_grids();
  
  const std::vector<const BasisElement*> elements_;
  std::vector<double> coefficients_;

  double dx_;
  gsl_matrix * function_grid_;
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
  virtual GaussianKernelElement& operator=(const GaussianKernelElement& rhs);

  virtual double first_derivative(const gsl_vector* input,
				  long int coord_index) const;

  virtual double first_derivative_finite_diff(const gsl_vector* input,
					      long int coord_index) const;

  virtual double norm() const;
  virtual double norm_finite_diff() const;

  inline double get_ax() const { return ax_; }
  const gsl_matrix* get_covariance_matrix() const;
  inline long int get_dimension() const { return dimension_; }
  inline double get_dx() const { return dx_; }
  inline const double get_exponent_power() const { return exponent_power_; }
  // TODO: THIS NEEDS TO BE IMPLEMENTED
  inline virtual const gsl_matrix* get_function_grid() const { return 0; }
  const gsl_vector* get_mean_vector() const;
  inline int get_s() const { return s_; }
  inline const gsl_matrix * get_winv() const { return winv_; }
  
private:
  double dx_;
  long int dimension_;
  double exponent_power_;

  gsl_vector *mean_vector_;
  gsl_vector *input_gsl_;
  gsl_matrix *covariance_matrix_;
  
  MultivariateNormal mvtnorm_;

  int s_;
  double ax_;
  gsl_vector *ym_;
  gsl_matrix *work_; // = gsl_matrix_alloc(2,2);
  gsl_matrix *winv_; // = gsl_matrix_alloc(2,2);
  gsl_permutation *p_; // = gsl_permutation_alloc(2);
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
  BivariateGaussianKernelElement(const BivariateGaussianKernelElement& element);
  
  // BivariateGaussianKernelElement(const BivariateGaussianKernelElement& element);
  virtual ~BivariateGaussianKernelElement();
  virtual BivariateGaussianKernelElement& operator=(const BivariateGaussianKernelElement& rhs);

  inline virtual const gsl_matrix * get_function_grid() const
  { return function_grid_; }
  inline virtual const gsl_matrix * get_deriv_function_grid_dx() const
  { return deriv_function_grid_dx_; }
  inline virtual const gsl_matrix * get_deriv_function_grid_dy() const
  { return deriv_function_grid_dy_; }

private:
  void set_function_grid();
  void set_function_grid_dx();
  void set_function_grid_dy();
  void set_function_grids();
  
  gsl_matrix * function_grid_;
  gsl_matrix * deriv_function_grid_dx_;
  gsl_matrix * deriv_function_grid_dy_;
};

// ============== BIVARIATE CLASSICAL SOLVER ====================
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
  
  inline virtual double get_dx() const
  { return 0.0; }
  
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
