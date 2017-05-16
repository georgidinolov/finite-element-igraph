// the basis needs to give us the mass matrix, the system matrix, as
// well as functions which can be evaluated. Each Basis class needs to
// supply a mass and system matrix, as well as the functions used to
// create those matrices.

extern "C" {
#include "igraph.h"
}

#include "MultivariateNormal.hpp"
#include <iostream>
#include <string>
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
  virtual double get_dx() const=0;
};

// ============== BIVARIATE ELEMENT INTERFACE CLASS =============
// Abstract class implementation of function calls appropriate for a
// bivariate basis element, such as:
// a matrix containing the function values at node points. 
class BivariateElement
  : virtual public BasisElement
{
public:
  BivariateElement() {};
  virtual void save_function_grid(std::string file_name) const;

  virtual const gsl_matrix* get_function_grid() const =0;
  virtual const gsl_matrix* get_deriv_function_grid_dx() const =0;
  virtual const gsl_matrix* get_deriv_function_grid_dy() const =0;

  // Implemented with planar interpolation
  virtual double operator()(const gsl_vector* input) const;
};

// ============== FOURIER INTERPOLANT INTERFACE CLASS =============
// Abstract class implementation of function calls appropriate for a
// fourier interpolant element, such as: a matrix containing FFT of
// the function values of the bivariate function
class BivariateFourierInterpolant
  : virtual public BivariateElement
{
public:
  BivariateFourierInterpolant() {};

  // If the size of the grid of function values is n x n, then the FFT
  // grid is of size 2n x n, where there are 2n rows in order to
  // accomodate the real and imaginary parts of the FFT coefs.
  virtual const gsl_matrix* get_FFT_grid() const =0;
  virtual const gsl_matrix* get_deriv_FFT_grid_dx() const =0;
  virtual const gsl_matrix* get_deriv_FFT_grid_dy() const =0;

  // Implemented with Fouriera interpolation. The interpolation
  // follows the GSL convention.
  virtual double operator()(const gsl_vector* input) const;
};

// ============== LINEAR COMBINATION ELEMENT =====================
class BivariateLinearCombinationElement
  : public virtual BivariateElement
{
public:
  BivariateLinearCombinationElement();
  BivariateLinearCombinationElement(const std::vector<const BivariateElement*>& elements,
				    const std::vector<double>& coefficients);
  BivariateLinearCombinationElement(const BivariateLinearCombinationElement& lin_comb_element);
  BivariateLinearCombinationElement& operator=(const BivariateLinearCombinationElement& rhs);

  virtual ~BivariateLinearCombinationElement();

  virtual double norm() const;
  virtual double first_derivative(const gsl_vector* input,
				  long int coord_index) const;

  // PUBLIC GETTERS
  inline virtual double get_dx() const { return dx_; }
  inline virtual const gsl_matrix* get_function_grid() const
  { return function_grid_; };
  inline virtual const gsl_matrix* get_deriv_function_grid_dx() const
  { return deriv_function_grid_dx_; }
  inline virtual const gsl_matrix* get_deriv_function_grid_dy() const
  { return deriv_function_grid_dy_; }

  // PUBLIC SETTERS
  // WARNING: Call below function ONLY if you are sure the
  // function_grid_ is in agreement with the elements_ and
  // coefficients_.
  virtual void set_function_grid(const gsl_matrix* new_function_grid);
  // WARNING: Call below function ONLY if you are sure the
  // deriv_function_grid_dx_ is in agreement with the elements_ and
  // coefficients_.
  virtual void set_deriv_function_grid_dx(const gsl_matrix* new_deriv_function_grid_dx);
  // WARNING: Call below function ONLY if you are sure the
  // deriv_function_grid_dy_ is in agreement with the elements_ and
  // coefficients_.
  virtual void set_deriv_function_grid_dy(const gsl_matrix* new_deriv_function_grid_dy);

private:
  void set_function_grids(const std::vector<const BivariateElement*>& elements,
			  const std::vector<double>& coefficients);

  double dx_;
  gsl_matrix * function_grid_;
  gsl_matrix * deriv_function_grid_dx_;
  gsl_matrix * deriv_function_grid_dy_;
};


// ============== LINEAR COMBINATION ELEMENT FOURIER =====================
class BivariateLinearCombinationElementFourier
  : public BivariateLinearCombinationElement,
    public virtual BivariateFourierInterpolant
{
public:
  BivariateLinearCombinationElementFourier();
  //BivariateLinearCombinationElementFourier(const std::vector<const BivariateElement*>& elements,
  // 					   const std::vector<double>& coefficients);
  BivariateLinearCombinationElementFourier(const BivariateLinearCombinationElementFourier& lin_comb_element);
  BivariateLinearCombinationElementFourier(const BivariateLinearCombinationElement& element);
  
  BivariateLinearCombinationElementFourier& operator=(const BivariateLinearCombinationElementFourier& rhs);

  ~BivariateLinearCombinationElementFourier();

  void set_function_grid(const gsl_matrix* new_function_grid);

  inline const gsl_matrix* get_FFT_grid() const
  { return FFT_grid_; }
  void set_FFT_grid();

  inline const gsl_matrix* get_deriv_FFT_grid_dx() const
  { return NULL; }
  inline const gsl_matrix* get_deriv_FFT_grid_dy() const
  { return NULL; }

  void save_FFT_grid(std::string name) const;

private:
  gsl_matrix * FFT_grid_;

};

// ============== GAUSSIAN KERNEL ELEMENT =====================
class GaussianKernelElement
  : public virtual BasisElement
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


// ===================== BIVARIATE GAUSSIAN KERNEL ELEMENT ===================== 
class BivariateGaussianKernelElement
  : public virtual GaussianKernelElement,
    public virtual BivariateElement
{
public:
  BivariateGaussianKernelElement();
  BivariateGaussianKernelElement(double dx,
				 double exponent_power,
				 const gsl_vector* mean_vector,
				 const gsl_matrix* covariance_matrix);
  BivariateGaussianKernelElement(const BivariateGaussianKernelElement& element);
  virtual BivariateGaussianKernelElement& operator=(const BivariateGaussianKernelElement& rhs);
  virtual ~BivariateGaussianKernelElement();


  virtual double operator()(const gsl_vector* input) const;
  virtual double norm() const;
  virtual double first_derivative(const gsl_vector* input,
				  long int coord_index) const;
  virtual double get_dx() const;

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
  : public virtual BivariateElement
{
public:
  BivariateSolverClassical();
  BivariateSolverClassical(double sigma_x,
			   double sigma_y,
			   double rho,
			   double x_0,
			   double y_0);
  BivariateSolverClassical(const BivariateSolverClassical& biv_sol_class);
  virtual BivariateSolverClassical& operator=(const BivariateSolverClassical& rhs);
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

  virtual const gsl_matrix* get_function_grid() const;
  virtual void set_function_grid(double dx);

  // TODO(georgid): THIS NEEDS TO BE IMPLEMETED
  inline virtual const gsl_matrix* get_deriv_function_grid_dx() const
  { return NULL; }
  virtual const gsl_matrix* get_deriv_function_grid_dy() const 
  { return NULL; }
  
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

  gsl_matrix * function_grid_;
};
