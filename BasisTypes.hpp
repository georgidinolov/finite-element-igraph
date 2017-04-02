// the basis needs to give us the mass matrix, the system matrix, as
// well as functions which can be evaluated. Each Basis class needs to
// supply a mass and system matrix, as well as the functions used to
// create those matrices.

extern "C" {
#include "igraph.h"
}
#include "BasisElementTypes.hpp"
#include <vector>
  
// =================== BASE BASIS CLASS ======================
class BivariateBasis
{
public:
  virtual ~BivariateBasis() =0;
  virtual const gsl_matrix* get_mass_matrix() const =0;
  virtual const gsl_matrix* get_system_matrix_dx_dx() const =0;
  virtual const gsl_matrix* get_system_matrix_dx_dy() const =0;
  virtual const gsl_matrix* get_system_matrix_dy_dx() const =0;
  virtual const gsl_matrix* get_system_matrix_dy_dy() const =0;
  virtual const BivariateLinearCombinationElement& get_orthonormal_element(unsigned i) const=0;

  virtual double project(const BivariateElement& elem_1,
			 const BivariateElement& elem_2) const =0;
};


// ============== GAUSSIAN KERNEL BASIS CLASS ==============
class BivariateGaussianKernelBasis
  : public BivariateBasis
{
public:
  BivariateGaussianKernelBasis(double dx,
			       double rho,
			       double sigma,
			       double power,
			       double std_dev_factor);
  ~BivariateGaussianKernelBasis();

  virtual const BivariateGaussianKernelElement& get_basis_element(unsigned i) const;
  virtual const BivariateLinearCombinationElement& get_orthonormal_element(unsigned i) const;
  virtual const std::vector<BivariateLinearCombinationElement>&
  get_orthonormal_elements() const;
  
  virtual const gsl_matrix* get_mass_matrix() const;
  virtual const gsl_matrix* get_system_matrix_dx_dx() const;
  virtual const gsl_matrix* get_system_matrix_dx_dy() const;
  virtual const gsl_matrix* get_system_matrix_dy_dx() const;
  virtual const gsl_matrix* get_system_matrix_dy_dy() const;
  
  virtual double project(const BivariateElement& elem_1,
			 const BivariateElement& elem_2) const;

  // TODO(georgi): THIS NEEDS FASTER, SYMBOLIC IMPLEMENTATION
  virtual double project(const BivariateGaussianKernelElement& g_elem_1,
			 const BivariateGaussianKernelElement& g_elem_2) const;

  // coord_indeex_{1,2} = {0,1}, where 0 = dx, 1 = dy
  // TODO(georgi) : this needs to be done with enumerable elements instead of ints
  virtual double project_deriv(const BivariateElement& elem_1,
			       int coord_indeex_1, 
			       const BivariateElement& elem_2,
			       int coord_indeex_2) const;

  virtual double project_deriv_analytic(const BivariateElement& elem_1,
					long int coord_indeex_1, 
					const BivariateElement& elem_2,
					long int coord_indeex_2) const;
  
  
private:
  double dx_;
  // sets basis functions in the class
  void set_basis_functions(double rho,
			   double sigma,
			   double power,
			   double std_dev_factor);
  
  void set_orthonormal_functions();
  void set_orthonormal_functions_stable();
  void set_mass_matrix();

  void set_system_matrices();
  void set_system_matrices_stable();
  
  std::vector<BivariateGaussianKernelElement> basis_functions_;
  std::vector<BivariateLinearCombinationElement> orthonormal_functions_;

  gsl_matrix* system_matrix_dx_dx_;
  gsl_matrix* system_matrix_dx_dy_;
  gsl_matrix* system_matrix_dy_dx_;
  gsl_matrix* system_matrix_dy_dy_;
  
  gsl_matrix* mass_matrix_;
  gsl_matrix* inner_product_matrix_;

  gsl_matrix* deriv_inner_product_matrix_dx_dx_;
  gsl_matrix* deriv_inner_product_matrix_dx_dy_;
  gsl_matrix* deriv_inner_product_matrix_dy_dx_;
  gsl_matrix* deriv_inner_product_matrix_dy_dy_;
};
