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
class BaseBasis
{
public:
  virtual ~BaseBasis() =0;

  virtual const igraph_matrix_t& get_mass_matrix() const =0;
  virtual const igraph_matrix_t& get_system_matrix() const =0;
};


// ============== GAUSSIAN KERNEL BASIS CLASS ==============
class BivariateGaussianKernelBasis
  : public BaseBasis
{
public:
  BivariateGaussianKernelBasis(double dx,
			       double rho,
			       double sigma,
			       double power,
			       double std_dev_factor);
  ~BivariateGaussianKernelBasis();

  const LinearCombinationElement& get_orthonormal_element(unsigned i) const;
  
  virtual const igraph_matrix_t& get_mass_matrix() const;
  virtual const igraph_matrix_t& get_system_matrix() const;
  
  virtual double project(const BasisElement& elem_1,
			 const BasisElement& elem_2) const;
private:
  double dx_;
  // sets basis functions in the class
  void set_basis_functions(double rho,
			   double sigma,
			   double power,
			   double std_dev_factor);
  
  void set_orthonormal_functions();
  void set_mass_matrix();
  
  std::vector<GaussianKernelElement> basis_functions_;
  std::vector<LinearCombinationElement> orthonormal_functions_;
  igraph_matrix_t system_matrix_;
  igraph_matrix_t mass_matrix_;
  igraph_matrix_t inner_product_matrix_;
};
