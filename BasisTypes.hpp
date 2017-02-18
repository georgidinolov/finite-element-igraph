// the basis needs to give us the mass matrix, the system matrix, as
// well as functions which can be evaluated. Each Basis class needs to
// supply a mass and system matrix, as well as the functions used to
// create those matrices.

extern "C" {
#include "igraph.h"
}
#include "MultivariateNormal.hpp"
#include <vector>

// =================== BASIS ELEMENT CLASS ===================
class BasisElement
{
public:
  virtual ~BasisElement() =0;
  virtual double operator()(const igraph_vector_t& input) const =0;
};

class GaussianKernelElement
  : public BasisElement
{
public:
  GaussianKernelElement();
  GaussianKernelElement(long unsigned dimension,
			double exponent_power,
			const igraph_vector_t& mean_vector,
			const igraph_matrix_t& covariance_matrix);
  
  GaussianKernelElement(const GaussianKernelElement& element);

  ~GaussianKernelElement();
  virtual double operator()(const igraph_vector_t& input) const;

  const igraph_vector_t& get_mean_vector() const;
  const igraph_matrix_t& get_covariance_matrix() const;
  
private:
  long int dimension_;
  double exponent_power_;
  igraph_vector_t mean_vector_;
  igraph_matrix_t covariance_matrix_;
  MultivariateNormal mvtnorm_;
};
  
// =================== BASE BASIS CLASS ======================
class BaseBasis
{
public:
  virtual ~BaseBasis() =0;

  virtual const igraph_matrix_t& get_mass_matrix() const =0;
  virtual const igraph_matrix_t& get_system_matrix() const =0;
};


// ============== GAUSSIAN KERNEL BASIS CLASS ==============
class GaussianKernelBasis
  : public BaseBasis
{
public:
  GaussianKernelBasis(double rho,
		      double sigma,
		      double power,
		      double std_dev_factor);
  ~GaussianKernelBasis();

  virtual const igraph_matrix_t& get_mass_matrix() const;
  virtual const igraph_matrix_t& get_system_matrix() const;

private:
  // sets basis functions in the class
  void set_basis_functions(double rho,
			   double sigma,
			   double power,
			   double std_dev_factor);
  
  std::vector<GaussianKernelElement> basis_functions_;
  igraph_matrix_t system_matrix_;
  igraph_matrix_t mass_matrix_;
};
