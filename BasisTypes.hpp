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
  BasisElement() {};
  virtual ~BasisElement() =0;
  virtual double operator()(const igraph_vector_t& input) const =0;
  virtual double norm() const =0;
};

// ============== LINEAR COMBINATION ELEMENT =====================
class LinearCombinationElement
  : public BasisElement
{
public:

  LinearCombinationElement(const std::vector<const BasisElement*> elements,
			   const std::vector<double>& coefficients);
  virtual ~LinearCombinationElement();
  virtual double operator()(const igraph_vector_t& input) const;
  virtual double norm() const;

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
  GaussianKernelElement(long unsigned dimension,
			double exponent_power,
			const igraph_vector_t& mean_vector,
			const igraph_matrix_t& covariance_matrix);
  
  GaussianKernelElement(const GaussianKernelElement& element);

  virtual ~GaussianKernelElement();
  virtual double operator()(const igraph_vector_t& input) const;

  virtual double first_derivative(const igraph_vector_t& input,
				  long int coord_index) const;

  virtual double norm() const;
  virtual double norm_finite_diff() const;

  const igraph_vector_t& get_mean_vector() const;
  const igraph_matrix_t& get_covariance_matrix() const;
  
private:
  long int dimension_;
  double exponent_power_;
  igraph_vector_t mean_vector_;
  igraph_matrix_t covariance_matrix_;
  MultivariateNormal mvtnorm_;

  void set_norm();
  double norm_;
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
class BivariateGaussianKernelBasis
  : public BaseBasis
{
public:
  BivariateGaussianKernelBasis(double rho,
		      double sigma,
		      double power,
		      double std_dev_factor);
  ~BivariateGaussianKernelBasis();

  virtual const igraph_matrix_t& get_mass_matrix() const;
  virtual const igraph_matrix_t& get_system_matrix() const;

private:
  // sets basis functions in the class
  void set_basis_functions(double rho,
			   double sigma,
			   double power,
			   double std_dev_factor);
  
  void set_orthonormal_functions();
  
  std::vector<GaussianKernelElement> basis_functions_;
  std::vector<LinearCombinationElement> orthonormal_functions_;
  igraph_matrix_t system_matrix_;
  igraph_matrix_t mass_matrix_;
};
