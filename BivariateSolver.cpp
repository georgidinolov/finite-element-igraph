#include <algorithm>
#include "BivariateSolver.hpp"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <iostream>

BivariateSolver::BivariateSolver(const BivariateBasis& basis,
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
				 double dx)
  : a_(a),
    b_(b),
    c_(c),
    d_(d),
    mvtnorm_(MultivariateNormal()),
    basis_(basis),
    small_t_solution_(new BivariateSolverClassical()),
    t_(t),
    dx_(dx),
    IC_coefs_(gsl_vector_alloc(basis.get_orthonormal_elements().size())),
    mass_matrix_(gsl_matrix_alloc(basis_.get_orthonormal_elements().size(),
				  basis_.get_orthonormal_elements().size())),
    stiffness_matrix_(gsl_matrix_alloc(basis_.get_orthonormal_elements().size(),
				       basis_.get_orthonormal_elements().size())),
    eval_(gsl_vector_alloc(basis_.get_orthonormal_elements().size())),
    evec_(gsl_matrix_alloc(basis_.get_orthonormal_elements().size(),
			   basis_.get_orthonormal_elements().size())),
    solution_coefs_(gsl_vector_alloc(basis.get_orthonormal_elements().size()))
{
  if (x_0_ < 0.0 || x_0_ > 1.0 || y_0_ < 0.0 || y_0_ > 1.0) {
    std::cout << "ERROR: IC out of range" << std::endl;
  }

  // STEP 1
  double x_0_1 = 0 - a;
  double b_1 = b - a;
  double a_1 = a - a;

  double y_0_1 = 0 - c;
  double c_1 = c - c;
  double d_1 = d - c;

  // STEP 2
  double Lx_2 = b_1 - a_1;
  double x_0_2 =  x_0_1 / Lx_2;
  double  a_2 = a_1 / Lx_2;
  double b_2 = b_1 / Lx_2;
  double sigma_x_2 = sigma_x / Lx_2;

  double Ly_2 = d_1 - c_1;
  double y_0_2 =  y_0_1 / Ly_2;
  double c_2 = c_1 / Ly_2;
  double d_2 = d_1 / Ly_2;
  double sigma_y_2 = sigma_y / Ly_2;

  sigma_x_ = sigma_x_2;
  sigma_y_ = sigma_y_2;
  rho_ = rho;
  x_0_ = x_0_2;
  y_0_ = y_0_2;

  delete small_t_solution_;
  small_t_solution_ = new BivariateSolverClassical(sigma_x_,
						   sigma_y_,
						   rho_,
						   x_0_,
						   y_0_),
  
  std::cout << "small tt = " << small_t_solution_->get_t() << std::endl;
  small_t_solution_->set_function_grid(dx_);

  set_IC_coefs();
  set_mass_and_stiffness_matrices();
  set_eval_and_evec();
  set_solution_coefs();
}

BivariateSolver::~BivariateSolver()
{
  delete small_t_solution_;
  
  // freeing vectors
  gsl_vector_free(solution_coefs_);
  
  // freeing matrices
  gsl_matrix_free(mass_matrix_);
  gsl_matrix_free(stiffness_matrix_);
}

void BivariateSolver::set_t(double t)
{
  t_ = t;
  set_solution_coefs();
}

void BivariateSolver::set_diffusion_parameters(double sigma_x,
					       double sigma_y,
					       double rho)
{
  sigma_x_ = sigma_x;
  sigma_y_ = sigma_y;
  rho_ = rho;

  delete small_t_solution_;
  small_t_solution_ = new BivariateSolverClassical(sigma_x_,
						   sigma_y_,
						   rho_,
						   x_0_,
						   y_0_);
  set_mass_and_stiffness_matrices();
  set_eval_and_evec();
  set_solution_coefs();
}

double BivariateSolver::
operator()(const gsl_vector* input) const
{
  double x = gsl_vector_get(input,0);
  double y = gsl_vector_get(input,1);

  // STEP 1
  double x_1 = x - a_;
  double y_1 = y - c_;

  double a_1 = a_ - a_;
  double b_1 = b_ - a_;
  double c_1 = c_ - c_;
  double d_1 = d_ - c_;
  
  // STEP 2
  double Lx_2 = b_1 - a_1;
  double x_2 =  x_1 / Lx_2;

  double Ly_2 = d_1 - c_1;
  double y_2 =  y_1 / Ly_2;

  gsl_vector* scaled_input = gsl_vector_alloc(2);
  gsl_vector_set(scaled_input, 0, x_2);
  gsl_vector_set(scaled_input, 1, y_2);  

  double out = 0;
  if ((t_ - small_t_solution_->get_t()) <= 1e-32) {
    out = (*small_t_solution_)(scaled_input, t_);
  } else {
    int x_int = x_2/dx_;
    int y_int = y_2/dx_;

    for (unsigned i=0; i<basis_.get_orthonormal_elements().size(); ++i) {
      out = out + gsl_vector_get(solution_coefs_, i)*
	(gsl_matrix_get(basis_.get_orthonormal_element(i).get_function_grid(),
			x_int,
			y_int));
    }
  }

  out = out / (Lx_2 * Ly_2);
  return out;

  gsl_vector_free(scaled_input);
}

void BivariateSolver::set_IC_coefs()
{
  // Assigning coefficients for IC
  for (unsigned i=0; i<IC_coefs_->size; ++i) {
    gsl_vector_set(IC_coefs_, i,
		   basis_.project(*small_t_solution_,
				  basis_.get_orthonormal_element(i)));
  }
}

void BivariateSolver::set_mass_and_stiffness_matrices()
{
  // Mass matrix
  gsl_matrix_memcpy(mass_matrix_, basis_.get_mass_matrix());

  // Stiffness matrix
  const gsl_matrix* system_matrix_dx_dx = basis_.get_system_matrix_dx_dx();
  const gsl_matrix* system_matrix_dy_dy = basis_.get_system_matrix_dy_dy();
  const gsl_matrix* system_matrix_dx_dy = basis_.get_system_matrix_dx_dy();
  const gsl_matrix* system_matrix_dy_dx = basis_.get_system_matrix_dy_dx();

  double in = 0;
  for (unsigned i=0; i<basis_.get_orthonormal_elements().size(); ++i) {
    for (unsigned j=0; j<basis_.get_orthonormal_elements().size(); ++j) {
      in = -0.5*std::pow(sigma_x_,2)*gsl_matrix_get(system_matrix_dx_dx, i, j)
	+ -rho_*sigma_x_*sigma_y_*0.5*(gsl_matrix_get(system_matrix_dx_dy, i, j)+
				       gsl_matrix_get(system_matrix_dy_dx, i, j))
	+ -0.5*std::pow(sigma_y_,2)*gsl_matrix_get(system_matrix_dy_dy, i, j);
      gsl_matrix_set(stiffness_matrix_, i, j, in);
    }
  } 
}

void BivariateSolver::set_eval_and_evec()
{
  // System matrix
  int s = 0;
  unsigned K = basis_.get_orthonormal_elements().size();

  gsl_permutation * p = gsl_permutation_alloc(K);
  gsl_matrix* system_matrix = gsl_matrix_alloc(K,K);
  gsl_matrix* mass_matrix_inv = gsl_matrix_alloc(K,K);
  gsl_matrix* exp_system_matrix = gsl_matrix_alloc(K,K);

  gsl_linalg_LU_decomp(mass_matrix_, p, &s);
  gsl_linalg_LU_invert(mass_matrix_, p, mass_matrix_inv);

  gsl_blas_dsymm(CblasLeft, CblasUpper, 1.0,
		 mass_matrix_inv,
		 stiffness_matrix_,
		 0.0,
		 system_matrix);

  gsl_matrix *evec_tr = gsl_matrix_alloc(K, K);

  gsl_eigen_symmv_workspace * w = 
    gsl_eigen_symmv_alloc(K);
  
  gsl_eigen_symmv(system_matrix, eval_, evec_, w);

  gsl_eigen_symmv_sort(eval_, evec_, 
		       GSL_EIGEN_SORT_ABS_ASC);

  gsl_permutation_free(p);
  gsl_matrix_free(system_matrix);
  gsl_matrix_free(mass_matrix_inv);
  gsl_matrix_free(exp_system_matrix);
  gsl_matrix_free(evec_tr);
  gsl_eigen_symmv_free(w);
}

void BivariateSolver::set_solution_coefs()
{
  unsigned K = basis_.get_orthonormal_elements().size();  
  gsl_matrix* evec =gsl_matrix_alloc(K,K);
  gsl_matrix* evec_tr =gsl_matrix_alloc(K,K);
  gsl_matrix* exp_system_matrix =gsl_matrix_alloc(K,K);
  
  gsl_matrix_memcpy(evec, evec_);
  gsl_matrix_transpose_memcpy(evec_tr, evec_);

  // evec %*% diag(eval)
  for (unsigned i=0; i<K; ++i) {
    gsl_vector_view col_i = gsl_matrix_column(evec, i);
    gsl_vector_scale(&col_i.vector, std::exp(gsl_vector_get(eval_, i)*
					     (t_-small_t_solution_->get_t())));
  }
  // exp_system_matrix = [evec %*% diag(exp(eval*(t-t_small)))] %*% t(evec)
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,
		 evec, evec_tr,
		 0.0,
		 exp_system_matrix);
  // exp_system_matrix %*% ic_coefs_
  gsl_blas_dsymv(CblasUpper, 1.0, exp_system_matrix, IC_coefs_,
		 0.0, solution_coefs_);
  
  gsl_matrix_free(evec);
  gsl_matrix_free(evec_tr);
  gsl_matrix_free(exp_system_matrix);
}
