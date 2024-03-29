// the basis needs to give us the mass matrix, the system matrices, as
// well as functions which can be evaluated. Each Basis class needs to
// supply a mass and system matrix, as well as the functions used to
// create those matrices.

extern "C" {
#include "igraph.h"
}
#include "BasisTypes.hpp"

class BivariateImageWithTime
{
public:
  BivariateImageWithTime();
  BivariateImageWithTime(const BivariateImageWithTime& image_with_time);
  virtual BivariateImageWithTime& operator=(const BivariateImageWithTime& rhs);

  BivariateImageWithTime(const gsl_vector* position,
			 double time,
			 double mult_factor);

  BivariateImageWithTime(const gsl_vector* position,
			 double time,
			 double mult_factor,
			 const std::vector<unsigned>& refl);

  virtual ~BivariateImageWithTime();

  inline const gsl_vector * get_position() const {
    return position_;
  }
  inline void set_position(const gsl_vector* position) {
    gsl_vector_memcpy(position_, position);
  }
  
  inline double get_t() const {
    return t_;
  }
  inline void set_t(double t) {
    t_ = t;
  }

  inline double get_mult_factor() const {
    return mult_factor_;
  }
  inline void set_mult_factor(double mm) {
    mult_factor_ = mm;
  }

  inline std::vector<unsigned> get_reflection_sequence() const {
    return reflection_sequence_;
  }
  inline void set_reflection_sequence(const std::vector<unsigned>& refl) {
    reflection_sequence_ = refl;
  }
  
private:
  gsl_vector * position_;
  double t_;
  double mult_factor_;
  std::vector<unsigned> reflection_sequence_;
};

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

  void set_data_for_small_t(double a,
			    double x_0,
			    double b,
			    double c,
			    double y_0,
			    double d);

  void set_diffusion_parameters_and_data(double sigma_x,
					 double sigma_y,
					 double rho,
					 double t,
					 double a,
					 double x_0,
					 double b,
					 double c,
					 double y_0,
					 double d);
  void set_diffusion_parameters_and_data_small_t(double sigma_x,
						 double sigma_y,
						 double rho,
						 double t,
						 double a,
						 double x_0,
						 double b,
						 double c,
						 double y_0,
						 double d);
  inline void set_x_t_2(double x)
  { x_t_2_ = x; };
  inline void set_y_t_2(double y)
  { y_t_2_ = y; };
  
  const gsl_vector* get_solution_coefs() const;
  const gsl_vector* get_ic_coefs() const;
  const gsl_vector* get_evals() const;

  inline double get_a_2() const
  { return a_2_; };
  inline double get_b_2() const
  { return b_2_; };
  inline double get_c_2() const
  { return c_2_; };
  inline double get_d_2() const
  { return d_2_; };
  inline double get_sigma_x_2() const
  { return sigma_x_2_; };
  inline double get_sigma_y_2() const
  { return sigma_y_2_; };

  inline double get_sigma_x() const
  { return sigma_x_; };
  inline double get_sigma_y() const
  { return sigma_y_; };
  
  inline double get_rho() const
  { return rho_; };
  inline double get_x_0_2() const
  { return x_0_2_; };
  inline double get_y_0_2() const
  { return y_0_2_; };
  inline double get_x_t_2() const
  { return x_t_2_; };
  inline double get_y_t_2() const
  { return y_t_2_; };
  inline double get_t_2() const
  { return t_2_; };

  inline bool get_flipped_xy_flag() const
  { return flipped_xy_flag_; };
  
  virtual double operator()(const gsl_vector* input) const;
  double analytic_solution(const gsl_vector* input) const;
  double numerical_solution_small_t(const gsl_vector* input) const;
  
  virtual double numerical_likelihood_second_order(const gsl_vector* input, double h);
  virtual double numerical_likelihood_first_order_small_t(const gsl_vector* input,
							  double small_t,
							  double h);
  virtual double numerical_likelihood_first_order_small_t_ax(const gsl_vector* input,
							     double small_t,
							     double h);
  virtual double numerical_likelihood_first_order_small_t_ax_bx(const gsl_vector* input,
								double h);
  virtual double numerical_likelihood_first_order_small_t_ax_bx_ay(const gsl_vector* input,
								double small_t,
								double h);
  virtual double numerical_likelihood_first_order_small_t_ax_bx_ay_by_type_41(const gsl_vector* input,
									      double small_t,
									      double h);

  virtual double likelihood_small_t_type_41_truncated(const gsl_vector* input,
						      double small_t,
						      double h);
  virtual double likelihood_small_t_type_31_truncated(const gsl_vector* input,
						      double small_t,
						      double h);
  
  
  virtual double likelihood_small_t_41_truncated_symmetric(const gsl_vector* input,
							   double small_t,
							   double h);

  virtual double likelihood_small_t_type_4_truncated(const gsl_vector* input,
						     double small_t,
						     double h);
  
  std::vector<BivariateImageWithTime> small_t_image_positions() const;
  // 202 131
  std::vector<BivariateImageWithTime> small_t_image_positions_type_1(bool PRINT) const;
  // 202 313
  std::vector<BivariateImageWithTime> small_t_image_positions_type_2(bool PRINT) const;
  // 131 020
  std::vector<BivariateImageWithTime> small_t_image_positions_type_3(bool PRINT) const;
  // 131 202
  std::vector<BivariateImageWithTime> small_t_image_positions_type_4(bool PRINT) const;
  // 313 020
  std::vector<BivariateImageWithTime> small_t_image_positions_type_31(bool PRINT) const;
  // 313 202 
  std::vector<BivariateImageWithTime> small_t_image_positions_type_41(bool PRINT) const;
  std::vector<BivariateImageWithTime> small_t_image_positions_type_41_all(bool PRINT) const;
  std::vector<BivariateImageWithTime> small_t_image_positions_type_41_symmetric(bool PRINT) const;

  std::vector<double> dPdax(const gsl_vector* raw_input,
			    double h);
  std::vector<double> dPdbx(const gsl_vector* raw_input,
			    double h);
  std::vector<double> dPday(const gsl_vector* raw_input,
			    double h);
  std::vector<double> dPdby(const gsl_vector* raw_input,
			    double h);

  std::vector<BivariateImageWithTime> small_t_image_positions_1_3(bool PRINT) const;
  
  std::vector<BivariateImageWithTime> small_t_image_positions_ax() const;
  std::vector<BivariateImageWithTime> small_t_image_positions_ax_bx() const;
  std::vector<BivariateImageWithTime> small_t_image_positions_ax_bx_ay() const;
  
  virtual double numerical_likelihood_first_order(const gsl_vector* input, double h);
  virtual double numerical_likelihood(const gsl_vector* input, double h);
  virtual double numerical_likelihood_extended(const gsl_vector* input, double h);
  // only valid for rho = 0
  virtual double analytic_likelihood(const gsl_vector* input, int little_n);
  virtual double analytic_likelihood_ax(const gsl_vector* input, int little_n);
  virtual double analytic_likelihood_ax_bx(const gsl_vector* input, int little_n);
  virtual double analytic_likelihood_ax_bx_ay(const gsl_vector* input,
					      int little_n);
  inline const BivariateSolverClassical* get_small_t_solution() const
  {
    return small_t_solution_;
  }

  void figure_chapter_3_proof_1() const;
  void figure_chapter_3_proof_2() const;
  void figure_chapter_3_illustration_1() const;

  static double wrapper(const std::vector<double> &x, 
  			std::vector<double> &grad,
  			void * data);
  static double wrapper_small_t(const std::vector<double> &x, 
				std::vector<double> &grad,
				void * data);
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
  double t_2_;
  bool flipped_xy_flag_;

  double x_t_2_;
  double y_t_2_;

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
  double scale_back_t(double t_2) const;
  void set_IC_coefs();
  // requires sigma_x_, sigma_y_, rho_;
  void set_mass_and_stiffness_matrices();
  // requires mass_matrix_ and stiffness_matrix_;
  void set_eval_and_evec();
  // requires t_, eval_, evec_;
  void set_solution_coefs();

  double extrapolate_t_direction(const double likelihood_upper_bound,
				 const double t_lower_bound,
				 const double t_2_current,
				 const double t_current,
				 const bool flipped_xy_flag,
				 const gsl_vector* input,
				 const double h);

  double extrapolate_sigma_y_direction(const double likelihood_upper_bound,
				       const double sigma_y_2_lower_bound_in,
				       const double sigma_y_2_current,
				       const double sigma_x_current,
				       const double sigma_y_current,
				       const bool flipped_xy_flag,
				       const gsl_vector* input,
				       const double h);
  void print_diffusion_params() const;
  // gsl_vector * xi_eta_input_;
  // gsl_vector * initial_condition_xi_eta_;
  // gsl_matrix * Rotation_matrix_;
  // gsl_matrix * Variance_;
  // gsl_vector * initial_condition_xi_eta_reflected_;
};
