#include "2DAdvectionDiffusionSolverImages.hpp"
#include "2DBrownianMotionPath.hpp"
#include "BivariateSolver.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>

int main() {
  std::cout << std::fixed << std::setprecision(32);
  unsigned order = 1e6;
  double sigma_x_data_gen = 1.0;
  double sigma_y_data_gen = 0.25;
  double rho_data_gen = 0.1;
  double x_0 = 0.0;
  double y_0 = 0.0;
  double t = 1;
  long unsigned seed = 2000;

  BrownianMotion BM = BrownianMotion(seed,
				     order,
				     rho_data_gen,
				     sigma_x_data_gen,
				     sigma_y_data_gen,
				     x_0,
				     y_0,
				     t);

  std::cout << "ax = " << BM.get_a() << std::endl;
  std::cout << "x_T = " << BM.get_x_T() << std::endl;
  std::cout << "bx = " << BM.get_b() << std::endl;

  std::cout << "ay = " << BM.get_c() << std::endl;
  std::cout << "y_T = " << BM.get_y_T() << std::endl;
  std::cout << "by = " << BM.get_d() << std::endl;

  double dx = 5e-3;
  BivariateGaussianKernelBasis* basis = new BivariateGaussianKernelBasis(dx,
  								    rho_data_gen,
  								    0.30,
  								    1,
  								    0.5);
  BivariateSolver FEM_solver = BivariateSolver(*basis,
  					       sigma_x_data_gen, 
  					       sigma_y_data_gen,
					       rho_data_gen,
  					       BM.get_a(),
					       BM.get_x_0(),
					       BM.get_b(),
					       BM.get_c(),
					       BM.get_y_0(),
					       BM.get_d(),
					       t,
  					       dx);

  int x_index = (BM.get_x_T() - BM.get_a())/dx;
  int y_index = (BM.get_y_T() - BM.get_c())/dx;
  std::cout << "x_index = " << x_index << "; y_index = " << y_index << std::endl;
  
  TwoDAdvectionDiffusionSolverImages MOI_solver =
    TwoDAdvectionDiffusionSolverImages(0,
				       0,
				       sigma_x_data_gen,
				       sigma_y_data_gen,
				       100,
				       BM.get_x_0(),
				       BM.get_y_0(),
				       BM.get_a(),
				       BM.get_b(),
				       BM.get_c(),
				       BM.get_d());

  unsigned N = 1.0/dx + 1;
  gsl_vector* input = gsl_vector_alloc(2);
  gsl_vector_set(input, 0, BM.get_x_T());
  gsl_vector_set(input, 1, BM.get_y_T());

  std::cout << "FEM_solver(input) = " 
  	    << FEM_solver(input) << std::endl;
  std::cout << "MOI_solver.solve(input) = " 
	    << MOI_solver.solve(t,
				BM.get_a() + x_index*dx,
				BM.get_c() + y_index*dx) << std::endl;
  std::cout << std::endl;

  std::cout << "FEM_solver(input - da) = ";
  FEM_solver.set_data(BM.get_a() - dx,
		      BM.get_x_0(),
		      BM.get_b(),
		      BM.get_c(),
		      BM.get_y_0(),
		      BM.get_d());
  std::cout << FEM_solver(input) << std::endl;
  std::cout << "MOI_solver.solve(input - da) = " 
	    << MOI_solver.solve(t,
				BM.get_a() + x_index*dx,
				BM.get_c() + y_index*dx,
				BM.get_a()-dx, BM.get_b(),
				BM.get_c(), BM.get_d()) << std::endl;
  std::cout << std::endl;

  std::cout << "FEM_solver(input + db) = ";
  FEM_solver.set_data(BM.get_a(),
		      BM.get_x_0(),
		      BM.get_b() + dx,
		      BM.get_c(),
		      BM.get_y_0(),
		      BM.get_d());
  std::cout << FEM_solver(input) << std::endl;
  std::cout << "MOI_solver.solve(input + db) = " 
	    << MOI_solver.solve(t,
				BM.get_a() + x_index*dx,
				BM.get_c() + y_index*dx,
				BM.get_a(), BM.get_b() + dx,
				BM.get_c(), BM.get_d()) << std::endl;
  std::cout << std::endl;

  // NUMERICAL DERIVS A
  FEM_solver.set_data(BM.get_a() + dx,
		      BM.get_x_0(),
		      BM.get_b(),
		      BM.get_c(),
		      BM.get_y_0(),
		      BM.get_d());
  double FEM_solution_pda = FEM_solver(input);
  FEM_solver.set_data(BM.get_a() - dx,
		      BM.get_x_0(),
		      BM.get_b(),
		      BM.get_c(),
		      BM.get_y_0(),
		      BM.get_d());
  double FEM_solution_mda = FEM_solver(input);
  
  std::cout << "FEM da = "
	    << (FEM_solution_pda - FEM_solution_mda)/(2*dx) << std::endl;
  std::cout << "MOI da = "
	    << (MOI_solver.solve(t,
				BM.get_a() + x_index*dx,
				BM.get_c() + y_index*dx,
				BM.get_a()+dx, BM.get_b(),
				BM.get_c(), BM.get_d()) -
		MOI_solver.solve(t,
				BM.get_a() + x_index*dx,
				BM.get_c() + y_index*dx,
				BM.get_a()-dx, BM.get_b(),
				 BM.get_c(), BM.get_d()))/(2*dx)
	    << std::endl;
  std::cout << std::endl;

  // NUMERICAL DERIVS B
  FEM_solver.set_data(BM.get_a(),
		      BM.get_x_0(),
		      BM.get_b() + dx,
		      BM.get_c(),
		      BM.get_y_0(),
		      BM.get_d());
  double FEM_solution_pdb = FEM_solver(input);
  FEM_solver.set_data(BM.get_a(),
		      BM.get_x_0(),
		      BM.get_b() - dx,
		      BM.get_c(),
		      BM.get_y_0(),
		      BM.get_d());
  double FEM_solution_mdb = FEM_solver(input);
  std::cout << "FEM db = "
	    << (FEM_solution_pdb - FEM_solution_mdb)/(2*dx) << std::endl;
  std::cout << "MOI db = "
	    << (MOI_solver.solve(t,
				BM.get_a() + x_index*dx,
				BM.get_c() + y_index*dx,
				BM.get_a(), BM.get_b() + dx,
				BM.get_c(), BM.get_d()) -
		MOI_solver.solve(t,
				 BM.get_a() + x_index*dx,
				 BM.get_c() + y_index*dx,
				 BM.get_a(), BM.get_b(),
				 BM.get_c(), BM.get_d()))/dx
	    << std::endl;
  std::cout << std::endl;

  // NUMERICAL DERIVS A,B
  FEM_solver.set_data(BM.get_a() + dx,
		      BM.get_x_0(),
		      BM.get_b() + dx,
		      BM.get_c(),
		      BM.get_y_0(),
		      BM.get_d());
  double FEM_solution_papb = FEM_solver(input);
  FEM_solver.set_data(BM.get_a() + dx,
		      BM.get_x_0(),
		      BM.get_b() - dx,
		      BM.get_c(),
		      BM.get_y_0(),
		      BM.get_d());
  double FEM_solution_pamb = FEM_solver(input);
  FEM_solver.set_data(BM.get_a() - dx,
		      BM.get_x_0(),
		      BM.get_b() + dx,
		      BM.get_c(),
		      BM.get_y_0(),
		      BM.get_d());
  double FEM_solution_mapb = FEM_solver(input);
  FEM_solver.set_data(BM.get_a() - dx,
		      BM.get_x_0(),
		      BM.get_b() - dx,
		      BM.get_c(),
		      BM.get_y_0(),
		      BM.get_d());
  double FEM_solution_mamb = FEM_solver(input);


  FEM_solver.set_data(BM.get_a(),
		      BM.get_x_0(),
		      BM.get_b(),
		      BM.get_c(),
		      BM.get_y_0(),
		      BM.get_d());
  
  std::cout << "FEM.numerical_likelihood(input,dx) = "
	    << FEM_solver.numerical_likelihood_first_order(input, dx)
	    << std::endl;
  
  std::cout << "MOI / dadb = "
	    << (-MOI_solver.solve(t,
				  BM.get_a() + x_index*dx,
				  BM.get_c() + y_index*dx,
				  BM.get_a(), BM.get_b() + dx,
				  BM.get_c(), BM.get_d()) +
		MOI_solver.solve(t,
				 BM.get_a() + x_index*dx,
				 BM.get_c() + y_index*dx,
				 BM.get_a(), BM.get_b(),
				 BM.get_c(), BM.get_d()) +
		MOI_solver.solve(t,
				 BM.get_a() + x_index*dx,
				 BM.get_c() + y_index*dx,
				 BM.get_a() - dx, BM.get_b() + dx,
				 BM.get_c(), BM.get_d()) -
		MOI_solver.solve(t,
				 BM.get_a() + x_index*dx,
				 BM.get_c() + y_index*dx,
				 BM.get_a() - dx, BM.get_b(),
				 BM.get_c(), BM.get_d()))/(dx*dx)
	    << std::endl;
    std::cout << "MOI / dadb analyic = "
	      << MOI_solver.likelihood(t, BM.get_x_T(), BM.get_y_T())
	      << std::endl;
    std::cout << std::endl;
  
  
  gsl_matrix* left = gsl_matrix_alloc(N,N);
  gsl_matrix* right = gsl_matrix_alloc(N,N);
  for (unsigned i=0; i<basis->get_orthonormal_elements().size(); ++i)
    {
      if (i==0)
	{
	  gsl_matrix_memcpy(left, basis->get_orthonormal_element(i).
			    get_function_grid());
	  gsl_matrix_scale(left, gsl_vector_get(FEM_solver.get_solution_coefs(),
						i));
	}
      else
	{
	  gsl_matrix_memcpy(right, basis->get_orthonormal_element(i).
			    get_function_grid());
	  gsl_matrix_scale(right, gsl_vector_get(FEM_solver.get_solution_coefs(),
						 i));
	  gsl_matrix_add(left, right);
	}
    }

  basis->save_matrix(left, "bivariate-solution.csv");
  
  gsl_matrix_free(left);
  gsl_matrix_free(right);

  // LIKELIHOOD LOOP
  std::ofstream output_file;
  output_file.open("likelihood-rho.txt");
  output_file << std::fixed << std::setprecision(32);
  output_file << "log.likelihood\n";
  
  long unsigned seed_init = 2000;
  N = 100;
  // GENERATE DATA
  std::vector<BrownianMotion> BMs (0);
  for (unsigned i=0; i<N; ++i) {
    seed = seed_init + i;
    BM = BrownianMotion(seed,
			order,
			rho_data_gen,
			sigma_x_data_gen,
			sigma_y_data_gen,
			x_0,
			y_0,
			t);
    BMs.push_back(BM);
  }
  
  unsigned R = 7;
  double dr = 0.1;
  double rho_init = -0.3;
  for (unsigned r=0; r<R; ++r) {
    double rho = rho_init + dr*r;
    delete basis;
    basis = new BivariateGaussianKernelBasis(dx,
					     rho,
					     0.30,
					     1,
					     0.5);
    double log_likelihood = 0;
    for (unsigned i=0; i<N; ++i) {
      gsl_vector_set(input, 0, BMs[i].get_x_T());
      gsl_vector_set(input, 1, BMs[i].get_y_T());
      
      BivariateSolver FEM_solver_2 = BivariateSolver(*basis,
						     sigma_x_data_gen, 
						     sigma_y_data_gen,
						     rho,
						     BMs[i].get_a(),
						     BMs[i].get_x_0(),
						     BMs[i].get_b(),
						     BMs[i].get_c(),
						     BMs[i].get_y_0(),
						     BMs[i].get_d(),
						     t,
						     dx);
      
      MOI_solver =
  	TwoDAdvectionDiffusionSolverImages(0,
  					   0,
  					   sigma_x_data_gen,
  					   sigma_y_data_gen,
  					   100,
  					   BMs[i].get_x_0(),
  					   BMs[i].get_y_0(),
  					   BMs[i].get_a(),
  					   BMs[i].get_b(),
  					   BMs[i].get_c(),
  					   BMs[i].get_d());
      double FEM_likelihood = FEM_solver_2.
	numerical_likelihood_first_order(input, dx);
      
      std::cout << "i=" << i << "; ";
      std::cout << "FEM.numerical_likelihood(input,dx) = "
  		<< FEM_likelihood
  		<< "; MOI_solver /dadb analy = "
  		<< MOI_solver.likelihood(t,
  					 BMs[i].get_x_T(), BMs[i].get_y_T());
      std::cout << std::endl;

      if (FEM_likelihood > 1e-16) {
  	log_likelihood = log_likelihood + log(FEM_likelihood);
      }
    }
    output_file << log_likelihood << "\n";
  }

  gsl_vector_free(input);
  output_file.close();
  return 0;
}


