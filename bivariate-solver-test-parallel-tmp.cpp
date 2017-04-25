#include "2DBrownianMotionPath.hpp"
#include "BivariateSolver.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <string>
#include <vector>

int main() {
  std::cout << std::fixed << std::setprecision(32);

  omp_set_dynamic(0);
  unsigned order = 1e5;
  double sigma_x_data_gen = 1.0;
  double sigma_y_data_gen = 0.25;
  double rho_data_gen = 0.5;
  double x_0 = 0.0;
  double y_0 = 0.0;
  double t = 1;
  long unsigned seed_init = 1000;
  double dx = 1e-2;
  unsigned n_rhos = 2;
  gsl_vector* input = gsl_vector_alloc(2);

  // LIKELIHOOD LOOP
  std::ofstream output_file;
  output_file.open("likelihood-rho.txt");
  output_file << std::fixed << std::setprecision(32);
  output_file << "log.likelihood, rho\n";
  

  unsigned N = 120;
  // GENERATE DATA
  std::vector<BrownianMotion> BMs (0);
  for (unsigned i=0; i<N; ++i) {
    unsigned seed = seed_init + i;
    BrownianMotion BM = BrownianMotion(seed,
				       order,
				       rho_data_gen,
				       sigma_x_data_gen,
				       sigma_y_data_gen,
				       x_0,
				       y_0,
				       t);
    BMs.push_back(BM);
  }

  unsigned n_threads = omp_get_max_threads();
  std::cout << "number of available threads = " << omp_get_max_threads() << "\n";
  std::cout << "number of threads in use = " << omp_get_num_threads() << "\n";
  std::cout << "throwing bases on heap\n";
  std::vector<BivariateGaussianKernelBasis*> bases (n_rhos);
  double rho_min = 0.0;
  double rho_max = 0.8;
  std::vector<double> basis_rhos (n_rhos);
  double drho = (rho_max-rho_min)/(n_rhos-1);

  for (unsigned k=0; k<n_rhos; ++k) {
    double current_rho = rho_min + drho*k;
    if (current_rho <= rho_max && current_rho >= rho_min) {
      basis_rhos[k] = current_rho;
    } else if (current_rho > rho_max) {
      basis_rhos[k] = rho_max;
    } else if (current_rho < rho_min) {
      basis_rhos[k] = rho_min;
    } else {
      std::cout << "ERROR: rho cannot be set" << std::endl;
      basis_rhos[k] = rho_min + (rho_max-rho_min)/2.0;
    }
    
    std::cout << "on basis " << k << std::endl;
    bases[k] = new BivariateGaussianKernelBasis(dx, 
						basis_rhos[k],
						0.3,
						1, 
						0.5);
  }

  n_threads = 3;
  std::cout << "copying bases vectors for threads" << std::endl;
  std::vector< std::vector<BivariateGaussianKernelBasis> > bases_per_thread (n_threads);
  std::vector<BivariateSolver> solvers (n_threads);
  std::vector<double> log_likelihoods = std::vector<double> (N);

  for (unsigned i=0; i<n_threads; ++i) {
    bases_per_thread[i] = std::vector<BivariateGaussianKernelBasis> (n_rhos);
    std::vector<BivariateGaussianKernelBasis>& current_basis = bases_per_thread[i];

    for (unsigned j=0; j<n_rhos; ++j) {
      current_basis[j] = *bases[j];
    }
  }

  std::cout << "copying bases vectors for data" << std::endl;
  std::vector< std::vector<BivariateGaussianKernelBasis> > bases_per_data (N);
  for (unsigned i=0; i<N; ++i) {
    bases_per_data[i] = std::vector<BivariateGaussianKernelBasis> (n_rhos);
    std::vector<BivariateGaussianKernelBasis>& current_basis = bases_per_data[i];

    for (unsigned j=0; j<n_rhos; ++j) {
      current_basis[j] = *bases[j];
    }
  }  


  unsigned i=0;
  unsigned thread_index = 0;
  BivariateSolver FEM_solver = BivariateSolver();
  auto t1 = std::chrono::high_resolution_clock::now();


#pragma omp parallel for num_threads(n_threads) shared(bases_per_data, bases, solvers, bases_per_thread, BMs, t, dx, sigma_y_data_gen, sigma_x_data_gen, log_likelihoods) private(i, thread_index, FEM_solver)
    for (i=0; i<N; ++i) {
      BivariateGaussianKernelBasis local_basis = BivariateGaussianKernelBasis(dx, 
									      0.0,
									      0.3,
									      1, 
									      0.5);
      BivariateSolver solver = BivariateSolver(&local_basis); 

      // BivariateGaussianKernelBasis basis = *bases[0];
      //      std::vector<BivariateGaussianKernelBasis>& local_bases = bases_per_thread[thread_index];
      // solvers[thread_index] = BivariateSolver(&basis,
      // 					      sigma_x_data_gen,
      // 					      sigma_y_data_gen,
      // 					      0.1,
      // 					      BMs[thread_index].get_a(),
      // 					      BMs[thread_index].get_x_0(),
      // 					      BMs[thread_index].get_b(),
      // 					      BMs[thread_index].get_c(),
      // 					      BMs[thread_index].get_y_0(),
      // 					      BMs[thread_index].get_d(),
      // 					      t, 
      // 					      dx);
      // solvers[thread_index] = BivariateSolver();
    }

  auto t2 = std::chrono::high_resolution_clock::now();    
  std::cout << "OMP duration = "
   	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
   	    << " milliseconds\n";  
  
  unsigned j = 0;
  std::vector<BivariateGaussianKernelBasis> local_bases (0);
  gsl_vector* local_input;
  log_likelihoods = std::vector<double> (N);

  omp_set_dynamic(0);
// #pragma omp parallel num_threads(6) shared(bases, BMs, t, dx, sigma_y_data_gen, sigma_x_data_gen, log_likelihoods) private(local_bases, j, FEM_solver, local_input)
//   {
//     local_bases = std::vector<BivariateGaussianKernelBasis> (bases.size());
//     for (unsigned i=0; i<bases.size(); ++i) {
//        local_bases[i] = BivariateGaussianKernelBasis(*bases[i]);
//     }
//     FEM_solver = BivariateSolver(&local_bases[0],
// 					 sigma_x_data_gen,
// 					 sigma_y_data_gen,
// 					 0.1,
// 					 BMs[omp_get_thread_num()].get_a(),
// 					 BMs[omp_get_thread_num()].get_x_0(),
// 					 BMs[omp_get_thread_num()].get_b(),
// 					 BMs[omp_get_thread_num()].get_c(),
// 					 BMs[omp_get_thread_num()].get_y_0(),
// 					 BMs[omp_get_thread_num()].get_d(),
// 					 t, 
// 					 dx);
//     local_input = gsl_vector_alloc(2);
    
//     std::cout << "I am thread " << omp_get_thread_num() << " and I have "
// 	      << local_bases.size() << " basis objects. The address of the container is"
// 	      << &local_bases << ". ";
//     std::cout << "The addresses of the basis objets are " 
// 	     << &local_bases[0] << ", " << &local_bases[1] << ", "
// 	      << &local_bases[2] << ". ";
//     std::cout << "The address of my own solver is " 
// 	      << &FEM_solver << "." << std::endl;

//     auto t1 = std::chrono::high_resolution_clock::now();
//     for (j = omp_get_thread_num()*5; 
// 	 j<(omp_get_thread_num()+1)*5; 
// 	 ++j) {
//       std::cout << "thread = " << omp_get_thread_num();
//       std::cout << " handling data point " << j 
// 		<< " with basis vector " << &local_bases
// 		<< " and solver " << &FEM_solver 
// 		<< " and input " << local_input << ". ";
//       gsl_vector_set(local_input, 0, BMs[j].get_x_T());
//       gsl_vector_set(local_input, 1, BMs[j].get_y_T());
//       std::cout << "BMs[j].get_a() = " << BMs[j].get_a() << std::endl;
//       // FEM_solver_ptr->set_data(BMs[j].get_a(),
//       // 			       BMs[j].get_x_0(),
//       // 			       BMs[j].get_b(),
//       // 			       BMs[j].get_c(),
//       // 			       BMs[j].get_y_0(),
//       // 			       BMs[j].get_d());
//       FEM_solver = BivariateSolver(&local_bases[0],
//       					sigma_x_data_gen,
//       					sigma_y_data_gen,
//       					0.1,
//       					BMs[j].get_a(),
//       					BMs[j].get_x_0(),
//       					BMs[j].get_b(),
//       					BMs[j].get_c(),
//       					BMs[j].get_y_0(),
//       					BMs[j].get_d(),
//       					t,
//       					dx);
//       log_likelihoods[j] = FEM_solver.numerical_likelihood_first_order(local_input, 1.0/1024);
//     }
//     auto t2 = std::chrono::high_resolution_clock::now();    
//     std::cout << "duration = "
//    	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
//    	    << " milliseconds\n";  

//     // for (unsigned i=0; i<bases.size(); ++i) {
//     //   delete local_bases[i];
//     // }
//     gsl_vector_free(local_input);
//     // delete FEM_solver_ptr;
//   }
  
  // unsigned R = 10;
  // double dr = 0.1;
  // double rho_init = -0.2;
  // for (unsigned r=0; r<R; ++r) {
  //   double rho = rho_init + dr*r;
  //   BivariateGaussianKernelBasis * basis;
    
  //   unsigned k = 0;
  //   std::vector<double> abs_differences (n_rhos);
  //   for (k=0; k<n_rhos; ++k) {
  //     std::cout << "rho = " << rho << "; ";
  //     std::cout << "rho - basis_rhos[k] = " << rho - basis_rhos[k] << "; ";
  //     std::cout << "std::abs(rho-basis_rhos[k]) = " << std::abs(rho-basis_rhos[k])
  // 		<< std::endl;
  //   }

  //   k = 0;
  //   std::generate(abs_differences.begin(), 
  //   		  abs_differences.end(), [&k, &rho, &basis_rhos]
  //   		  { 
  //   		    double out = std::abs(rho - basis_rhos[k]);
  //   		    k++; 
  //   		    return out;
  //   		  });

  //   for (k=0; k<n_rhos; ++k) {
  //     std::cout << "abs_differences[" << k << "] = "
  //   		<< abs_differences[k] << std::endl;
  //   }

  //   k=0;
  //   std::vector<double> abs_differences_indeces (n_rhos);
  //   std::generate(abs_differences_indeces.begin(),
  // 		  abs_differences_indeces.end(),
  // 		  [&k]{ return k++; });
  //   std::sort(abs_differences_indeces.begin(), 
  //   	      abs_differences_indeces.end(),
  //   	    [&abs_differences] (unsigned i1, unsigned i2) -> bool
  //   	    {
  //   	      return abs_differences[i1] < abs_differences[i2];
  //   	    });
  //   std::cout << "abs_differences_indeces[0] = " << abs_differences_indeces[0] << std::endl;
  //   basis = bases[abs_differences_indeces[0]];

  //   double log_likelihood = 0;
  //   auto t1 = std::chrono::high_resolution_clock::now();    
  //   BivariateSolver FEM_solver_2 = BivariateSolver();
  //   for (unsigned i=0; i<N; ++i) {
  //     gsl_vector_set(input, 0, BMs[i].get_x_T());
  //     gsl_vector_set(input, 1, BMs[i].get_y_T());
      
  //     FEM_solver_2 = BivariateSolver(basis,
  // 				     sigma_x_data_gen, 
  // 				     sigma_y_data_gen,
  // 				     rho,
  // 				     BMs[i].get_a(),
  // 				     BMs[i].get_x_0(),
  // 				     BMs[i].get_b(),
  // 				     BMs[i].get_c(),
  // 				     BMs[i].get_y_0(),
  // 				     BMs[i].get_d(),
  // 				     t,
  // 				     dx);
  //     double FEM_likelihood = 0.0;
  //     // FEM_likelihood = FEM_solver_2.numerical_likelihood_first_order(input, 
  //     // 								     1.0/64);
      
  //     std::cout << "i=" << i << "; ";
  //     std::cout << "FEM.numerical_likelihood(input,dx) = "
  // 		<< FEM_likelihood << "; "
  // 		<< "r = " << r;
  //     std::cout << std::endl;

  //     if (FEM_likelihood > 1e-16) {
  // 	log_likelihood = log_likelihood + log(FEM_likelihood);
  //     }
  //   }
  //   auto t2 = std::chrono::high_resolution_clock::now();    
  //   std::cout << "duration = "
  //  	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  //  	    << " milliseconds\n";  

  //   output_file << log_likelihood << ", " << rho << "\n";
  // }

  // gsl_vector_free(input);
  // output_file.close();
  return 0;
}


