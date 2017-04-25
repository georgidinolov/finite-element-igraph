#include "2DBrownianMotionPath.hpp"
#include "BivariateSolver.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <string>
#include <stdio.h>
#include <vector>


static std::vector<BivariateGaussianKernelBasis>* private_bases;
#pragma omp threadprivate(private_bases)

void increment_counter(int & counter)
{
  counter ++;
}

double likelihood_serial(std::vector<BivariateGaussianKernelBasis>& bases,
			 const std::vector<BrownianMotion>& BMs,
			 const std::vector<double>& basis_rhos,
			 double sigma_x_data_gen,
			 double sigma_y_data_gen,
			 double rho,
			 double t,
			 double dx)
{
  unsigned N = BMs.size();
  double sum_of_elements_serial = 0;
  gsl_vector* input = gsl_vector_alloc(2);

  std::vector<double> abs_differences (basis_rhos.size());
  unsigned k = 0;
  std::generate(abs_differences.begin(), 
		abs_differences.end(), [&k, &rho, &basis_rhos]
		{ 
		  double out = std::abs(rho - basis_rhos[k]);
		  k++; 
		  return out;
		});
  
  k=0;
  std::vector<double> abs_differences_indeces (basis_rhos.size());
  std::generate(abs_differences_indeces.begin(),
		abs_differences_indeces.end(),
		[&k]{ return k++; });
  std::sort(abs_differences_indeces.begin(), 
	    abs_differences_indeces.end(),
    	    [&abs_differences] (unsigned i1, unsigned i2) -> bool
    	    {
    	      return abs_differences[i1] < abs_differences[i2];
    	    });

  k = abs_differences_indeces[0];

  for (unsigned i=0; i<N; ++i) {
    BivariateSolver solver = BivariateSolver(&bases[k],
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
    gsl_vector_set(input, 0, BMs[i].get_x_T());
    gsl_vector_set(input, 1, BMs[i].get_y_T());

    double lik = solver.numerical_likelihood_first_order(input, 1.0/256);
    if (lik > 0) {
      sum_of_elements_serial = sum_of_elements_serial + log(lik);
    }
  }

  gsl_vector_free(input);
  return sum_of_elements_serial;
}

std::vector<double> likelihood(const std::vector<BrownianMotion>& BMs,
		  const std::vector<double>& basis_rhos,
		  double sigma_x_data_gen,
		  double sigma_y_data_gen,
		  double rho,
		  double t,
		  double dx,
		  double dx_likelihood) 
{
  unsigned i = 0;
  int tid = 0;
  gsl_vector* input;
  unsigned N = BMs.size();
  std::vector<double> likelihoods (N);
  std::vector<double> abs_differences (basis_rhos.size());
  unsigned k = 0;
  std::generate(abs_differences.begin(), 
		abs_differences.end(), [&k, &rho, &basis_rhos]
		{ 
		  double out = std::abs(rho - basis_rhos[k]);
		  k++; 
		  return out;
		});
  
  k=0;
  std::vector<double> abs_differences_indeces (basis_rhos.size());
  std::generate(abs_differences_indeces.begin(),
		abs_differences_indeces.end(),
		[&k]{ return k++; });
  std::sort(abs_differences_indeces.begin(), 
	    abs_differences_indeces.end(),
    	    [&abs_differences] (unsigned i1, unsigned i2) -> bool
    	    {
    	      return abs_differences[i1] < abs_differences[i2];
    	    });

  k = abs_differences_indeces[0];
  
#pragma omp parallel default(none) private(tid, i, input) shared(N, BMs, likelihoods) firstprivate(k, rho, sigma_x_data_gen, sigma_y_data_gen, t, dx, dx_likelihood)
  {
    tid = omp_get_thread_num();
    input = gsl_vector_alloc(2);
    
#pragma omp for
    for (i=0; i<N; ++i) {
      tid = omp_get_thread_num();
      BivariateSolver solver = BivariateSolver(&(*private_bases)[k],
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
      gsl_vector_set(input, 0, BMs[i].get_x_T());
      gsl_vector_set(input, 1, BMs[i].get_y_T());
      double likelihood = solver.numerical_likelihood_first_order(input, 
								  dx_likelihood);
      if (likelihood > 0) {
	likelihoods[i] = log(likelihood);
      } else {
	likelihoods[i] = 0;
	printf("For rho=%f, data %d produces neg likelihood.\n", rho, i);
      }
    }

    gsl_vector_free(input);
  }

  double sum_of_elements_parallel = 0;
  std::for_each(likelihoods.begin(), likelihoods.end(), [&sum_of_elements_parallel] (double lik) {
      sum_of_elements_parallel += lik;
    });

  return likelihoods;
}


int main() {
  omp_set_dynamic(0);

  static int counter = 0;
#pragma omp threadprivate(counter)

  int tid;
 
  std::cout << std::fixed << std::setprecision(32);
  unsigned order = 1e6;
  double sigma_x_data_gen = 1.0;
  double sigma_y_data_gen = 1.0;
  double rho_data_gen = 0.6;
  double x_0 = 0.0;
  double y_0 = 0.0;
  double t = 1;
  long unsigned seed_init = 4002;
  double dx = 5e-3;
  double rho_min = -0.8;
  double rho_max = 0.8;
  unsigned n_rhos = 21;
  gsl_vector* input = gsl_vector_alloc(2);
  unsigned N = 64;

  // LIKELIHOOD LOOP
  std::ofstream output_file;
  output_file.open("likelihood-rho.txt");
  output_file << std::fixed << std::setprecision(32);
  // HEADER
  // output_file << "log.likelihood.512, log.likelihood.256, log.likelihood.128, log.likelihood.64, rho\n";
  for (unsigned i=0; i<N; ++i) {
    output_file << "log.likelihood." << i << ", ";
  }
  output_file << "rho\n";


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
    std::cout << "(" << BM.get_a() << "," << BM.get_x_T() << "," << BM.get_b() << ") "
	      << "(" << BM.get_c() << "," << BM.get_y_T() << "," << BM.get_d() << ")"
	      << std::endl;
  }

  // MASTER COPY OF BASES START 
  std::cout << "throwing bases on heap\n";
  std::vector<BivariateGaussianKernelBasis> bases (n_rhos);
  

  std::vector<double> basis_rhos (n_rhos);
  double drho = (rho_max-rho_min)/(n_rhos-1);
  
#pragma omp parallel for default(shared)
  for (unsigned k=0; k<n_rhos; ++k) {
    double current_rho = rho_min + drho*k;
    if (current_rho <= rho_max && current_rho >= rho_min) {
      basis_rhos[k] = current_rho;
    } else if (current_rho > rho_max) {
      basis_rhos[k] = rho_max;
    } else if (current_rho < rho_min) {
      basis_rhos[k] = rho_min;
    } else {
      printf("ERROR: rho cannot be set\n");
      basis_rhos[k] = rho_min + (rho_max-rho_min)/2.0;
    }
    bases[k] =  BivariateGaussianKernelBasis(dx, 
					     basis_rhos[k],
					     0.3,
					     1, 
					     0.5);
    printf("done with basis %d, with rho = %f\n", k, basis_rhos[k]);
  }
  // MASTER COPY OF BASES END

  // BASES COPY FOR THREADS START
  std::vector<double> likelihoods (N);
  unsigned n_threads = 64;
  tid = 0;
  unsigned i = 0;

  //  BivariateSolver solver = BivariateSolver();
  std::cout << "copying bases vectors for threads as private variables" << std::endl;
  omp_set_num_threads(n_threads);

  auto t1 = std::chrono::high_resolution_clock::now();    
#pragma omp parallel default(none) private(tid, i, input) shared(n_rhos, N, BMs, bases, likelihoods) firstprivate(sigma_x_data_gen, sigma_y_data_gen, t, dx)
  {
    tid = omp_get_thread_num();
    input = gsl_vector_alloc(2);
    
    private_bases = new std::vector<BivariateGaussianKernelBasis> (n_rhos);
    for (i=0; i<n_rhos; ++i) {
      (*private_bases)[i] = bases[i];
    }

    printf("Thread %d: counter %d\n", tid, counter);
  }
  auto t2 = std::chrono::high_resolution_clock::now();    
  std::cout << "OMP duration = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " milliseconds\n";
  std::cout << "DONE copying bases vectors for threads as private variables" << std::endl;
  std::cout << std::endl;

  double rho_init = -0.8;
  double dr = 0.08;
  unsigned R = 21;
  for (unsigned r=0; r<R; ++r) {
    double rho = rho_init + dr*r;

    t1 = std::chrono::high_resolution_clock::now();    
    std::vector<double> log_likelihoods = likelihood(BMs,basis_rhos,
						     sigma_x_data_gen,
						     sigma_y_data_gen,
						     rho,
						     t,dx, 1.0/64);
    for (unsigned i=0; i<log_likelihoods.size(); ++i) {
      output_file << log_likelihoods[i] << ", ";
    }
    output_file << rho << "\n";

    // log_likelihood = likelihood(BMs,basis_rhos,sigma_x_data_gen,sigma_y_data_gen, rho,t,dx, 1.0/256);
    // output_file << log_likelihood << ", ";

    //  log_likelihood = likelihood(BMs,basis_rhos,sigma_x_data_gen,sigma_y_data_gen, rho,t,dx, 1.0/128);
    // output_file << log_likelihood << ", ";

    //  log_likelihood = likelihood(BMs,basis_rhos,sigma_x_data_gen,sigma_y_data_gen, rho,t,dx, 1.0/64);
    // output_file << log_likelihood << ", ";
    t2 = std::chrono::high_resolution_clock::now();    
    std::cout << "duration = "
   	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
   	    << " milliseconds\n";  
  }


#pragma omp parallel default(none)
  {
    delete private_bases;
  }

//   std::vector<BivariateSolver> solvers (n_threads);
//   std::vector<double> log_likelihoods = std::vector<double> (N);

//   for (unsigned i=0; i<n_threads; ++i) {
//     bases_per_thread[i] = std::vector<BivariateGaussianKernelBasis> (n_rhos);
//     std::vector<BivariateGaussianKernelBasis>& current_basis = bases_per_thread[i];

//     for (unsigned j=0; j<n_rhos; ++j) {
//       current_basis[j] = *bases[j];
//     }
//   }

//   std::cout << "copying bases vectors for data" << std::endl;
//   std::vector< std::vector<BivariateGaussianKernelBasis> > bases_per_data (N);
//   for (unsigned i=0; i<N; ++i) {
//     bases_per_data[i] = std::vector<BivariateGaussianKernelBasis> (n_rhos);
//     std::vector<BivariateGaussianKernelBasis>& current_basis = bases_per_data[i];

//     for (unsigned j=0; j<n_rhos; ++j) {
//       current_basis[j] = *bases[j];
//     }
//   }  


//   unsigned i=0;
//   unsigned thread_index = 0;
//   BivariateSolver FEM_solver = BivariateSolver();
//   auto t1 = std::chrono::high_resolution_clock::now();


// #pragma omp parallel for num_threads(n_threads) shared(bases_per_data, bases, solvers, bases_per_thread, BMs, t, dx, sigma_y_data_gen, sigma_x_data_gen, log_likelihoods) private(i, thread_index, FEM_solver)
//     for (i=0; i<N; ++i) {
//       BivariateGaussianKernelBasis local_basis = BivariateGaussianKernelBasis(dx, 
// 									      0.0,
// 									      0.3,
// 									      1, 
// 									      0.5);
//       BivariateSolver solver = BivariateSolver(&local_basis); 

//       // BivariateGaussianKernelBasis basis = *bases[0];
//       //      std::vector<BivariateGaussianKernelBasis>& local_bases = bases_per_thread[thread_index];
//       // solvers[thread_index] = BivariateSolver(&basis,
//       // 					      sigma_x_data_gen,
//       // 					      sigma_y_data_gen,
//       // 					      0.1,
//       // 					      BMs[thread_index].get_a(),
//       // 					      BMs[thread_index].get_x_0(),
//       // 					      BMs[thread_index].get_b(),
//       // 					      BMs[thread_index].get_c(),
//       // 					      BMs[thread_index].get_y_0(),
//       // 					      BMs[thread_index].get_d(),
//       // 					      t, 
//       // 					      dx);
//       // solvers[thread_index] = BivariateSolver();
//     }

//   auto t2 = std::chrono::high_resolution_clock::now();    
//   std::cout << "OMP duration = "
//    	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
//    	    << " milliseconds\n";  
  
//   unsigned j = 0;
//   std::vector<BivariateGaussianKernelBasis> local_bases (0);
//   gsl_vector* local_input;
//   log_likelihoods = std::vector<double> (N);

//   omp_set_dynamic(0);
// // #pragma omp parallel num_threads(6) shared(bases, BMs, t, dx, sigma_y_data_gen, sigma_x_data_gen, log_likelihoods) private(local_bases, j, FEM_solver, local_input)
// //   {
// //     local_bases = std::vector<BivariateGaussianKernelBasis> (bases.size());
// //     for (unsigned i=0; i<bases.size(); ++i) {
// //        local_bases[i] = BivariateGaussianKernelBasis(*bases[i]);
// //     }
// //     FEM_solver = BivariateSolver(&local_bases[0],
// // 					 sigma_x_data_gen,
// // 					 sigma_y_data_gen,
// // 					 0.1,
// // 					 BMs[omp_get_thread_num()].get_a(),
// // 					 BMs[omp_get_thread_num()].get_x_0(),
// // 					 BMs[omp_get_thread_num()].get_b(),
// // 					 BMs[omp_get_thread_num()].get_c(),
// // 					 BMs[omp_get_thread_num()].get_y_0(),
// // 					 BMs[omp_get_thread_num()].get_d(),
// // 					 t, 
// // 					 dx);
// //     local_input = gsl_vector_alloc(2);
    
// //     std::cout << "I am thread " << omp_get_thread_num() << " and I have "
// // 	      << local_bases.size() << " basis objects. The address of the container is"
// // 	      << &local_bases << ". ";
// //     std::cout << "The addresses of the basis objets are " 
// // 	     << &local_bases[0] << ", " << &local_bases[1] << ", "
// // 	      << &local_bases[2] << ". ";
// //     std::cout << "The address of my own solver is " 
// // 	      << &FEM_solver << "." << std::endl;

// //     auto t1 = std::chrono::high_resolution_clock::now();
// //     for (j = omp_get_thread_num()*5; 
// // 	 j<(omp_get_thread_num()+1)*5; 
// // 	 ++j) {
// //       std::cout << "thread = " << omp_get_thread_num();
// //       std::cout << " handling data point " << j 
// // 		<< " with basis vector " << &local_bases
// // 		<< " and solver " << &FEM_solver 
// // 		<< " and input " << local_input << ". ";
// //       gsl_vector_set(local_input, 0, BMs[j].get_x_T());
// //       gsl_vector_set(local_input, 1, BMs[j].get_y_T());
// //       std::cout << "BMs[j].get_a() = " << BMs[j].get_a() << std::endl;
// //       // FEM_solver_ptr->set_data(BMs[j].get_a(),
// //       // 			       BMs[j].get_x_0(),
// //       // 			       BMs[j].get_b(),
// //       // 			       BMs[j].get_c(),
// //       // 			       BMs[j].get_y_0(),
// //       // 			       BMs[j].get_d());
// //       FEM_solver = BivariateSolver(&local_bases[0],
// //       					sigma_x_data_gen,
// //       					sigma_y_data_gen,
// //       					0.1,
// //       					BMs[j].get_a(),
// //       					BMs[j].get_x_0(),
// //       					BMs[j].get_b(),
// //       					BMs[j].get_c(),
// //       					BMs[j].get_y_0(),
// //       					BMs[j].get_d(),
// //       					t,
// //       					dx);
// //       log_likelihoods[j] = FEM_solver.numerical_likelihood_first_order(local_input, 1.0/1024);
// //     }
// //     auto t2 = std::chrono::high_resolution_clock::now();    
// //     std::cout << "duration = "
// //    	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
// //    	    << " milliseconds\n";  

// //     // for (unsigned i=0; i<bases.size(); ++i) {
// //     //   delete local_bases[i];
// //     // }
// //     gsl_vector_free(local_input);
// //     // delete FEM_solver_ptr;
// //   }
  
//   // unsigned R = 10;
//   // double dr = 0.1;
//   // double rho_init = -0.2;
//   // for (unsigned r=0; r<R; ++r) {
//   //   double rho = rho_init + dr*r;
//   //   BivariateGaussianKernelBasis * basis;
    
//   //   unsigned k = 0;
//   //   std::vector<double> abs_differences (n_rhos);
//   //   for (k=0; k<n_rhos; ++k) {
//   //     std::cout << "rho = " << rho << "; ";
//   //     std::cout << "rho - basis_rhos[k] = " << rho - basis_rhos[k] << "; ";
//   //     std::cout << "std::abs(rho-basis_rhos[k]) = " << std::abs(rho-basis_rhos[k])
//   // 		<< std::endl;
//   //   }

//   //   k = 0;
//   //   std::generate(abs_differences.begin(), 
//   //   		  abs_differences.end(), [&k, &rho, &basis_rhos]
//   //   		  { 
//   //   		    double out = std::abs(rho - basis_rhos[k]);
//   //   		    k++; 
//   //   		    return out;
//   //   		  });

//   //   for (k=0; k<n_rhos; ++k) {
//   //     std::cout << "abs_differences[" << k << "] = "
//   //   		<< abs_differences[k] << std::endl;
//   //   }

//   //   k=0;
//   //   std::vector<double> abs_differences_indeces (n_rhos);
//   //   std::generate(abs_differences_indeces.begin(),
//   // 		  abs_differences_indeces.end(),
//   // 		  [&k]{ return k++; });
//   //   std::sort(abs_differences_indeces.begin(), 
//   //   	      abs_differences_indeces.end(),
//   //   	    [&abs_differences] (unsigned i1, unsigned i2) -> bool
//   //   	    {
//   //   	      return abs_differences[i1] < abs_differences[i2];
//   //   	    });
//   //   std::cout << "abs_differences_indeces[0] = " << abs_differences_indeces[0] << std::endl;
//   //   basis = bases[abs_differences_indeces[0]];

//   //   double log_likelihood = 0;
//   //   auto t1 = std::chrono::high_resolution_clock::now();    
//   //   BivariateSolver FEM_solver_2 = BivariateSolver();
//   //   for (unsigned i=0; i<N; ++i) {
//   //     gsl_vector_set(input, 0, BMs[i].get_x_T());
//   //     gsl_vector_set(input, 1, BMs[i].get_y_T());
      
//   //     FEM_solver_2 = BivariateSolver(basis,
//   // 				     sigma_x_data_gen, 
//   // 				     sigma_y_data_gen,
//   // 				     rho,
//   // 				     BMs[i].get_a(),
//   // 				     BMs[i].get_x_0(),
//   // 				     BMs[i].get_b(),
//   // 				     BMs[i].get_c(),
//   // 				     BMs[i].get_y_0(),
//   // 				     BMs[i].get_d(),
//   // 				     t,
//   // 				     dx);
//   //     double FEM_likelihood = 0.0;
//   //     // FEM_likelihood = FEM_solver_2.numerical_likelihood_first_order(input, 
//   //     // 								     1.0/64);
      
//   //     std::cout << "i=" << i << "; ";
//   //     std::cout << "FEM.numerical_likelihood(input,dx) = "
//   // 		<< FEM_likelihood << "; "
//   // 		<< "r = " << r;
//   //     std::cout << std::endl;

//   //     if (FEM_likelihood > 1e-16) {
//   // 	log_likelihood = log_likelihood + log(FEM_likelihood);
//   //     }
//   //   }
//   //   auto t2 = std::chrono::high_resolution_clock::now();    
//   //   std::cout << "duration = "
//   //  	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
//   //  	    << " milliseconds\n";  

//   //   output_file << log_likelihood << ", " << rho << "\n";
//   // }

//   // gsl_vector_free(input);
  output_file.close();
  return 0;
}


