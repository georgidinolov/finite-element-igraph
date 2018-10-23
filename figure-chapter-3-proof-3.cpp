#include <algorithm>
#include "BivariateSolver.hpp"
#include "GaussianInterpolator.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <iostream>
#include <sstream>
#include <limits>
#include "nlopt.hpp"
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc < 6 || argc > 6) {
    printf("You must provide input\n");
    printf("The input is: \n\nnumber data points;\nrho_basis;\nsigma_x;\nsigma_y;\ndx_likelihood;\n");
    exit(0);
  }

  unsigned N = std::stoi(argv[1]);
  double rho_basis = std::stod(argv[2]);
  double sigma_x_basis = std::stod(argv[3]);
  double sigma_y_basis = std::stod(argv[4]);
  double dx_likelihood = std::stod(argv[5]);
  double dx = 1.0/600.0;

  static int counter = 0;
  static BivariateGaussianKernelBasis* private_bases;
  static gsl_rng* r_ptr_threadprivate;

#pragma omp threadprivate(private_bases, counter, r_ptr_threadprivate)
  omp_set_dynamic(0);
  omp_set_num_threads(1);

  BivariateGaussianKernelBasis basis_positive =
    BivariateGaussianKernelBasis(dx,
				 rho_basis,
				 sigma_x_basis,
				 sigma_y_basis,
				 1.0,
				 1.0);

  long unsigned seed_init = 10;
  gsl_rng * r_ptr_local;
  const gsl_rng_type * Type;
  gsl_rng_env_setup();
  Type = gsl_rng_default;
  r_ptr_local = gsl_rng_alloc(Type);
  gsl_rng_set(r_ptr_local, seed_init + N);

  int tid=0;
  unsigned i=0;
#pragma omp parallel default(none) private(tid, i) shared(basis_positive, r_ptr_local)
  {
    tid = omp_get_thread_num();

    private_bases = new BivariateGaussianKernelBasis();
    (*private_bases) = basis_positive;

    r_ptr_threadprivate = gsl_rng_clone(r_ptr_local);
    gsl_rng_set(r_ptr_threadprivate, tid);

    printf("Thread %d: counter %d\n", tid, counter);
  }

  std::vector<likelihood_point> points_for_kriging (1);
  points_for_kriging[0].x_0_tilde=1.00e-01;
  points_for_kriging[0].y_0_tilde=3.00e-01;
  points_for_kriging[0].x_t_tilde=3.00e-01;
  points_for_kriging[0].y_t_tilde=5.00e-01;
  points_for_kriging[0].sigma_y_tilde=0.1;
  points_for_kriging[0].t_tilde=0.058;
  points_for_kriging[0].rho=0.9;
  points_for_kriging[0].log_likelihood=-4.32267567159303e+00;
  points_for_kriging[0].FLIPPED=0;

  // points_for_kriging[1].x_0_tilde=6.50160070618248e-02;
  // points_for_kriging[1].y_0_tilde=7.61119804763752e-01;
  // points_for_kriging[1].x_t_tilde=1.85135596642142e-01;
  // points_for_kriging[1].y_t_tilde=3.19775518059827e-02;
  // points_for_kriging[1].sigma_y_tilde=2.24954509905047e-01;
  // points_for_kriging[1].t_tilde=9.06089050024474e+00;
  // points_for_kriging[1].rho=8.65650669386276e-01;
  // points_for_kriging[1].log_likelihood=-4.32267567159303e+00;
  // points_for_kriging[1].FLIPPED=0;

  std::vector<likelihood_point> points_for_integration (1);

  auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(none) private(i) shared(points_for_kriging, N, seed_init, r_ptr_local) firstprivate(dx_likelihood, dx)
    {
#pragma omp for
      for (i=0; i<N; ++i) {
	double raw_input_array [2] = {points_for_kriging[i].x_t_tilde,points_for_kriging[i].y_t_tilde};
	gsl_vector_view raw_input = gsl_vector_view_array(raw_input_array,2);
	
	long unsigned seed = seed_init + i;
	double likelihood = 0.0;
	double rho = points_for_kriging[i].rho;
	double x [2] = {points_for_kriging[i].x_t_tilde,
			points_for_kriging[i].y_t_tilde};
	
	gsl_vector_view gsl_x = gsl_vector_view_array(x, 2);
	  
	if (!std::signbit(rho)) {
	  BivariateSolver solver = BivariateSolver(private_bases,
						   1.0,
						   points_for_kriging[i].sigma_y_tilde,
						   rho,
						   0.0,
						   points_for_kriging[i].x_0_tilde,
						   1.0,
						   0.0,
						   points_for_kriging[i].y_0_tilde,
						   1.0,
						   points_for_kriging[i].t_tilde,
						   dx);
	    
	  x[0] = points_for_kriging[i].x_t_tilde;
	  x[1] = points_for_kriging[i].y_t_tilde;
	  
	  std::vector<BivariateImageWithTime> all_images =
	    solver.small_t_image_positions_type_41_all(false);
	  printf("pdf(\"./src/kernel-expansion/documentation/chapter-3/chapter-3-figure-proof-3.pdf\", 6,6)\n");
	  printf("plot(x=c(-35,40),y=c(-35,40),xlab=\"\",ylab=\"\",type=\"n\");\n");
	  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
		 0.0, 1.0, 0.0, 0.0);
	  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
		 1.0, 1.0, 0.0, 1.0);
	  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
		 1.0, 0.0, 1.0, 1.0);
	  printf("lines(x=c(%g,%g), y=c(%g,%g));\n ",
		 0.0, 0.0, 1.0, 0.0);
	  
	  for (unsigned ii=0; ii<all_images.size(); ++ii) {
	    std::vector<unsigned> reflection_sequence_per_final_image_current =
	      all_images[ii].get_reflection_sequence();
	    
	    if (((reflection_sequence_per_final_image_current.size() == 4) &&
		 (reflection_sequence_per_final_image_current[0] == 1) &&
		 (reflection_sequence_per_final_image_current[1] == 3) &&
		 (reflection_sequence_per_final_image_current[2] == 0) &&
		 (reflection_sequence_per_final_image_current[3] == 2)) |
		((reflection_sequence_per_final_image_current.size() == 4) &&
		 (reflection_sequence_per_final_image_current[0] == 1) &&
		 (reflection_sequence_per_final_image_current[1] == 3) &&
		 (reflection_sequence_per_final_image_current[2] == 2) &&
		 (reflection_sequence_per_final_image_current[3] == 0)) |
		((reflection_sequence_per_final_image_current.size() == 4) &&
		 (reflection_sequence_per_final_image_current[0] == 3) &&
		 (reflection_sequence_per_final_image_current[1] == 1) &&
		 (reflection_sequence_per_final_image_current[2] == 0) &&
		 (reflection_sequence_per_final_image_current[3] == 2)) |
		((reflection_sequence_per_final_image_current.size() == 4) &&
		 (reflection_sequence_per_final_image_current[0] == 3) &&
		 (reflection_sequence_per_final_image_current[1] == 1) &&
		 (reflection_sequence_per_final_image_current[2] == 2) &&
		 (reflection_sequence_per_final_image_current[3] == 0)))
	      {
		printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"green\"); ## ",
		       ii,
		       gsl_vector_get(all_images[ii].get_position(),0),
		       gsl_vector_get(all_images[ii].get_position(),1),
		       ii,
		       all_images[ii].get_mult_factor(),
		       gsl_vector_get(all_images[ii].get_position(),0),
		       gsl_vector_get(all_images[ii].get_position(),1));
	      }  else if (reflection_sequence_per_final_image_current.size() > 4) {
	      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"blue\"); ## ",
		     ii,
		     gsl_vector_get(all_images[ii].get_position(),0),
		     gsl_vector_get(all_images[ii].get_position(),1),
		     ii,
		     all_images[ii].get_mult_factor(),
		     gsl_vector_get(all_images[ii].get_position(),0),
		     gsl_vector_get(all_images[ii].get_position(),1));
	    } else if (reflection_sequence_per_final_image_current.size() == 0) {
	      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g,pch=20,col=\"red\"); ## ",
		     ii,
		     gsl_vector_get(all_images[ii].get_position(),0),
		     gsl_vector_get(all_images[ii].get_position(),1),
		     ii,
		     all_images[ii].get_mult_factor(),
		     gsl_vector_get(all_images[ii].get_position(),0),
		     gsl_vector_get(all_images[ii].get_position(),1));
	    } else {
	      printf("image.%i = c(%g,%g); sign.%i=%g; points(%g,%g); ## ",
		     ii,
		     gsl_vector_get(all_images[ii].get_position(),0),
		     gsl_vector_get(all_images[ii].get_position(),1),
		     ii,
		     all_images[ii].get_mult_factor(),
		     gsl_vector_get(all_images[ii].get_position(),0),
		     gsl_vector_get(all_images[ii].get_position(),1));
	    }
    
	    for (const unsigned& refl : reflection_sequence_per_final_image_current) {
	      printf("%i ", refl);
	    }
    
	    printf("\n");
	  }
	  printf("dev.off();\n ");
	}
	
	points_for_kriging[i].log_likelihood = log(likelihood);
	printf("Thread %d with address %p produces likelihood %g\n",
	       omp_get_thread_num(),
	       private_bases,
	       likelihood);
      }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    gsl_rng_free(r_ptr_local);
    return 0;
}
