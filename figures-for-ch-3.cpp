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
  if (argc < 8 || argc > 8) {
    printf("You must provide input\n");
    printf("The input is: \n\nnumber data points;\nrho_basis;\nsigma_x;\nsigma_y;\ndx_likelihood;\nfile prefix;\ndata_file for points;\n");
    exit(0);
  }

  unsigned N = std::stoi(argv[1]);
  double rho_basis = std::stod(argv[2]);
  double sigma_x_basis = std::stod(argv[3]);
  double sigma_y_basis = std::stod(argv[4]);
  double dx_likelihood = std::stod(argv[5]);
  std::string file_prefix = argv[6];
  std::string input_file_name = argv[7];
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
  points_for_kriging[1].x_0_tilde=5.00160070618248e-01;
  points_for_kriging[0].y_0_tilde=5.01119804763752e-01;
  points_for_kriging[1].x_t_tilde=2.85135596642142e-01;
  points_for_kriging[0].y_t_tilde=3.19775518059827e-01;
  points_for_kriging[0].sigma_y_tilde=0.5;
  points_for_kriging[0].t_tilde=0.00510;
  points_for_kriging[0].rho=0.0;
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
	double rho = 0.0; //points_for_kriging[i].rho;
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
	    
	  const BivariateSolverClassical* small_t_solution =
	    solver.get_small_t_solution();

	  gsl_matrix* Rotation_matrix = gsl_matrix_alloc(2,2);
	  gsl_matrix_memcpy(Rotation_matrix, small_t_solution->get_rotation_matrix());

	  double Rotation_matrix_inv [4];
	  gsl_matrix_view Rotation_matrix_inv_view =
	    gsl_matrix_view_array(Rotation_matrix_inv, 2, 2);

	  gsl_matrix_memcpy(&Rotation_matrix_inv_view.matrix,
			    small_t_solution->get_rotation_matrix());
  
	  int s = 0;
	  gsl_permutation * p = gsl_permutation_alloc(2);
	  gsl_linalg_LU_decomp(Rotation_matrix, p, &s);
	  gsl_linalg_LU_invert(Rotation_matrix, p, &Rotation_matrix_inv_view.matrix);
	  gsl_permutation_free(p);
	  gsl_matrix_free(Rotation_matrix);
  
	  gsl_vector * scaled_input = solver.scale_input(&raw_input.vector);
	  double corner_points_array [20] = {solver.get_a_2(), solver.get_b_2(), solver.get_b_2(), solver.get_a_2(), solver.get_x_0_2(), gsl_vector_get(scaled_input,0), solver.get_x_0_2(), solver.get_x_0_2(), solver.get_x_0_2(), solver.get_x_0_2(),
					     solver.get_c_2(), solver.get_c_2(), solver.get_d_2(), solver.get_d_2(), solver.get_y_0_2(), gsl_vector_get(scaled_input,1), solver.get_y_0_2(), solver.get_y_0_2(), solver.get_y_0_2(), solver.get_y_0_2()};
	  gsl_matrix_view corner_points_view = gsl_matrix_view_array(corner_points_array,
								     2, 10);
	  double corner_points_transformed_array [20];
	  gsl_matrix_view corner_points_transformed_view =
	    gsl_matrix_view_array(corner_points_transformed_array, 2, 10);
	  
	  double images_array [32];
	  for (unsigned i=0; i<16; ++i) {
	    images_array[i] = solver.get_x_0_2();
	    images_array[i+16] = solver.get_y_0_2();
	  }
	  double images_transformed_array [32];

	  gsl_matrix_view images_view = gsl_matrix_view_array(images_array, 2, 16);
	  gsl_matrix_view images_transformed_view = gsl_matrix_view_array(images_transformed_array, 2, 16);

	  printf("library(mvtnorm);\n");
	  printf("library(gridExtra);\n");
	  printf("library(MASS);\n");
	  printf("pdf(\"./src/kernel-expansion/documentation/chapter-3/illustration-rho-0-normalized.pdf\");\n");
	  printf("par(mar=c(2,2,0,0));\n");
	  printf("plot(x=0, type=\"n\", xlim = c(-0.5, 1.5), ylim=c(-0.5, 1.5), xlab=\"x\", ylab=\"y\");\n");
	  printf("lines(x=c(0,1), y=c(0,0), lwd=2, col=\"black\");\n"); // border 1
	  printf("lines(x=c(1,1), y=c(0,1), lwd=2, col=\"black\");\n"); // border 2
	  printf("lines(x=c(1,0), y=c(1,1), lwd=2, col=\"black\");\n"); // border 3
	  printf("lines(x=c(0,0), y=c(1,0), lwd=2, col=\"black\");\n"); // border 4
	  printf("points(x=%g, y=%g, lwd=3, pch=20, col=\"black\");\n",
		 solver.get_x_0_2(),
		 solver.get_y_0_2()); // IC
	  printf("samples <- rmvnorm(n=1e5, mean=c(%g,%g), sigma=diag(%g*c(1,%g)));\n",
		 solver.get_x_0_2(),
		 solver.get_y_0_2(),
		 solver.get_t_2(),
		 solver.get_sigma_y_2());
	  printf("z <- kde2d(samples[,1], samples[,2]);\n");
	  printf("contour(z, add=TRUE);\n");
	  printf("dev.off();\n");

	  // C = alpha*op(A)*op(B) + beta*C
	  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
			 CblasNoTrans, //op(B) = B
			 1.0, //alpha=1
			 small_t_solution->get_rotation_matrix(), //A
			 &corner_points_view.matrix, //B
			 0.0, //beta=0
			 &corner_points_transformed_view.matrix); //C

	  // C = alpha*op(A)*op(B) + beta*C
	  gsl_blas_dgemm(CblasNoTrans, //op(A) = A
			 CblasNoTrans, //op(B) = B
			 1.0, //alpha=1
			 small_t_solution->get_rotation_matrix(), //A
			 &images_view.matrix, //B
			 0.0, //beta=0
			 &images_transformed_view.matrix); //C

	  gsl_vector_view lower_left_transformed_view =
	    gsl_matrix_column(&corner_points_transformed_view.matrix,0);
	  gsl_vector_view lower_right_transformed_view =
	    gsl_matrix_column(&corner_points_transformed_view.matrix,1);
	  gsl_vector_view upper_right_transformed_view =
	    gsl_matrix_column(&corner_points_transformed_view.matrix,2);
	  gsl_vector_view upper_left_transformed_view =
	    gsl_matrix_column(&corner_points_transformed_view.matrix,3);
	  gsl_vector_view initial_condition_transformed_view =
	    gsl_matrix_column(&corner_points_transformed_view.matrix,4);

	  printf("pdf(\"./src/kernel-expansion/documentation/chapter-3/illustration-rho-0-transformed.pdf\");\n");
	  printf("par(mar=c(2,2,0,0));\n");
	  printf("lower.left.corner=c(%g,%g);\n",
		 gsl_vector_get(&lower_left_transformed_view.vector, 0),
		 gsl_vector_get(&lower_left_transformed_view.vector, 1));
	  printf("lower.right.corner=c(%g,%g);\n",
		 gsl_vector_get(&lower_right_transformed_view.vector, 0),
		 gsl_vector_get(&lower_right_transformed_view.vector, 1));
	  printf("upper.left.corner=c(%g,%g);\n",
		 gsl_vector_get(&upper_left_transformed_view.vector, 0),
		 gsl_vector_get(&upper_left_transformed_view.vector, 1));
	  printf("upper.right.corner=c(%g,%g);\n",
		 gsl_vector_get(&upper_right_transformed_view.vector, 0),
		 gsl_vector_get(&upper_right_transformed_view.vector, 1));

	  printf("plot(x=0, type=\"n\", xlim = 2*c(-max(abs(c(lower.left.corner, lower.right.corner, upper.left.corner, upper.right.corner))), max(abs(c(lower.left.corner, lower.right.corner, upper.left.corner, upper.right.corner)))), ylim = 2*c(-max(abs(c(lower.left.corner, lower.right.corner, upper.left.corner, upper.right.corner))), max(abs(c(lower.left.corner, lower.right.corner, upper.left.corner, upper.right.corner)))), xlab=\"x\", ylab=\"y\");\n");
	  printf("lines(x=c(lower.left.corner[1],lower.right.corner[1]), y=c(lower.left.corner[2],lower.right.corner[2]), lwd=2, col=\"black\");\n"); // border 1
	  printf("lines(x=c(lower.right.corner[1],upper.right.corner[1]), y=c(lower.right.corner[2],upper.right.corner[2]), lwd=2, col=\"black\");\n"); // border 2
	  printf("lines(x=c(upper.right.corner[1],upper.left.corner[1]), y=c(upper.right.corner[2],upper.left.corner[2]), lwd=2, col=\"black\");\n"); // border 3
	  printf("lines(x=c(upper.left.corner[1],lower.left.corner[1]), y=c(upper.left.corner[2],lower.left.corner[2]), lwd=2, col=\"black\");\n"); // border 4
	  printf("points(x=%g, y=%g, lwd=3, pch=20, col=\"black\");\n",
		 gsl_vector_get(&initial_condition_transformed_view.vector, 0),
		 gsl_vector_get(&initial_condition_transformed_view.vector, 1)); // IC
	  printf("samples <- rmvnorm(n=1e5, mean=c(%g,%g), sigma=diag(%g*c(1,%g)));\n",
		 gsl_vector_get(&initial_condition_transformed_view.vector, 0),
		 gsl_vector_get(&initial_condition_transformed_view.vector, 1),
		 solver.get_t_2(),
		 1.0);
	  printf("z <- kde2d(samples[,1], samples[,2]);\n");
	  printf("contour(z, add=TRUE);\n");

	  std::vector<std::vector<gsl_vector*>> lines {
	    std::vector<gsl_vector*> {&lower_left_transformed_view.vector,
		&lower_right_transformed_view.vector}, // line 1
	      std::vector<gsl_vector*> {&upper_right_transformed_view.vector,
		  &lower_right_transformed_view.vector}, // line 2
		std::vector<gsl_vector*> {&upper_left_transformed_view.vector,
		    &upper_right_transformed_view.vector}, // line 3
		  std::vector<gsl_vector*> {&upper_left_transformed_view.vector,
		      &lower_left_transformed_view.vector} // line 4
	  };

	  std::vector<double> distance_to_line {
	    small_t_solution->
	      distance_from_point_to_axis_raw(lines[0][0],
					      lines[0][1],
					      &initial_condition_transformed_view.vector,
					      &initial_condition_transformed_view.vector),
	      small_t_solution->
	      distance_from_point_to_axis_raw(lines[1][0],
					      lines[1][1],
					      &initial_condition_transformed_view.vector,
					      &initial_condition_transformed_view.vector),
	      small_t_solution->
	      distance_from_point_to_axis_raw(lines[2][0],
					      lines[2][1],
					      &initial_condition_transformed_view.vector,
					      &initial_condition_transformed_view.vector),
	      small_t_solution->
	      distance_from_point_to_axis_raw(lines[3][0],
					      lines[3][1],
					      &initial_condition_transformed_view.vector,
					      &initial_condition_transformed_view.vector)};

	  std::vector<unsigned> distance_to_line_indeces (4);
	  unsigned n=0;
	  std::generate(distance_to_line_indeces.begin(),
			distance_to_line_indeces.end(),
			[&n]{ return n++; });

	  std::sort(distance_to_line_indeces.begin(), distance_to_line_indeces.end(),
		    [&distance_to_line] (unsigned i1, unsigned i2) -> bool
		    {
		      return distance_to_line[i1] < distance_to_line[i2];
		    });

	  std::vector<gsl_vector_view> images_vector (16);

	    for (unsigned i=0; i<16; ++i) {
	      images_vector[i] =
		gsl_matrix_column(&images_transformed_view.matrix,i);
	    }

	    unsigned counter = 0;
	    for (unsigned l=0; l<2; ++l) {
	      for (unsigned k=0; k<2; ++k) {
		for (unsigned j=0; j<2; ++j) {
		  for (unsigned i=0; i<2; ++i) {
		    gsl_vector* current_image = &images_vector[counter].vector;
		    if (i==1) {
		      small_t_solution->reflect_point_raw(lines[distance_to_line_indeces[0]][0],
							   lines[distance_to_line_indeces[0]][1],
							   current_image);
		    }
		    if (j==1) {
		      small_t_solution->reflect_point_raw(lines[distance_to_line_indeces[1]][0],
							   lines[distance_to_line_indeces[1]][1],
							   current_image);
		    }
		    if (k==1) {
		      small_t_solution->reflect_point_raw(lines[distance_to_line_indeces[2]][0],
							   lines[distance_to_line_indeces[2]][1],
							   current_image);
		    }
		    if (l==1) {
		      small_t_solution->reflect_point_raw(lines[distance_to_line_indeces[3]][0],
							   lines[distance_to_line_indeces[3]][1],
							   current_image);
		    }
		    counter = counter + 1;
		    printf("## image %i distances: ", counter);

		    double d1 = small_t_solution->
		      distance_from_point_to_axis_raw(lines[distance_to_line_indeces[0]][0],
						      lines[distance_to_line_indeces[0]][1],
						      &initial_condition_transformed_view.vector,
						      current_image);
		    double d2 = small_t_solution->
		      distance_from_point_to_axis_raw(lines[distance_to_line_indeces[1]][0],
						      lines[distance_to_line_indeces[1]][1],
						      &initial_condition_transformed_view.vector,
						      current_image);
		    double d3 = small_t_solution->
		      distance_from_point_to_axis_raw(lines[distance_to_line_indeces[2]][0],
						      lines[distance_to_line_indeces[2]][1],
						      &initial_condition_transformed_view.vector,
						      current_image);
		    double d4 = small_t_solution->
		      distance_from_point_to_axis_raw(lines[distance_to_line_indeces[3]][0],
						      lines[distance_to_line_indeces[3]][1],
						      &initial_condition_transformed_view.vector,
						      current_image);
		    printf("%g %g %g %g\n", d1,d2,d3,d4);
	  
		  }
		}
	      }
	    }

	    printf("\nimage.1=c(%g,%g); image.2=c(%g,%g); image.3=c(%g,%g); image.4=c(%g,%g);\n",
		   gsl_vector_get(&images_vector[0].vector,0),
		   gsl_vector_get(&images_vector[0].vector,1),
		   gsl_vector_get(&images_vector[1].vector,0),
		   gsl_vector_get(&images_vector[1].vector,1),
		   gsl_vector_get(&images_vector[2].vector,0),
		   gsl_vector_get(&images_vector[2].vector,1),
		   gsl_vector_get(&images_vector[3].vector,0),
		   gsl_vector_get(&images_vector[3].vector,1));
	    // printf("points(image.1[1], image.1[2],lwd=10, pch=20,col=\"green\");\n");
	    // printf("points(image.2[1], image.2[2],lwd=10, pch=20,col=\"green\");\n");
	    // printf("points(image.3[1], image.3[2],lwd=10, pch=20,col=\"green\");\n");
	    // printf("points(image.4[1], image.4[2],lwd=10, pch=20,col=\"green\");\n");
	    printf("\nimage.5=c(%g,%g); image.6=c(%g,%g); image.7=c(%g,%g); image.8=c(%g,%g);\n",
		   gsl_vector_get(&images_vector[4].vector,0),
		   gsl_vector_get(&images_vector[4].vector,1),
		   gsl_vector_get(&images_vector[5].vector,0),
		   gsl_vector_get(&images_vector[5].vector,1),
		   gsl_vector_get(&images_vector[6].vector,0),
		   gsl_vector_get(&images_vector[6].vector,1),
		   gsl_vector_get(&images_vector[7].vector,0),
		   gsl_vector_get(&images_vector[7].vector,1));
	    // printf("points(image.5[1], image.5[2],lwd=10, pch=20,col=\"green\");\n");
	    // printf("points(image.6[1], image.6[2],lwd=10, pch=20,col=\"green\");\n");
	    // printf("points(image.7[1], image.7[2],lwd=10, pch=20,col=\"green\");\n");
	    // printf("points(image.8[1], image.8[2],lwd=10, pch=20,col=\"green\");\n");

	    printf("\nimage.9=c(%g,%g); image.10=c(%g,%g); image.11=c(%g,%g); image.12=c(%g,%g);\n",
		   gsl_vector_get(&images_vector[8].vector,0),
		   gsl_vector_get(&images_vector[8].vector,1),
		   gsl_vector_get(&images_vector[9].vector,0),
		   gsl_vector_get(&images_vector[9].vector,1),
		   gsl_vector_get(&images_vector[10].vector,0),
		   gsl_vector_get(&images_vector[10].vector,1),
		   gsl_vector_get(&images_vector[11].vector,0),
		   gsl_vector_get(&images_vector[11].vector,1));
	    // printf("points(image.9[1], image.9[2],lwd=10, pch=20,col=\"green\");\n");
	    // printf("points(image.10[1], image.10[2],lwd=10, pch=20,col=\"green\");\n");
	    // printf("points(image.11[1], image.11[2],lwd=10, pch=20,col=\"green\");\n");
	    // printf("points(image.12[1], image.12[2],lwd=10, pch=20,col=\"green\");\n");
	    printf("\nimage.13=c(%g,%g); image.14=c(%g,%g); image.15=c(%g,%g); image.16=c(%g,%g);\n",
		   gsl_vector_get(&images_vector[12].vector,0),
		   gsl_vector_get(&images_vector[12].vector,1),
		   gsl_vector_get(&images_vector[13].vector,0),
		   gsl_vector_get(&images_vector[13].vector,1),
		   gsl_vector_get(&images_vector[14].vector,0),
		   gsl_vector_get(&images_vector[14].vector,1),
		   gsl_vector_get(&images_vector[15].vector,0),
		   gsl_vector_get(&images_vector[15].vector,1));
	    // printf("points(image.13[1], image.13[2],lwd=10, pch=20,col=\"green\");\n");
	    // printf("points(image.14[1], image.14[2],lwd=10, pch=20,col=\"green\");\n");
	    // printf("points(image.15[1], image.15[2],lwd=10, pch=20,col=\"green\");\n");
	    // printf("points(image.16[1], image.16[2],lwd=10, pch=20,col=\"green\");\n");

	    // printf("points(image.16[1], image.16[2],lwd=10, pch=20,col=\"red\");\n");
	    printf("dev.off();\n");

	    unsigned M = 10;
	    double sigma_max = 1.0;
	    std::vector<double> sigma_tildes(M);
	    double sigma = 0.0;
	    double dsigma = sigma_max/M;
	    std::generate(sigma_tildes.begin(), sigma_tildes.end(), [&] ()->double {sigma = sigma + dsigma; return sigma; });

	    for (double sigma_tilde : sigma_tildes) {
	      solver.set_diffusion_parameters(1.0,
					      sigma_tilde,
					      rho);
	      double out_analytic = solver.analytic_likelihood_ax(&raw_input.vector, 10000);
	      double out_numeric = solver.numerical_likelihood_first_order_small_t(&raw_input.vector,
										   1,
										   dx_likelihood);
	      double before = solver.analytic_solution(scaled_input);
	      double CC = 1.0/( 2.0*(1-solver.get_rho()*solver.get_rho())*
				solver.get_t()*solver.get_sigma_y()*solver.get_sigma_y() );
	      printf("\n## (%g,%g,%g)\n\n",
	      	     sigma_tilde,
		     out_analytic,
		     out_numeric);
		     
	    }
	    printf("\n");



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
