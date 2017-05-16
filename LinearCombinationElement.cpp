#include <algorithm>
#include "BasisElementTypes.hpp"
#include <chrono>
#include <fstream>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_linalg.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <string>

// ============== LINEAR COMBINATION ELEMENT =====================
BivariateLinearCombinationElement::BivariateLinearCombinationElement()
  : dx_(1),
    function_grid_(gsl_matrix_alloc(2,2)),
    deriv_function_grid_dx_(gsl_matrix_alloc(2,2)),
    deriv_function_grid_dy_(gsl_matrix_alloc(2,2))
{}

BivariateLinearCombinationElement::
BivariateLinearCombinationElement(const std::vector<const BivariateElement*>& elements,
				  const std::vector<double>& coefficients)
  : dx_(elements[0]->get_dx()),
    function_grid_(gsl_matrix_alloc(1/elements[0]->get_dx() + 1,
    				    1/elements[0]->get_dx() + 1)),
    deriv_function_grid_dx_(gsl_matrix_alloc(1/elements[0]->get_dx() + 1,
					     1/elements[0]->get_dx() + 1)),
    deriv_function_grid_dy_(gsl_matrix_alloc(1/elements[0]->get_dx() + 1,
					     1/elements[0]->get_dx() + 1))
{
  if (elements.size() != coefficients.size()) {
    std::cout << "ERROR: elements and coefficients not of same size!"
	      << std::endl;
  }

  set_function_grids(elements, coefficients);
}

BivariateLinearCombinationElement::
BivariateLinearCombinationElement(const BivariateLinearCombinationElement& element)
  : dx_(element.get_dx()),
    function_grid_(gsl_matrix_alloc(element.function_grid_->size1,
				    element.function_grid_->size2)),
    deriv_function_grid_dx_(gsl_matrix_alloc(element.deriv_function_grid_dx_->size1,
					     element.deriv_function_grid_dx_->size2)),
    deriv_function_grid_dy_(gsl_matrix_alloc(element.deriv_function_grid_dy_->size1,
					     element.deriv_function_grid_dy_->size2))
{
  gsl_matrix_memcpy(function_grid_, element.function_grid_);
  gsl_matrix_memcpy(deriv_function_grid_dx_, element.deriv_function_grid_dx_);
  gsl_matrix_memcpy(deriv_function_grid_dy_, element.deriv_function_grid_dy_);
}

BivariateLinearCombinationElement& BivariateLinearCombinationElement::
operator=(const BivariateLinearCombinationElement& rhs)
{
  dx_ = rhs.get_dx();
  
  gsl_matrix_free(function_grid_);
  function_grid_ = gsl_matrix_alloc(rhs.function_grid_->size1,
				    rhs.function_grid_->size2);

  gsl_matrix_free(deriv_function_grid_dx_);
  deriv_function_grid_dx_ = gsl_matrix_alloc(1/rhs.get_dx() + 1,
					     1/rhs.get_dx() + 1);

  gsl_matrix_free(deriv_function_grid_dy_);
  deriv_function_grid_dy_ = gsl_matrix_alloc(1/rhs.get_dx() + 1,
					     1/rhs.get_dx() + 1);

  gsl_matrix_memcpy(function_grid_, rhs.function_grid_);
  gsl_matrix_memcpy(deriv_function_grid_dx_, rhs.deriv_function_grid_dx_);
  gsl_matrix_memcpy(deriv_function_grid_dy_, rhs.deriv_function_grid_dy_);

  return *this;
}

BivariateLinearCombinationElement::~BivariateLinearCombinationElement()
{
  gsl_matrix_free(function_grid_);
  gsl_matrix_free(deriv_function_grid_dx_);
  gsl_matrix_free(deriv_function_grid_dy_);
}

double BivariateLinearCombinationElement::norm() const
{
  int N = 1.0/dx_ + 1;
  double integral = 0;
  double row_sum = 0;

  for (int i=0; i<N; ++i) {
    gsl_vector_const_view row_i_1 =
      gsl_matrix_const_row(get_function_grid(),
			   i);
    gsl_vector_const_view row_i_2 =
      gsl_matrix_const_row(get_function_grid(),
			   i);
    gsl_blas_ddot(&row_i_1.vector, &row_i_2.vector, &row_sum);
    integral = integral + row_sum;
  }
  if (std::signbit(integral)) {
    integral = -1.0*std::exp(std::log(std::abs(integral)) + 2*std::log(dx_));
  } else {
    integral = std::exp(std::log(std::abs(integral)) + 2*std::log(dx_));
  }
  return sqrt(integral);
}

double BivariateLinearCombinationElement::
first_derivative(const gsl_vector* input,
		 long int coord_index) const
{
  double dx = get_dx();
  gsl_vector* input_plus = gsl_vector_alloc(2);
  gsl_vector_memcpy(input_plus, input);

  gsl_vector_set(input_plus, coord_index, 
		 gsl_vector_get(input, coord_index) + dx);

  double deriv = ((*this)(input_plus) - (*this)(input))/dx;
  gsl_vector_free(input_plus);

  return deriv;
}


void BivariateLinearCombinationElement::
set_function_grid(const gsl_matrix* new_function_grid) 
{
  gsl_matrix_memcpy(function_grid_, new_function_grid);
}

void BivariateLinearCombinationElement::
set_deriv_function_grid_dx(const gsl_matrix* new_deriv_function_grid_dx) 
{
  gsl_matrix_memcpy(deriv_function_grid_dx_, new_deriv_function_grid_dx);
}

void BivariateLinearCombinationElement::
set_deriv_function_grid_dy(const gsl_matrix* new_deriv_function_grid_dy) 
{
  gsl_matrix_memcpy(deriv_function_grid_dy_, new_deriv_function_grid_dy);
}

void BivariateLinearCombinationElement::
set_function_grids(const std::vector<const BivariateElement*>& elements,
		   const std::vector<double>& coefficients)
{
  // double dx = get_dx();
  // double in = 0;
  // double in_dx = 0;
  // double in_dy = 0;

  gsl_matrix* workspace_left = gsl_matrix_alloc(1/dx_ + 1, 1/dx_ + 1);
  gsl_matrix* workspace_right = gsl_matrix_alloc(1/dx_ + 1, 1/dx_ + 1);

  gsl_matrix_memcpy(function_grid_, elements[0]->get_function_grid());
  gsl_matrix_scale(function_grid_, coefficients[0]);
  
  gsl_matrix_memcpy(deriv_function_grid_dx_,
  		    elements[0]->get_deriv_function_grid_dx());
  gsl_matrix_scale(deriv_function_grid_dx_, coefficients[0]);

  gsl_matrix_memcpy(deriv_function_grid_dy_,
  		    elements[0]->get_deriv_function_grid_dy());
  gsl_matrix_scale(deriv_function_grid_dy_, coefficients[0]);

  for (unsigned k=1; k<elements.size(); ++k) {
    if ( std::abs(coefficients[k]) > 1e-32) {
      gsl_matrix_memcpy(workspace_right, elements[k]->get_function_grid());
      gsl_matrix_scale(workspace_right, coefficients[k]);
      gsl_matrix_add(function_grid_, workspace_right);
      
      gsl_matrix_memcpy(workspace_right, elements[k]->get_deriv_function_grid_dx());
      gsl_matrix_scale(workspace_right, coefficients[k]);
      gsl_matrix_add(deriv_function_grid_dx_, workspace_right);
      
      gsl_matrix_memcpy(workspace_right, elements[k]->get_deriv_function_grid_dy());
      gsl_matrix_scale(workspace_right, coefficients[k]);
      gsl_matrix_add(deriv_function_grid_dy_, workspace_right);
    }

    // if ( std::abs(coefficients[k]) > 1e-32) {
    //   gsl_matrix_scale(function_grid_, 1.0/coefficients[k]);
    //   gsl_matrix_add(function_grid_, elements[k]->get_function_grid());
    //   gsl_matrix_scale(function_grid_, coefficients[k]);

    //   gsl_matrix_scale(deriv_function_grid_dx_, 1.0/coefficients[k]);
    //   gsl_matrix_add(deriv_function_grid_dx_,
    //   		     elements[k]->get_deriv_function_grid_dx());
    //   gsl_matrix_scale(deriv_function_grid_dx_, coefficients[k]);

    //   gsl_matrix_scale(deriv_function_grid_dy_, 1.0/coefficients[k]);
    //   gsl_matrix_add(deriv_function_grid_dy_,
    //   		     elements[k]->get_deriv_function_grid_dy());
    //   gsl_matrix_scale(deriv_function_grid_dy_, coefficients[k]);
    // }
  }

  gsl_matrix_free(workspace_left);
  gsl_matrix_free(workspace_right);

  // for (int i=0; i<1/get_dx() + 1; ++i) {
  //   for (int j=0; j<1/get_dx() + 1; ++j) {
  //     in = 0;
  //     for (unsigned k=0; k<elements_.size(); ++k) {
  // 	in = in + coefficients[k]*
  // 	  gsl_matrix_get(elements_[k]->get_function_grid(),i,j);
  // 	// in_dx = in_dx + coefficients[k]*
  // 	//   gsl_matrix_get(elements_[k]->get_deriv_function_grid_dx(),i,j);
  // 	// in_dy = in_dy + coefficients[k]*
  // 	//   gsl_matrix_get(elements_[k]->get_deriv_function_grid_dx(),i,j);
  //     }
  //     gsl_matrix_set(function_grid_, i,j, in);
  //     // gsl_matrix_set(deriv_function_grid_dx_, i,j, in_dx);
  //     // gsl_matrix_set(deriv_function_grid_dy_, i,j, in_dy);
  //   }
  // }

}


// ==================== FOURIER INTERPOLANT =====================
BivariateLinearCombinationElementFourier::
BivariateLinearCombinationElementFourier()
  : BivariateLinearCombinationElement(),
    FFT_grid_(gsl_matrix_alloc(2*2,2))   
{}

BivariateLinearCombinationElementFourier::BivariateLinearCombinationElementFourier(const BivariateLinearCombinationElementFourier& element)
  : BivariateLinearCombinationElement(element),
    FFT_grid_(gsl_matrix_alloc(element.FFT_grid_->size1,
			       element.FFT_grid_->size2))
{
  gsl_matrix_memcpy(FFT_grid_, element.FFT_grid_);
}

BivariateLinearCombinationElementFourier::BivariateLinearCombinationElementFourier(const BivariateLinearCombinationElement& element)
  : BivariateLinearCombinationElement(element),
    FFT_grid_(gsl_matrix_alloc(1/get_dx(),
			       1/get_dx()))
{
  set_FFT_grid();
}

BivariateLinearCombinationElementFourier& BivariateLinearCombinationElementFourier::operator=(const BivariateLinearCombinationElementFourier& rhs)
{
    FFT_grid_ = gsl_matrix_alloc(rhs.FFT_grid_->size1,
				 rhs.FFT_grid_->size2);
    gsl_matrix_memcpy(FFT_grid_, rhs.FFT_grid_);

    return *this;
}

BivariateLinearCombinationElementFourier::~BivariateLinearCombinationElementFourier()
{
  gsl_matrix_free(FFT_grid_);
}

void BivariateLinearCombinationElementFourier::
set_function_grid(const gsl_matrix* new_function_grid)
{
  BivariateLinearCombinationElement::set_function_grid(new_function_grid);
  set_FFT_grid();
}

void BivariateLinearCombinationElementFourier::set_FFT_grid()
{
  
  int n = 1/get_dx();
  
  gsl_matrix_free(FFT_grid_);
  FFT_grid_ = gsl_matrix_alloc(2*n, n);

  // FFT FIRST PASS (ROWS) START
  for (unsigned i=0; i<n; ++i) {
    double fft_row [2 * n];
    for (unsigned j=0; j<n; ++j) {
      fft_row[2*j] = gsl_matrix_get(get_function_grid(), i, j);
      fft_row[2*j + 1] = 0.0;
    }

    gsl_fft_complex_radix2_forward(fft_row, 1, n);

    for (unsigned j=0; j<n; ++j) {
      gsl_matrix_set(FFT_grid_, 2*i, j, fft_row[2*j]);
      gsl_matrix_set(FFT_grid_, 2*i + 1, j, fft_row[2*j + 1]);
    }
  }
  // FFT FIRST PASS END

  // FFT SECOND PASS (COLUMNS) START
  for (unsigned j=0; j<n; ++j) {
    double fft_col [2 * n];
    gsl_vector_view fft_col_view = gsl_vector_view_array(fft_col, 2*n);
    gsl_vector_view fft_col_rhs = gsl_matrix_column(FFT_grid_,j);
    gsl_vector_memcpy(&fft_col_view.vector, &fft_col_rhs.vector);

    gsl_fft_complex_radix2_forward(fft_col, 1, n);

    gsl_vector_memcpy(&fft_col_rhs.vector, &fft_col_view.vector);
  }
  // FFT SECOND PASS END
}


void BivariateLinearCombinationElementFourier::
save_FFT_grid(std::string file_name) const
{

  std::ofstream output_file;
  output_file.open(file_name);
  output_file << std::fixed << std::setprecision(32);

  for (unsigned i=0; i<FFT_grid_->size1; ++i) {
    for (unsigned j=0; j<FFT_grid_->size2; ++j) {
      if (j==FFT_grid_->size2-1) 
	{
	  output_file << gsl_matrix_get(FFT_grid_, i,j) 
		      << "\n";
	} else {
	output_file << gsl_matrix_get(FFT_grid_, i,j) << ",";
      }
    }
  }
  output_file.close();
}
