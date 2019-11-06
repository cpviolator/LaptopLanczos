#ifndef ALGOHELPERS_H
#define ALGOHELPERS_H

#include "linAlgHelpers.h"

std::vector<double> ritz_mat;

//Functions used in the lanczos algorithm
//---------------------------------------

//The Engine
void lanczosStep(Complex **mat, std::vector<Complex*> &kSpace,
		 double *beta, double *alpha,
		 Complex *r, int num_keep, int j, double a_min, double a_max) {

  //Compute r = A * v_j - b_{j-i} * v_{j-1}      
  //r = A * v_j
  matVec(mat, r, kSpace[j]);
  //chebyOp(mat, r, kSpace[j], a_min, a_max);

  //a_j = v_j^dag * r
  alpha[j] = (cDotProd(kSpace[j], r)).real();    

  //r = r - a_j * v_j
  axpy(-alpha[j], kSpace[j], r);

  int start = (j > num_keep && j>0) ? j - 1 : 0;
  for (int i = start; i < j; i++) {

    // r = r - b_{j-1} * v_{j-1}
    axpy(-beta[i], kSpace[i], r);
  }

  // Orthogonalise r against the K space
  if (j > 0)
    for (int k = 0; k < 10; k++) orthogonalise(r, kSpace, j);

  //b_j = ||r|| 
  beta[j] = normalise(r);

  //Prepare next step.
  copy(kSpace[j+1], r);
}

void reorder(std::vector<Complex*> &kSpace, double *alpha, int nKr, bool reverse) {
  int i = 0;
  Complex temp[Nvec];
  if (reverse) {
    while (i < nKr) {
      if ((i == 0) || (alpha[i - 1] >= alpha[i]))
	i++;
      else {
	double tmp = alpha[i];
	alpha[i] = alpha[i - 1];
	alpha[--i] = tmp;
	copy(temp, kSpace[i]);
	copy(kSpace[i], kSpace[i-1]);
	copy(kSpace[i-1], temp);
      }
    }
  } else {
    while (i < nKr) {
      if ((i == 0) || (alpha[i - 1] <= alpha[i])) 
	i++;
      else {
	double tmp = alpha[i];
	alpha[i] = alpha[i - 1];
	alpha[--i] = tmp;
	copy(temp, kSpace[i]);
	copy(kSpace[i], kSpace[i-1]);
	copy(kSpace[i-1], temp);
      }
    }
  }
}

void eigensolveFromArrowMat(int num_locked, int arrow_pos, int nKr, double *alpha, double *beta, double *residua, bool reverse) {
  
  int dim = nKr - num_locked;
  
  // Eigen objects
  MatrixXd A = MatrixXd::Zero(dim, dim);
  ritz_mat.resize(dim * dim);
  for (int i = 0; i < dim * dim; i++) ritz_mat[i] = 0.0;

  // Optionally invert the spectrum
  if (reverse) {
    for (int i = num_locked; i < nKr - 1; i++) {
      alpha[i] *= -1.0;
      beta[i] *= -1.0;
    }
    alpha[nKr - 1] *= -1.0;
  }
  
  // Construct arrow mat A_{dim,dim}
  for (int i = 0; i < dim; i++) {    
    // alpha populates the diagonal
    A(i,i) = alpha[i + num_locked];
  }
  
  for (int i = 0; i < arrow_pos - 1; i++) {  
    // beta populates the arrow
    A(i, arrow_pos - 1) = beta[i + num_locked];
    A(arrow_pos - 1, i) = beta[i + num_locked];
  }
  
  for (int i = arrow_pos - 1; i < dim - 1; i++) {
    // beta populates the sub-diagonal
    A(i, i + 1) = beta[i + num_locked];
    A(i + 1, i) = beta[i + num_locked];
  }
  
  // Eigensolve the arrow matrix 
  Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolver;
  eigenSolver.compute(A);
  
  // repopulate ritz matrix
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      //Place data in COLUMN major 
      ritz_mat[dim * i + j] = eigenSolver.eigenvectors().col(i)[j];
      //printf("%+.4e ",ritz_mat[dim * i + j]);      
    }
    //printf("\n");
  }
  
  for (int i = 0; i < dim; i++) {
    residua[i + num_locked] = fabs(beta[nKr - 1] * eigenSolver.eigenvectors().col(i)[dim - 1]);
    // Update the alpha array
    alpha[i + num_locked] = eigenSolver.eigenvalues()[i];
    if (verbose) printf("EFAM: resid = %e, alpha = %e\n", residua[i + num_locked], alpha[i + num_locked]);
  }

  // Put spectrum back in order
  if (reverse) {
    for (int i = num_locked; i < nKr; i++) { alpha[i] *= -1.0; }
  }  
}

void computeEvals(Complex **mat, std::vector<Complex*> &kSpace, double *residua, Complex *evals, int nEv) {
  
  //temp vector
  Complex temp[Nvec];
  for (int i = 0; i < nEv; i++) {
    // r = A * v_i
    matVec(mat, temp, kSpace[i]);
    
    // lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
    evals[i] = cDotProd(kSpace[i], temp) / norm(kSpace[i]);
    
    // Measure ||lambda_i*v_i - A*v_i||
    Complex n_unit(-1.0, 0.0);
    caxpby(evals[i], kSpace[i], n_unit, temp);
    residua[i] = norm(temp);
  }
}

void rotateVecs(std::vector<Complex*> &vecs, Eigen::MatrixXd mat, int num_locked, int iter_keep, int dim) {

  //loop over rows of V_k
  for(int j=0; j<Nvec; j++) {    
    
    //put jth row of V_k in temp
    Complex tmp[dim];  
    for(int i=0; i<dim; i++) {
      tmp[i] = vecs[i+num_locked][j];      
    }

    //take product of jth row of V_k and ith column of mat (ith eigenvector of T_k) 
    Complex sum = 0.0;
    for(int i=0; i<iter_keep; i++) {
      
      //Loop over elements to get the y_i[j]th element 
      for(int l=0; l<dim; l++) {
	sum += tmp[l]*mat.col(i)[l];
      }
      
      //Update the Ritz vector
      vecs[i+num_locked][j] = sum;
      sum = 0.0;
    }
  }
}

void computeKeptRitz(std::vector<Complex*> &kSpace, int nKr, int num_locked, int iter_keep, double *beta) {
  
  int dim = nKr - num_locked;
  MatrixXd mat = MatrixXd::Zero(dim, iter_keep);
  for (int j = 0; j < iter_keep; j++) 
    for (int i = 0; i < dim; i++) 
      mat(i,j) = ritz_mat[j*dim + i];  
  rotateVecs(kSpace, mat, num_locked, iter_keep, dim); 
  
  //Update beta and residual
  copy(kSpace[num_locked + iter_keep], kSpace[nKr]);
  for (int i = 0; i < iter_keep; i++)
    beta[i + num_locked] = beta[nKr - 1] * mat.col(i)[nKr-num_locked-1];  
}

void permuteVecs(std::vector<Complex*> &kSpace, Eigen::MatrixXd mat, int num_locked, int size){

  std::vector<int> pivots(size);
  for (int i=0; i<size; i++) {
    for (int j=0; j<size; j++) {
      if(mat(i,j) == 1) {
	pivots[j] = i;
      }
    }
  }

  // Identify cycles in the permutation array.
  // We shall use the sign bit as a marker. If the
  // sign is negative, the vectors has already been
  // swapped into the correct place. A positive
  // value indicates the start of a new cycle.

  Complex temp[Nvec];
  for (int i=0; i<size; i++) {
    //Cycles always start at 0, hence OR statement
    if(pivots[i] > 0 || i==0) {
      int k = i;
      // Identify vector to be placed at i
      int j = pivots[i];
      pivots[i] = -pivots[i];
      while (j > i) {
	copy(temp, kSpace[k+num_locked]);
	copy(kSpace[k+num_locked], kSpace[j+num_locked]);
	copy(kSpace[j+num_locked], temp);
	pivots[j] = -pivots[j];
	k = j;
	j = -pivots[j];
      }
    } else {
      //printf("%d already swapped\n", i);
    }
  }
  for (int i=0; i<size; i++) {
    if (pivots[i] > 0) {
      printf("Error at %d\n", i);
      exit(0);
    }
  }
}

void computeKeptRitzLU(std::vector<Complex*> &kSpace, int nKr, int num_locked, int iter_keep, int batch, double *beta, int iter) {
  
  int offset = nKr + 1;
  int dim = nKr - num_locked;

  printf("dim = %d\n", dim);
  printf("iter_keep = %d\n", iter_keep);
  printf("num_locked = %d\n", num_locked);
  printf("kspace size = %d\n", (int)kSpace.size());

  int batch_size = batch;
  int full_batches = iter_keep/batch_size;
  int batch_size_r = iter_keep%batch_size;
  bool do_batch_remainder = (batch_size_r != 0 ? true : false);
  
  printf("batch_size = %d\n", batch_size);
  printf("full_batches = %d\n", full_batches);
  printf("batch_size_r = %d\n", batch_size_r);
  
  if ((int)kSpace.size() < offset + batch_size) {
    for (int i = kSpace.size(); i < offset + batch_size; i++) {
      kSpace.push_back(new Complex[Nvec]);
    }
  }

  // Zero out the work space
  for (int i = offset; i < offset + batch_size; i++) {
    zero(kSpace[i]);
  }

  MatrixXd mat = MatrixXd::Zero(dim, iter_keep);
  for (int j = 0; j < iter_keep; j++) 
    for (int i = 0; i < dim; i++) 
      mat(i,j) = ritz_mat[j*dim + i];
  
  Eigen::FullPivLU<MatrixXd> matLU(mat);
  // RitzLU now contains the LU decomposition
  
  MatrixXd matUpper = MatrixXd::Zero(dim,iter_keep);
  matUpper = matLU.matrixLU().triangularView<Eigen::Upper>();
  MatrixXd matLower = MatrixXd::Identity(dim,dim);
  matLower.block(0,0,dim,iter_keep).triangularView<Eigen::StrictlyLower>() = matLU.matrixLU();

  //rotateVecs(kSpace, matLU.permutationP().inverse(), num_locked, dim, dim);
  permuteVecs(kSpace, matLU.permutationP().inverse(), num_locked, dim);    
  
  // Defines the column element (row index)
  // from which we reference other indicies
  int i_start, i_end, j_start, j_end;
  
  // Do L Portion
  //---------------------------------------------------------------------------
  // Loop over full batches
  for (int b = 0; b < full_batches; b++) {

    // batch triangle
    i_start = b*batch_size;
    i_end   = (b+1)*batch_size; 
    j_start = i_start;
    j_end   = i_end;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = j; i < i_end; i++) {	
	axpy(matLower.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }
    //batch pencil
    i_start = (b+1)*batch_size;
    i_end   = dim;
    j_start = b*batch_size;
    j_end   = (b+1)*batch_size;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = i_start; i < i_end; i++) {
	axpy(matLower.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }
    // copy back to correct position
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      copy(kSpace[j + num_locked], kSpace[k]);
      zero(kSpace[k]);
    }    
  }
  
  if(do_batch_remainder) {
    // remainder triangle
    i_start = full_batches*batch_size;
    i_end   = iter_keep;
    j_start = i_start;
    j_end   = i_end;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = j; i < i_end; i++) {
	axpy(matLower.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }   
    }
    //remainder pencil
    i_start = iter_keep;
    i_end   = dim;
    j_start = full_batches*batch_size;
    j_end   = iter_keep;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = i_start; i < i_end; i++) {	
	axpy(matLower.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }
    // copy back to correct position
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      copy(kSpace[j + num_locked], kSpace[k]);
      zero(kSpace[k]);
    }
  }
    
  // Do U Portion
  //---------------------------------------------------------------------------
  if(do_batch_remainder) {

    // remainder triangle
    i_start = full_batches*batch_size;
    i_end   = iter_keep;
    j_start = i_start;
    j_end   = i_end;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = i_start; i < j+1; i++) {	
	axpy(matUpper.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }
    //remainder pencil
    i_start = 0;
    i_end   = full_batches*batch_size; 
    j_start = full_batches*batch_size;
    j_end   = iter_keep;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = i_start; i < i_end; i++) {
	axpy(matUpper.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }
    // copy back to correct position
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      copy(kSpace[j + num_locked], kSpace[k]);
      zero(kSpace[k]);
    }    
  }
  
  // Loop over full batches
  for (int b = full_batches-1; b >= 0; b--) {

    // batch triangle
    i_start = b*batch_size;
    i_end   = (b+1)*batch_size; 
    j_start = i_start;
    j_end   = i_end;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = i_start; i < j+1; i++) {	
	axpy(matUpper.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }

    if(b>0) {
      //batch pencil
      i_start = 0;
      i_end   = b*batch_size; 
      j_start = b*batch_size;
      j_end   = (b+1)*batch_size;
      for (int j = j_start; j < j_end; j++) {
	int k = offset + j - j_start;
	for (int i = i_start; i < i_end; i++) {	  
	  axpy(matUpper.col(j)[i], kSpace[num_locked + i], kSpace[k]);
	}
      }
    }
    // copy back to correct position
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      copy(kSpace[j + num_locked], kSpace[k]);
      zero(kSpace[k]);
    }    
  }

  permuteVecs(kSpace, matLU.permutationQ().inverse(), num_locked, iter_keep);
  //rotateVecs(kSpace, matLU.permutationQ().inverse(), num_locked, iter_keep, iter_keep);  

  //Update residual
  copy(kSpace[num_locked + iter_keep], kSpace[nKr]);
  
  //Update beta
  for (int i = 0; i < iter_keep; i++)
    beta[i + num_locked] = beta[nKr - 1] * ritz_mat[dim * (i + 1) - 1];
  
}
#endif
