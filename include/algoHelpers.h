#ifndef ALGOHELPERS_H
#define ALGOHELPERS_H

#include "linAlgHelpers.h"

std::vector<double> ritz_mat;

//Functions used in the lanczos algorithm
//---------------------------------------

//The Engine
void lanczosStep(Complex **mat, std::vector<Complex*> &kSpace,
		 double *beta, double *alpha,
		 Complex *r, int num_keep, int j) {

  //Compute r = A * v_j - b_{j-i} * v_{j-1}      
  //r = A * v_j
  matVec(mat, r, kSpace[j]);

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

void reorder(std::vector<Complex*> kSpace, double *alpha, int nKr) {
  int i = 0;
  while (i < nKr) {
    if ((i == 0) || (alpha[i - 1] <= alpha[i]))
      i++;
    else {
      double tmp = alpha[i];
      alpha[i] = alpha[i - 1];
      alpha[--i] = tmp;
      std::swap(kSpace[i], kSpace[i - 1]);
    }
  }
}

void eigensolveFromArrowMat(int num_locked, int arrow_pos, int nKr, double *alpha, double *beta, double *residua) {
  int dim = nKr - num_locked;

  // Eigen objects
  MatrixXd A = MatrixXd::Zero(dim, dim);
  ritz_mat.resize(dim * dim);
  for (int i = 0; i < dim * dim; i++) ritz_mat[i] = 0.0;
  
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
  Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver;
  eigensolver.compute(A);
  
  // repopulate ritz matrix
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      ritz_mat[dim * i + j] = eigensolver.eigenvectors().col(i)[j];
      //printf("%+.4e ",ritz_mat[dim * i + j]);      
    }
    //printf("\n");
  }
  
  for (int i = 0; i < dim; i++) {
    residua[i + num_locked] = fabs(beta[nKr - 1] * eigensolver.eigenvectors().col(i)[dim - 1]);
    // Update the alpha array
    alpha[i + num_locked] = eigensolver.eigenvalues()[i];
    if (verbose) printf("EFAM: resid = %e, alpha = %e\n", residua[i + num_locked], alpha[i + num_locked]);
  }
}

void computeEvals(Complex **mat, std::vector<Complex*> kSpace, double *residua, Complex *evals, int nEv) {
  
  //temp vector
  Complex *temp = (Complex*)malloc(Nvec*sizeof(Complex));
  for (int i = 0; i < nEv; i++) {
    // r = A * v_i
    //zero(temp);
    matVec(mat, temp, kSpace[i]);
    
    // lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
    evals[i] = cDotProd(kSpace[i], temp) / norm(kSpace[i]);
        
    // Measure ||lambda_i*v_i - A*v_i||
    Complex n_unit(-1.0, 0.0);
    caxpby(evals[i], kSpace[i], n_unit, temp);
    residua[i] = norm(temp);
  }
  free(temp);
}

void computeRitz(std::vector<Complex*> ritzVecs, Eigen::MatrixXd mat, int nEv, int nKr) {
  
  //loop over rows of V_k
  for(int j=0; j<Nvec; j++) {      
    
    //put jth row of V_k in temp
    Complex tmp[nKr];  
    for(int i=0; i<nKr; i++) {
      tmp[i] = ritzVecs[i][j];      
    }

    //take product of jth row of V_k and ith column of mat (ith eigenvector of T_k) 
    Complex sum = 0.0;
    for(int i=0; i<nEv; i++) {
      
      //Loop over elements to get the y_i[j]th element 
      for(int l=0; l<nKr; l++) {
	sum += tmp[l]*mat.col(i)[l];
      }  
      //Update the Ritz vector
      ritzVecs[i][j] = sum;
      sum = 0.0;
    }
  }
}

void computeKeptRitz(std::vector<Complex*> &kSpace, int nKr, int num_locked, int iter_keep, double *beta) {
  
  int offset = nKr + 1;
  int dim = nKr - num_locked;

  printf("Dim = %d\n", dim);
  printf("Iter keep = %d\n", iter_keep);
  printf("Numlocked = %d\n", num_locked);
  printf("kspace size = %d\n", (int)kSpace.size());

  // Batch this step to 10 eigenpairs per restart, do so as top
  // level parameter.
  if ((int)kSpace.size() < offset + iter_keep) {
    for (int i = kSpace.size(); i < offset + iter_keep; i++) {
      kSpace.push_back(new Complex[Nvec]);
      zero(kSpace[i]);
      if (verbose) printf("Adding %d vector to kSpace with norm %e\n", i, norm(kSpace[i]));
    }
  }
  printf("kspace size = %d\n", (int)kSpace.size());
  
  Complex *temp = (Complex*)malloc(Nvec*sizeof(Complex));
  
  for (int i = 0; i < iter_keep; i++) {
    int k = offset + i;
    //printf("copying kSpace[%d] to temp.\n",num_locked);

    // DMH num_locked is the position of teh new starting
    //     residual of the TRLM.
    copy(temp, kSpace[num_locked]);
    ax(ritz_mat[dim * i], temp);
    copy(kSpace[k], temp);
    //printf("copying temp to kSpace[%d].\n",k);    
    for (int j = 1; j < dim; j++) {
      axpy(ritz_mat[i * dim + j], kSpace[num_locked + j], kSpace[k]);
    }
  }
  
  for (int i = 0; i < iter_keep; i++) {
    copy(kSpace[i + num_locked], kSpace[offset + i]);
  }
  
  copy(kSpace[num_locked + iter_keep], kSpace[nKr]);

  //Update beta
  for (int i = 0; i < iter_keep; i++)
    beta[i + num_locked] = beta[nKr - 1] * ritz_mat[dim * (i + 1) - 1];
  
  free(temp);
}
/*
void computeKeptRitz(std::vector<Complex*> &kSpace, int nKr, int num_locked, int iter_keep, double *beta) {
  
  int offset = nKr + 1;
  int dim = nKr - num_locked;

  printf("Dim = %d\n", dim);
  printf("Iter keep = %d\n", iter_keep);
  printf("Numlocked = %d\n", num_locked);
  printf("kspace size = %d\n", (int)kSpace.size());

  // Batch this step to 10 eigenpairs per restart  
  if ((int)kSpace.size() < offset + iter_keep) {
    for (int i = kSpace.size(); i < offset + iter_keep; i++) {
      kSpace.push_back(new Complex[Nvec]);
      zero(kSpace[i]);
      if (verbose) printf("Adding %d vector to kSpace with norm %e\n", i, norm(kSpace[i]));
    }
  }
  printf("kspace size = %d\n", (int)kSpace.size());
  
  Complex *temp = (Complex*)malloc(Nvec*sizeof(Complex));
  
  for (int i = 0; i < iter_keep; i++) {
    int k = offset + i;
    //printf("copying kSpace[%d] to temp.\n",num_locked);    
    copy(temp, kSpace[num_locked]);
    ax(ritz_mat[dim * i], temp);
    copy(kSpace[k], temp);
    //printf("copying temp to kSpace[%d].\n",k);    
    for (int j = 1; j < dim; j++) {
      axpy(ritz_mat[i * dim + j], kSpace[num_locked + j], kSpace[k]);
    }
  }

  
  for (int i = 0; i < iter_keep; i++) {
    copy(kSpace[i + num_locked], kSpace[offset + i]);
  }
  
  copy(kSpace[num_locked + iter_keep], kSpace[nKr]);

  //Update beta
  for (int i = 0; i < iter_keep; i++)
    beta[i + num_locked] = beta[nKr - 1] * ritz_mat[dim * (i + 1) - 1];
  
  free(temp);
}
*/
#endif
