
#ifndef ALGOHELPERS_H

#define ALGOHELPERS_H

#include "linAlgHelpers.h"

//Functions used in the lanczos algorithm
//---------------------------------------

//The Engine
void lanczosStep(double **mat, double **krylovSpace,
		 double *beta, double *alpha,
		 double *r, int j) {
  
  //Compute r = A * v_j - b_{j-i} * v_{j-1}      
  //r = A * v_j
  matVec(mat, r, krylovSpace[j]);

  //r = r - b_{j-1} * v_{j-1}
  if(j>0) axpy(-beta[j-1], krylovSpace[j-1], r);      
  
  //a_j = v_j^dag * r
  alpha[j] = dotProd(krylovSpace[j], r);    
  
  //r = r - a_j * v_j
  axpy(-alpha[j], krylovSpace[j], r);

  //b_j = ||r|| 
  beta[j] = sqrt(norm(r));
  
  //Orthogonalise
  if(beta[j] < (1.0)*sqrt(alpha[j]*alpha[j] + beta[j-1]*beta[j-1])) {
    
    //The residual vector r has been deemed insufficiently
    //orthogonal to the existing Krylov space. We must
    //orthogonalise it.
    //printf("orthogonalising Beta %d = %e\n", j, beta[j]);
    orthogonalise(r, krylovSpace, j);
  }
  
  //b_j = ||r|| 
  beta[j] = sqrt(norm(r));
  
  //Prepare next step.
  //v_{j+1} = r / b_j
  zero(krylovSpace[j+1]);
  axpy(1.0/beta[j], r, krylovSpace[j+1]);
  
}

void computeRitz(double **ritzVecs, Eigen::MatrixXd mat, int nev, int nkv) {
  
  //loop over rows of V_k
  for(int j=0; j<Nvec; j++) {      
    
    //put jth row of V_k in temp
    double tmp[nkv];  
    for(int i=0; i<nkv; i++) {
      tmp[i] = ritzVecs[i][j];      
    }

    //take product of jth row of V_k and ith column of S (ith eigenvector of T_k) 
    double sum = 0.0;
    for(int i=0; i<nev; i++) {
      
      //Loop over elements to get the y_i[j]th element 
      for(int l=0; l<nkv; l++) {
	sum += tmp[l]*mat.col(i)[l];
      }      
      //Update the Ritz vector
      ritzVecs[i][j] = sum;
      sum = 0.0;
    }
  }
}

#endif
