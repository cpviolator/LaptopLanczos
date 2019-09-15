#ifndef ALGOHELPERS_H
#define ALGOHELPERS_H

#include "linAlgHelpers.h"

//Functions used in the lanczos algorithm
//---------------------------------------

//The Engine
void lanczosStep(Complex **mat, std::vector<Complex*> kSpace,
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
  beta[j] = norm(r);

  //Prepare next step.
  //v_{j+1} = r / b_j
  zero(kSpace[j+1]);
  axpy(1.0/beta[j], r, kSpace[j+1]);

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

#endif
