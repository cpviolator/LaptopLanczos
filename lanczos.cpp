#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <vector>
#include <cstring>
#include <random>
#include <unistd.h>

#include "/Users/deanhowarth/lanczos/Eigen/Eigen/Eigenvalues"

using namespace std;

mt19937 rng(1235);
uniform_real_distribution<double> unif(0.0,1.0);
#define Nvec 512

//Simple Linear Algebra Helpers
void zero(double *vec) {
  for(int i=0; i<Nvec; i++) vec[i] = 0.0;
}

void copy(double *vec1, double *vec2) {
  for(int i=0; i<Nvec; i++) vec1[i] = vec2[i];
}

void ax(double C, double *vec) {
  for(int i=0; i<Nvec; i++) vec[i] *= C;
}

void axpy(double C, double *vec1, double *vec2) {
  for(int i=0; i<Nvec; i++) vec2[i] += C*vec1[i];
}

double dotProd(double *vec2, double *vec1) {
  double prod = 0.0;
  for(int i=0; i<Nvec; i++) prod += vec1[i]*vec2[i];
  return prod;
}

double norm(double *vec) {
  double sum = 0.0;
  for(int i=0; i<Nvec; i++) sum += vec[i]*vec[i];
  return sum;
}

void matVec(double **mat, double *out, double *in) {

  double tmp[Nvec];
  //Loop over rows of matrix
  for(int i=0; i<Nvec; i++) {
    tmp[i] = dotProd(mat[i], in);    
  }
  for(int i=0; i<Nvec; i++) {
    out[i] = tmp[i];
  }
}

//The Engine
void lanczosStep(double **mat, double **ritzVecs,
		 double *beta, double *alpha,
		 double *r, int j) {
  
  //Compute r = A * v_j - b_{j-i} * v_{j-1}      
  //r = A * v_j
  matVec(mat, r, ritzVecs[j]);

  //r = r - b_{j-1} * v_{j-1}
  if(j>0) axpy(-beta[j-1], ritzVecs[j-1], r);      
  
  //a_j = v_j^dag * r
  alpha[j] = dotProd(ritzVecs[j], r);    
  
  //r = r - a_j * v_j
  axpy(-alpha[j], ritzVecs[j], r);
  
  //b_j = ||r|| 
  beta[j] = sqrt(norm(r));
  
  //Prepare next step.
  //v_{j+1} = r / b_j
  zero(ritzVecs[j+1]);
  
  axpy(1.0/beta[j], r, ritzVecs[j+1]);

  //Orthonormalise
  if(j>0) {
    double s = 0.0;
    for(int i=0; i<j; i++) {
      s += dotProd(ritzVecs[i], r);
      axpy(-s, ritzVecs[i], r);
    }
  }  
}

void computeRitz(double **ritzVecs, Eigen::MatrixXd eigenSolverTD, int nkv) {
  
  //loop over rows of V_k
  for(int j=0; j<Nvec; j++) {      
    
    //put jth row of V_k in temp
    double tmp[nkv];  
    for(int i=0; i<nkv; i++) {
      tmp[i] = ritzVecs[i][j];      
    }

    //take product of jth row of V_k and ith column of S (ith eigenvector of T_k) 
    double sum = 0.0;
    for(int i=0; i<nkv; i++) {
      
      //Loop over elements to get the y_i[j]th element 
      for(int l=0; l<nkv; l++) {
	sum += tmp[l]*eigenSolverTD.col(i)[l];
      }      
      //Update the Ritz vector
      ritzVecs[i][j] = sum;
      sum = 0.0;
    }
  }
}

int main(int argc, char **argv) {
  
  int nkv = atoi(argv[1]);
  int nev = atoi(argv[2]);
  double diag = atof(argv[3]);

  int converged = 0;
  
  //double *mod_h_evals_sorted = (double*)malloc(nkv*sizeof(double));
  double *h_evals_resid      = (double*)malloc(nkv*sizeof(double));
  double *h_evals            = (double*)malloc(nkv*sizeof(double));
  //int *h_evals_sorted_idx    = (int*)malloc(nkv*sizeof(int));
  
  //Construct and solve a matrix using Eigen, use as a trusted reference.
  using Eigen::MatrixXd;
  MatrixXd ref = MatrixXd::Random(Nvec, Nvec);
  
  //Problem matrix
  double **mat = (double**)malloc(Nvec*sizeof(double*));  
  //Allocate space and populate
  for(int i=0; i<Nvec; i++) {
    mat[i] = (double*)malloc(Nvec*sizeof(double));
    ref(i,i) += diag;
    mat[i][i] = ref(i,i);
    
    for(int j=0; j<i; j++) {
      mat[i][j] = ref(i,j);
      mat[j][i] = mat[i][j];
      ref(j,i)  = ref(i,j);
    }
  }

  Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolver(ref);
  
  //Ritz vectors
  double **ritzVecs = (double**)malloc((1+nkv)*sizeof(double*));  
  //Allocate space for the Ritz vectors.
  for(int i=0; i<nkv+1; i++) {
    ritzVecs[i] = (double*)malloc(Nvec*sizeof(double));
    zero(ritzVecs[i]);
  }

  //Ortho locked
  bool *locked = (bool*)malloc((1+nkv)*sizeof(bool));
  for(int i=0; i<nkv+1; i++) locked[i] = false;
  
  //Tridiagonal matrix
  double alpha[nkv+1];
  double  beta[nkv+1];
  for(int i=0; i<nkv+1; i++) {
    alpha[i] = 0.0;
    beta[i] = 0.0;
  }  
  
  double *r = (double*)malloc(Nvec*sizeof(double));
  //Populate source with randoms.
  printf("Using random guess\n");
  for(int i=0; i<Nvec; i++) r[i] = drand48();
  //Ensure we are not trying to compute on a zero-field source
  double nor = norm(r);
  //Normalise initial source
  ax(1.0/sqrt(nor), r);
  //v_1
  copy(ritzVecs[0], r);
  
  // START LANCZOS
  // Lanczos Method for Symmetric Eigenvalue Problems
  // Based on Rudy Arthur's PHD thesis
  // Link as of 06 Oct 2018: https://www.era.lib.ed.ac.uk/bitstream/handle/1842/7825/Arthur2012.pdf
  //-----------------------------------------

  int restartIter = 1;
  
  while(converged<nev) {

    printf("Restart iteration %d\n", restartIter);    
    
    for(int j=0; j<nkv; j++) lanczosStep(mat, ritzVecs, beta, alpha, r, j);    
    //Compute the Tridiagonal matrix T_k (k = nkv)
    using Eigen::MatrixXd;
    MatrixXd triDiag = MatrixXd::Zero(nkv, nkv);
    MatrixXd QR = MatrixXd::Zero(nkv, nkv);
    
    for(int i=0; i<nkv; i++) {
      triDiag(i,i) = alpha[i];
      QR(i,i) = triDiag(i,i);
      
      if(i<nkv-1) {
	
	triDiag(i+1,i) = beta[i];
	QR(i+1,i) = triDiag(i+1,i);
	
	triDiag(i,i+1) = beta[i];
	QR(i,i+1) = triDiag(i,i+1);
      }
    }
    
    //Eigensolve T_k.
    Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolverTD(triDiag);
    //Ritz values are in ascending order if matrix is real.      
    //std::cout << eigenSolverTD.eigenvalues() << std::endl;
    
    //Perform Rayleigh-Ritz in-place matrix multiplication.
    //           y_i = V_k s_i
    // where T_k s_i = theta_i s_i  
    computeRitz(ritzVecs, eigenSolverTD.eigenvectors(), nkv);
    
    //Check for convergence    
    for(int i=0; i<nkv; i++) {
      
      //r = A * v_i
      matVec(mat, r, ritzVecs[i]);
      
      //lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
      h_evals[i] = dotProd(ritzVecs[i], r)/sqrt(norm(ritzVecs[i]));
      
      //Convergence check ||A * v_i - lambda_i * v_i||
      axpy(-h_evals[i], ritzVecs[i], r);
      h_evals_resid[i] =  sqrt(norm(r));
      
      if(h_evals_resid[i] < 1e-6) {
	if(locked[i] == false) converged++;
	locked[i] = true;
	
	printf("EigValue[%04d]: ref=%+.8e, comp=%+.8e, compres=%+.8e, "
	       "calcres=%+.8e, ratio=%+.8e, ||Evec||=%+.8e\n",
	       i, eigenSolver.eigenvalues()[i], h_evals[i], h_evals_resid[i],
	       beta[nkv-1]*eigenSolverTD.eigenvectors().col(i)[nkv-1],
	       h_evals_resid[i]/(beta[nkv-1]*eigenSolverTD.eigenvectors().col(i)[nkv-1]),
	       sqrt(norm(ritzVecs[i])));
      }
    }

    //Compute new resudial 
    
    restartIter++;
    
  }   
}
