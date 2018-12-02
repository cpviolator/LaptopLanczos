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
#include <omp.h>

#define Nvec 32
#include "Eigen/Eigenvalues"
#include "linAlgHelpers.h"
#include "algoHelpers.h"

using namespace std;

mt19937 rng(1235);
uniform_real_distribution<double> unif(0.0,1.0);

int main(int argc, char **argv) {
  
  int nkv = atoi(argv[1]);
  int nev = atoi(argv[2]);
  int check_interval = atoi(argv[3]);
  double diag = atof(argv[4]);
  double tol = atof(argv[5]);
  int threads = atoi(argv[6]);
  omp_set_num_threads(threads);
  Eigen::setNbThreads(threads);
  
  //double *mod_h_evals_sorted = (double*)malloc(nkv*sizeof(double));
  double *h_evals_resid      = (double*)malloc(nkv*sizeof(double));
  double *h_evals            = (double*)malloc(nkv*sizeof(double));
  
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

  printf("START EIGEN SOLUTION\n");
  double t1 = clock();  
  //Solve the problem matrix using Eigen, use as a reference.
  Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolver(ref);
  double t2e = clock() - t1;
  printf("END EIGEN SOLUTION\n");
  printf("Time to solve problem using Eigen   = %e\n", t2e/CLOCKS_PER_SEC);
  
  //Ritz vectors and Krylov Space
  double **ritzVecs = (double**)malloc((1+nkv)*sizeof(double*));
  double **krylovSpace = (double**)malloc((1+nkv)*sizeof(double*));  
  for(int i=0; i<nkv+1; i++) {
    ritzVecs[i] = (double*)malloc(Nvec*sizeof(double));
    krylovSpace[i] = (double*)malloc(Nvec*sizeof(double));
    zero(ritzVecs[i]);
    zero(krylovSpace[i]);
  }

  //Tracks if a Ritz vector is locked
  bool *locked = (bool*)malloc((1+nkv)*sizeof(bool));
  for(int i=0; i<nkv+1; i++) locked[i] = false;
  
  //Symmetric tridiagonal matrix
  double alpha[nkv+1];
  double  beta[nkv+1];
  for(int i=0; i<nkv+1; i++) {
    alpha[i] = 0.0;
    beta[i] = 0.0;
  }  

  //residual
  double *r = (double*)malloc(Nvec*sizeof(double));
  double *r_copy = (double*)malloc(Nvec*sizeof(double));
  //Populate source with randoms.
  printf("Using random guess\n");
  for(int i=0; i<Nvec; i++) r[i] = drand48();
  //Ensure we are not trying to compute on a zero-field source
  double nor = norm(r);
  //Normalise initial source
  ax(1.0/sqrt(nor), r);
  //v_1
  copy(krylovSpace[0], r);
  
  // START LANCZOS
  // Lanczos Method for Symmetric Eigenvalue Problems
  //-------------------------------------------------

  t1 = clock();
  printf("START LANCZOS SOLUTION\n");
  bool converged = false;
  int numConverged = 0;
  int j=0;
  while(!converged && j < nkv) {
    
    lanczosStep(mat, krylovSpace, beta, alpha, r, j);  

    if((j+1)%check_interval == 0) {
      
      int j_check = j+1;
      printf("Convergence Check at iter %04d gives ", j_check);
      
      for(int i=0; i<j_check; i++) copy(ritzVecs[i], krylovSpace[i]);
      copy(r_copy, r);
    
      //Compute the Tridiagonal matrix T_k (k = nkv)
      using Eigen::MatrixXd;
      MatrixXd triDiag = MatrixXd::Zero(j_check, j_check);
      
      for(int i=0; i<j_check; i++) {
	triDiag(i,i) = alpha[i];      
	if(i<j_check-1) {	
	  triDiag(i+1,i) = beta[i];	
	  triDiag(i,i+1) = beta[i];
	}
      }

      //Eigensolve the T_k matrix
      Eigen::Tridiagonalization<MatrixXd> TD(triDiag);
      Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolverTD;
      eigenSolverTD.computeFromTridiagonal(TD.diagonal(), TD.subDiagonal());
      
      //Ritz values are in ascending order if matrix is real.      
      //std::cout << eigenSolverTD.eigenvalues() << std::endl;
    
      //Perform Rayleigh-Ritz in-place matrix multiplication.
      //           y_i = V_k s_i
      // where T_k s_i = theta_i s_i  
      computeRitz(ritzVecs, eigenSolverTD.eigenvectors(), j_check, j_check);
    
      //Check for convergence    
      for(int i=0; i<j_check; i++) {
      
	//r = A * v_i
	matVec(mat, r_copy, ritzVecs[i]);
      
	//lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
	h_evals[i] = dotProd(ritzVecs[i], r_copy)/sqrt(norm(ritzVecs[i]));
      
	//Convergence check ||A * v_i - lambda_i * v_i||
	axpy(-h_evals[i], ritzVecs[i], r_copy);
	h_evals_resid[i] =  sqrt(norm(r_copy));
	
	if(h_evals_resid[i] < tol) {
	  locked[i] = true;
	}
      }

      //Halting check
      bool test = true;
      numConverged = 0;
      for(int i=0; i<nev; i++) {
	if(locked[i] == false) test = false;	
	else numConverged++;
      }
      printf("%04d converged eigenvalues.\n", numConverged);
      
      if(test == true) {
	for(int i=0; i<nev; i++) {
	  printf("EigValue[%04d]: EIGEN=%+.8e, LANCZOS=%+.8e, reldiff=%+.3e, compres=%+.3e, "
		 "calcres=%+.3e, ||Evec||=%+.3e\n",
		 i, eigenSolver.eigenvalues()[i], h_evals[i],
		 1.0-eigenSolver.eigenvalues()[i]/h_evals[i],
		 h_evals_resid[i],
		 beta[j_check-1]*eigenSolverTD.eigenvectors().col(i)[j_check-1],
		 sqrt(norm(ritzVecs[i])));
	}
	converged = true;
      }
    }
    j++;
  }
  if(!converged) {
    printf("Lanczos failed to compute the requested %d vectors in %d steps. Please either increase nkv or decrease nev\n", nev, nkv);
  }
  
  double t2l = clock() - t1;
  printf("END LANCZOS SOLUTION\n");
  printf("Time to solve problem using Eigen   = %e\n", t2e/CLOCKS_PER_SEC);
  printf("Time to solve problem using Lanczos (%d OMP threads) = %e\n", threads, t2l/(CLOCKS_PER_SEC*threads));
}
