#include <vector>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <vector>
#include <cstring>
#include <cfloat>
#include <random>
#include <unistd.h>
#include <omp.h>

#define Nvec 128
#include "Eigen/Eigenvalues"
using namespace std;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;

Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver;
bool verbose = false;

#define Complex complex<double>
#include "linAlgHelpers.h"
#include "algoHelpers.h"

int main(int argc, char **argv) {

  //Define the problem
  if (argc < 6 || argc > 6) {
    cout << "Compiled for Nvec = " << Nvec << endl;
    cout << "./lanczos <nKr> <nEv> <check-interval> <diag> <tol>" << endl;
    exit(0);
  }
  
  int nKr = atoi(argv[1]);
  int nEv = atoi(argv[2]);
  int check_interval = atoi(argv[3]);
  double diag = atof(argv[4]);
  double tol = atof(argv[5]);

  //Construct a matrix using Eigen.
  //---------------------------------------------------------------------  
  MatrixXcd ref = MatrixXcd::Random(Nvec, Nvec);
  
  //Copy Eigen ref matrix.
  Complex **mat = (Complex**)malloc(Nvec*sizeof(Complex*));
  for(int i=0; i<Nvec; i++) {
    mat[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    ref(i,i) = Complex(diag,0.0);
    mat[i][i] = ref(i,i);
    
    for(int j=0; j<i; j++) {
      mat[j][i] = ref(i,j);
      mat[i][j] = conj(ref(i,j));
      ref(j,i)  = conj(ref(i,j));
    }
  }

  //Eigensolve the matrix using Eigen, use as a reference.
  //---------------------------------------------------------------------  
  printf("START EIGEN SOLUTION\n");
  double t1 = clock();  
  Eigen::SelfAdjointEigenSolver<MatrixXcd> eigenSolver(ref);
  Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolverTD;
  double t2e = clock() - t1;
  printf("END EIGEN SOLUTION\n");
  printf("Time to solve problem using Eigen = %e\n", t2e/CLOCKS_PER_SEC);
  //-----------------------------------------------------------------------

  //Construct objects for Lanczos.
  //---------------------------------------------------------------------
  //Eigenvalues and their residuals
  double *residua          = (double*)malloc(nKr*sizeof(double));
  Complex *evals           = (Complex*)malloc(nKr*sizeof(Complex));
  double *mod_evals_sorted = (double*)malloc(nKr*sizeof(double));
  int *evals_sorted_idx    = (int*)malloc(nKr*sizeof(int));
  for(int i=0; i<nKr; i++) {
    residua[i]          = 0.0;
    evals[i]            = 0.0;
    mod_evals_sorted[i] = 0.0;
    evals_sorted_idx[i] = 0;
  }

  //Ritz vectors and Krylov Space. The eigenvectors will be stored here.
  std::vector<Complex*> kSpace(nKr+1);
  for(int i=0; i<nKr+1; i++) {
    kSpace[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    zero(kSpace[i]);
  }

  //Symmetric tridiagonal matrix
  double alpha[nKr];
  double  beta[nKr];
  for(int i=0; i<nKr; i++) {
    alpha[i] = 0.0;
    beta[i] = 0.0;
  }  

  //Residual vector. Also used as a temp vector
  Complex *r = (Complex*)malloc(Nvec*sizeof(Complex));

  printf("START LANCZOS SOLUTION\n");

  bool convergence = false;
  int num_converged = 0;
  double mat_norm = 0;  
  
  // Populate source with randoms.
  printf("Using random guess\n");
  for(int i=0; i<Nvec; i++) r[i] = drand48();

  //Normalise initial source
  normalise(r);
  
  //v_1
  copy(kSpace[0], r);

  t1 = clock();
  
  // START LANCZOS
  // Lanczos Method for Symmetric Eigenvalue Problems
  //-------------------------------------------------
  
  int j=0;
  while(!convergence && j < nKr) {
    
    lanczosStep(mat, kSpace, beta, alpha, r, -1, j, 1, 1);  

    if((j+1)%check_interval == 0) {

      //Compute the Tridiagonal matrix T_k (k = nKr)
      using Eigen::MatrixXd;
      MatrixXd triDiag = MatrixXd::Zero(j+1, j+1);
      
      for(int i=0; i<j+1; i++) {
	triDiag(i,i) = alpha[i];      
	if(i<j) {	
	  triDiag(i+1,i) = beta[i];	
	  triDiag(i,i+1) = beta[i];
	}
      }

      //Eigensolve the T_k matrix
      Eigen::Tridiagonalization<MatrixXd> TD(triDiag);
      eigenSolverTD.computeFromTridiagonal(TD.diagonal(), TD.subDiagonal());

      // mat_norm and rediua are updated.
      for (int i = 0; i < j+1; i++) {
	if (fabs(alpha[i]) > mat_norm) mat_norm = fabs(alpha[i]);
	residua[i] = fabs(beta[j] * eigenSolverTD.eigenvectors().col(i)[j]);
      }
      
      
      //Halting check
      if (nEv <= j+1) {
	num_converged = 0;
	for(int i=0; i<nEv; i++) {
	  if(residua[i] < tol * mat_norm) num_converged++;
	}	
	printf("%04d converged eigenvalues at iter %d\n", num_converged, j+1);	
	if (num_converged >= nEv) convergence = true;
	
      }
    }
    j++;
  }
  
  double t2l = clock() - t1;
  
  // Post computation report  
  if (!convergence) {    
    printf("lanczos failed to compute the requested %d vectors with a %d Krylov space\n", nEv, nKr);
  } else {
    printf("lanczos computed the requested %d vectors in %d steps in %e secs.\n", nEv, j, t2l/CLOCKS_PER_SEC);

    //Compute the Tridiagonal matrix
    using Eigen::MatrixXd;
    MatrixXd triDiag = MatrixXd::Zero(j, j);
    
    for(int i=0; i<j; i++) {
      triDiag(i,i) = alpha[i];      
      if(i<j-1) {	
	triDiag(i+1,i) = beta[i];	
	triDiag(i,i+1) = beta[i];
      }
    }

    //Eigensolve the T_k matrix
    Eigen::Tridiagonalization<MatrixXd> TD(triDiag);
    eigenSolverTD.computeFromTridiagonal(TD.diagonal(), TD.subDiagonal());
    
    // Compute eigenvalues
    rotateVecsReal(kSpace, eigenSolverTD.eigenvectors(), 0, j, j);
    computeEvals(mat, kSpace, residua, evals, nEv);
    for (int i = 0; i < nEv; i++) {
      printf("EigValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, evals[i].real(), evals[i].imag(), residua[i]);
    }
    
    for (int i = 0; i < nEv; i++) {
      printf("EigenComp[%04d]: %+.16e\n", i, (evals[i].real() - eigenSolver.eigenvalues()[i])/eigenSolver.eigenvalues()[i]); 
    }
  }
}
