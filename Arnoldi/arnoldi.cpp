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

#define Nvec 64
#include "Eigen/Eigenvalues"
using namespace std;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;

Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver;
#define Complex complex<double>

bool verbose = false;

#include "linAlgHelpers.h"
#include "algoHelpers.h"

int main(int argc, char **argv) {

  //Define the problem
  if (argc < 5 || argc > 5) {
    cout << "Compiled for " << Nvec << endl;
    cout << "./arnoldi <nKr> <nEv> <check-interval> <tol>" << endl;
    exit(0);
  }
  
  int nKr = atoi(argv[1]);
  int nEv = atoi(argv[2]);
  int check_interval = atoi(argv[3]);
  double tol = atof(argv[4]);

  //Construct a matrix using Eigen.
  //---------------------------------------------------------------------  
  MatrixXcd ref = MatrixXcd::Random(Nvec, Nvec);
  
  //Copy Eigen ref matrix.
  Complex **mat = (Complex**)malloc(Nvec*sizeof(Complex*));
  for(int i=0; i<Nvec; i++) {
    mat[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    for(int j=0; j<Nvec; j++) {
      mat[i][j] = ref(i,j);
    }
  }
    
  //Eigensolve the matrix using Eigen, use as a reference.
  //---------------------------------------------------------------------  
  printf("START EIGEN SOLUTION\n");
  double t1 = clock();  
  Eigen::ComplexEigenSolver<MatrixXcd> eigenSolver(ref);
  Eigen::ComplexEigenSolver<MatrixXcd> eigenSolverUH;
  double t2e = clock() - t1;
  printf("END EIGEN SOLUTION\n");
  printf("Time to solve problem using Eigen = %e\n", t2e/CLOCKS_PER_SEC);
  //cout << eigenSolver.eigenvalues() << endl;
  //-----------------------------------------------------------------------

  //Construct objects for Arnoldi.
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

  //Upper Hessenberg matrix
  std::vector<Complex*> upperHess(nKr+1);
  for(int i=0; i<nKr+1; i++) {
    upperHess[i] = (Complex*)malloc((nKr+1)*sizeof(Complex));
    for(int j=0; j<nKr+1; j++) upperHess[i][j] = 0.0;
  }
  
  //Residual vector. Also used as a temp vector
  Complex *r = (Complex*)malloc(Nvec*sizeof(Complex));

  printf("START ARNOLDI SOLUTION\n");

  bool convergence = false;
  int num_converged = 0;
  double mat_norm = 0;  
  
  // Populate source with randoms.
  printf("Using random guess\n");
  //for(int i=0; i<Nvec; i++) r[i] = drand48();
  for(int i=0; i<Nvec; i++) r[i] = 1.0;

  //Normalise initial source
  //normalise(r);
  
  //v_1
  //copy(kSpace[0], r);

  t1 = clock();
  
  // START ARNOLDI
  // ARNOLDI Method for Complex Eigenvalue Problems
  //-------------------------------------------------
  
  int j=0;
  while(!convergence && j < nKr) {
    
    arnoldiStep(mat, kSpace, upperHess, r, j);
    
    if((j+1)%check_interval == 0) {
      
      //Construct the Upper Hessenberg matrix H_k      
      MatrixXcd upperHessEigen = MatrixXcd::Zero(j+1, j+1);      
      for(int k=0; k<j+1; k++) {
	for(int i=0; i<j+1; i++) {
	  upperHessEigen(k,i) = upperHess[k][i];
	}
      }

      //cout << upperHessEigen << endl << endl;
      
      // Eigensolve dense UH
      eigenSolverUH.compute(upperHessEigen);      
      
      // mat_norm and rediua are updated.
      for (int i = 0; i < j+1; i++) {
	mat_norm = upperHessEigen.norm();
	residua[i] = abs(upperHess[j+1][j] * eigenSolverUH.eigenvectors().col(i)[j]);
      }
      
      //Halting check
      if (nEv <= j+1) {
	num_converged = 0;
	for(int i=0; i<j+1; i++) {
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
    printf("Arnoldi failed to compute the requested %d vectors with a %d Krylov space\n", nEv, nKr);
  } else {
    printf("Arnoldi computed the requested %d vectors in %d steps in %e secs.\n", nEv, j, t2l/CLOCKS_PER_SEC);

    //Construct the Upper Hessenberg matrix H_k      
    MatrixXcd upperHessEigen = MatrixXcd::Zero(j, j);      
    for(int k=0; k<j; k++) {
      for(int i=0; i<j; i++) {
	upperHessEigen(k,i) = upperHess[k][i];
      }
    }  
    // Eigensolve dense UH
    eigenSolverUH.compute(upperHessEigen);

    // Compute Eigenvalues
    rotateVecsComplex(kSpace, eigenSolverUH.eigenvectors(), 0, j, j);
    computeEvals(mat, kSpace, residua, evals, j);
    for (int i = 0; i < nEv; i++) {
      int idx = j - 1 - i;
      printf("EigValue[%04d]: ||(%+.8e, %+.8e)|| = %+.8e residual %.8e\n", i, evals[idx].real(), evals[idx].imag(), abs(evals[idx]), residua[idx]);
    }
    
    for (int i = 0; i < nEv; i++) {
      int idx = j - 1 - i;
      int idx_e = Nvec - 1 - i;
      printf("EigenComp[%04d]: (%+.8e,%+.8e)\n", i, (evals[idx].real() - eigenSolver.eigenvalues()[idx_e].real())/eigenSolver.eigenvalues()[idx_e].real(), (evals[idx].imag() - eigenSolver.eigenvalues()[idx_e].imag())/eigenSolver.eigenvalues()[idx_e].imag());
    }
  }  
}
