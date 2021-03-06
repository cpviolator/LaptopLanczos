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

#define Nvec 1024
#include "Eigen/Eigenvalues"
using namespace std;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver;

bool verbose = false;

#define Complex complex<double>
#include "linAlgHelpers.h"
#include "lapack.h"
#include "algoHelpers.h"

int main(int argc, char **argv) {

  //Define the problem
  if (argc < 12 || argc > 12) {
    cout << "Built for matrix size " << Nvec << endl;
    cout << "./trlm <nKr> <nEv> <max-restarts> <diag> <tol> <amin> <amax> <poly> <spectrum: 0=LR, 1=SR> <LU> <batch>" << endl;
    exit(0);
  }
  
  int nKr = atoi(argv[1]);
  int nEv = atoi(argv[2]);
  int max_restarts = atoi(argv[3]);
  double diag = atof(argv[4]);
  double tol = atof(argv[5]);
  double a_min = atof(argv[6]);
  double a_max = atof(argv[7]);
  int poly = atoi(argv[8]);
  bool reverse = (atoi(argv[9]) == 0 ? true : false);
  bool LU = (atoi(argv[10]) == 1 ? true : false);
  int batch_size = atoi(argv[11]);
  
  if (!(nKr > nEv + 6)) {
    printf("nKr=%d must be greater than nEv+6=%d\n", nKr, nEv + 6);
    exit(0);
  }

  printf("Mat size = %d\n", Nvec);
  printf("nKr = %d\n", nKr);
  printf("nEv = %d\n", nEv);
  printf("Restarts = %d\n", max_restarts);
  printf("diag = %e\n", diag);
  printf("tol = %e\n", tol);
  printf("reverse = %s\n", reverse == true ? "true" :  "false");
  
  //Construct a matrix using Eigen.
  //---------------------------------------------------------------------  
  MatrixXcd ref = MatrixXcd::Random(Nvec, Nvec);
  
  // Copy to mat, make hermitean
  Complex **mat = (Complex**)malloc(Nvec*sizeof(Complex*));
  for(int i=0; i<Nvec; i++) {
    mat[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    ref(i,i) = Complex(diag, 0.0);
    mat[i][i] = ref(i,i);
    for(int j=0; j<i; j++) {
      mat[i][j] = ref(i,j);
      ref(j,i) = conj(ref(i,j));
      mat[j][i] = ref(j,i);
    }
  }
  
  //Eigensolve the matrix using Eigen, use as a reference.
  //---------------------------------------------------------------------  
  printf("START EIGEN SOLUTION\n");
  double t1 = clock();  
  Eigen::SelfAdjointEigenSolver<MatrixXcd> eigensolverRef(ref);
  for(int i=0; i<Nvec; i++) cout << i << " " << eigensolverRef.eigenvalues()[i] << endl;
  double t2e = clock() - t1;
  printf("END EIGEN SOLUTION\n");
  printf("Time to solve problem using Eigen = %e\n", t2e/CLOCKS_PER_SEC);
  //-----------------------------------------------------------------------

  //a_min = eigensolverRef.eigenvalues()[nKr+16];
  //a_max = eigensolverRef.eigenvalues()[Nvec-1]+1.0;

  //double a_min = 35;
  //double a_max = 150;
  
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

  // Implictly Restarted Lanczos Method for Symmetric Eigenvalue Problems
  // Algorithm 4.7
  // http://www.netlib.org/utk/people/JackDongarra/etemplates/node118.html
  // Step n in the algorithm are denoted by (n) in the code.
  //----------------------------------------------------------------------

  printf("START LANCZOS SOLUTION\n");

  double tol_partial = tol;
  double epsilon = DBL_EPSILON;
  double mat_norm = 0.0;
  bool partial_converged = false;
  bool full_converged = false;
  int OP = 0;
  int restart_iter = 0;
  int iter_converged = 0;
  int iter_locked = 0;
  int iter_keep = 0;
  int num_converged = 0;
  int num_locked = 0;
  int num_keep = 0;
  
  // Populate source with randoms.
  printf("Using random guess\n");
  for(int i=0; i<Nvec; i++) r[i] = drand48();

  //Normalise initial source
  normalise(r);
  
  //v_1
  copy(kSpace[0], r);

  int nKrtop = nKr;
  int nEvtop = nEv;
  int nKr0 = nKr / 4;
  int nEv0 = nEv / 4;  
  //nKr /= 4;
  //nEv /= 4;
  
  t1 = clock();
  string test = "test";
  saveTRLMSolverState(kSpace, alpha, beta, nKr, test);
  while(!full_converged) {

    //loadTRLMSolverState(kSpace, alpha, beta, nKr, test);    
    // Loop over restart iterations.
    while(restart_iter < max_restarts && !partial_converged) {
      
      // (2) p = m-k steps to get to the m-step factorisation
      for (int step = num_keep; step < nKr; step++) lanczosStep(mat, kSpace, beta, alpha, r, num_keep, step, a_min, a_max, poly);
      OP += poly*(nKr - num_keep);

      printf("Restart %d complete\n", restart_iter+1);
      
      int arrow_pos = std::max(num_keep - num_locked + 1, 2);
      
      // The eigenvalues are returned in the alpha array
      eigensolveFromArrowMat(num_locked, arrow_pos, nKr, alpha, beta, residua, reverse);
      
      // mat_norm is updated.
      for (int i = num_locked; i < nKr; i++) {
	if (verbose) printf("fabs(alpha[%d]) = %e  :  mat norm = %e\n", i, fabs(alpha[i]), mat_norm);
	if (fabs(alpha[i]) > mat_norm) {
	  mat_norm = fabs(alpha[i]);
	}
      }
      
      // Locking check
      iter_locked = 0;
      for (int i = 1; i < (nKr - num_locked); i++) {
	if (residua[i + num_locked] < epsilon * mat_norm) {
	  printf("**** Locking %d resid=%+.6e condition=%.6e ****\n", i, residua[i + num_locked], epsilon * mat_norm);
	  iter_locked = i;
	} else {
	  // Unlikely to find new locked pairs
	  break;
	}
      }

      // Convergence check
      iter_converged = iter_locked;
      for (int i = iter_locked + 1; i < nKr - num_locked; i++) {
	if (residua[i + num_locked] < tol_partial * mat_norm) {
	  printf("**** Converged %d resid=%+.6e condition=%.6e ****\n", i, residua[i + num_locked], tol_partial * mat_norm);
	  iter_converged = i;
	} else {
	  // Unlikely to find new converged pairs
	  break;
	}
      }

      iter_keep = std::min(iter_converged + (nKr - num_converged) / 2, nKr - num_locked - 12);
      //iter_keep = std::min(max_keep, iter_keep);

      if(!LU) computeKeptRitz(kSpace, nKr, num_locked, iter_keep, beta);
      else computeKeptRitzLU(kSpace, nKr, num_locked, iter_keep, batch_size, beta);
    
      num_converged = num_locked + iter_converged;
      num_keep = num_locked + iter_keep;
      num_locked += iter_locked;

      printf("iter Conv = %d\n", iter_converged);
      printf("iter Keep = %d\n", iter_keep);
      printf("iter Lock = %d\n", iter_locked);
      printf("%04d converged eigenvalues at restart iter %04d\n", num_converged, restart_iter + 1);
      printf("num_converged = %d\n", num_converged);
      printf("num_keep = %d\n", num_keep);
      printf("num_locked = %d\n", num_locked);
     
      // Check for convergence
      if (num_converged >= nEv && tol_partial <= tol) {
	reorder(kSpace, alpha, nKr, reverse);
	//saveTRLMSolverState(kSpace, alpha, beta, nKr, test);
	full_converged = true;
	partial_converged = true;
	//converged = true;
      }
      if(tol_partial > tol) {
	tol_partial *= 1e-2;
	epsilon *= 1e-2;
	num_converged = 0;
	if(nKr < nKrtop) nKr += nKr0;
	if(nEv < nEvtop) nEv += nEv0;
	cout << nKr << " " << nEv << endl;
	//num_keep = 0;
	//num_locked = 0;
	///iterRefineReal(kSpace, r, alpha, beta, nKr-1);
      }
      restart_iter++;
    }
    
    tol_partial *= 1e-1;
    epsilon *= 1e-1;
    if(tol_partial < tol) full_converged = true;
    else {
      cout << "tol_partial satisfied. Tightening it up!" << endl;
      partial_converged = false;
      poly -= 0;
    }
  }


  
  t2e = clock() - t1;
  
  // Post computation report  
  if (!full_converged) {    
    printf("TRLM failed to compute the requested %d vectors with a %d search space and %d Krylov space in %d "
	   "restart steps.\n",
	   nEv, nEv, nKr, max_restarts);
  } else {
    printf("TRLM computed the requested %d vectors in %d restart steps and %d OP*x operations in %e secs.\n", nEv,
	   restart_iter, OP, t2e/CLOCKS_PER_SEC);
    
    // Dump all Ritz values and residua
    for (int i = 0; i < nEv; i++) {
      printf("RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, alpha[i], 0.0, residua[i]);
    }
    
    // Compute eigenvalues
    computeEvals(mat, kSpace, residua, evals, nKr);
    for (int i = 0; i < nKr; i++) alpha[i] = evals[i].real();
    //reorder(kSpace, alpha, nEv, reverse);
    for (int i = 0; i < nEv; i++) {
      printf("EigValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, evals[i].real(), evals[i].imag(),
	     residua[i]);
    }

    for (int i = 0; i < nEv; i++) {
      //int idx = reverse ? (Nvec-1) - i : i;      
      int idx = i;
      printf("EigenComp[%04d]: %+.16e\n", i, (evals[i].real() - eigensolverRef.eigenvalues()[idx])/eigensolverRef.eigenvalues()[idx]);
    }
    free(evals);
  }
  free(r);
}
