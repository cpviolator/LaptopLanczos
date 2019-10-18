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

#define Nvec 2048
#include "Eigen/Eigenvalues"
using namespace std;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;

bool verbose = false;

#define Complex complex<double>
#include "linAlgHelpers.h"
#include "algoHelpers.h"

int main(int argc, char **argv) {

  //Define the problem
  if (argc < 7 || argc > 8) {
    cout << "Build for matrix size " << Nvec << endl;
    cout << "./trlm <nKr> <nEv> <max-restarts> <diag> <tol> <max_keep>" << endl;
    exit(0);
  }
  
  int nKr = atoi(argv[1]);
  int nEv = atoi(argv[2]);
  int max_restarts = atoi(argv[3]);
  double diag = atof(argv[4]);
  double tol = atof(argv[5]);
  int max_keep = atoi(argv[6]);

  if (!(nKr > nEv + 6)) {
    printf("nKr=%d must be greater than nEv+6=%d\n", nKr, nEv + 6);
    exit(0);
  }
  
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

  // Implictly Restarted Lanczos Method for Symmetric Eigenvalue Problems
  // Algorithm 4.7
  // http://www.netlib.org/utk/people/JackDongarra/etemplates/node118.html
  // Step n in the algorithm are denoted by (n) in the code.
  //----------------------------------------------------------------------

  printf("START LANCZOS SOLUTION\n");

  double epsilon = DBL_EPSILON;
  double mat_norm = 0.0;
  bool converged = false;
  int iter = 0;
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

  t1 = clock();
  
  // Loop over restart iterations.
  while(restart_iter < max_restarts && !converged) {
    
    // (2) p = m-k steps to get to the m-step factorisation
    for (int step = num_keep; step < nKr; step++) lanczosStep(mat, kSpace, beta, alpha, r, num_keep, step);
    iter += (nKr - num_keep);

    printf("Restart %d complete\n", restart_iter+1);
    
    int arrow_pos = std::max(num_keep - num_locked + 1, 2);

    // The eigenvalues are returned in the alpha array
    eigensolveFromArrowMat(num_locked, arrow_pos, nKr, alpha, beta, residua);

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
	printf("**** Locking %d resid=%+.6e condition=%.6e ****\n", i, residua[i + num_locked],
	       epsilon * mat_norm);
	iter_locked = i;
      } else {
	// Unlikely to find new locked pairs
	break;
      }
    }

    // Convergence check
    iter_converged = iter_locked;
    for (int i = iter_locked + 1; i < nKr - num_locked; i++) {
      if (residua[i + num_locked] < tol * mat_norm) {
	printf("**** Converged %d resid=%+.6e condition=%.6e ****\n", i, residua[i + num_locked], tol * mat_norm);
	iter_converged = i;
      } else {
	// Unlikely to find new converged pairs
	break;
      }
    }

    iter_keep = std::min(iter_converged + (nKr - num_converged) / 2, nKr - num_locked - 12);
    iter_keep = std::min(max_keep, iter_keep);

    computeKeptRitz(kSpace, nKr, num_locked, iter_keep, beta);
    
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
    if (num_converged >= nEv) {
      reorder(kSpace, alpha, nKr);
      converged = true;
    }
    
    restart_iter++;
    
  }

  t2e = clock() - t1;
  
  // Post computation report  
  if (!converged) {    
    printf("TRLM failed to compute the requested %d vectors with a %d search space and %d Krylov space in %d "
	   "restart steps.\n",
	   nEv, nEv, nKr, max_restarts);
  } else {
    printf("TRLM computed the requested %d vectors in %d restart steps and %d OP*x operations in %e secs.\n", nEv,
	   restart_iter, iter, t2e/CLOCKS_PER_SEC);
    
    // Dump all Ritz values and residua
    for (int i = 0; i < nEv; i++) {
      printf("RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, alpha[i], 0.0, residua[i]);
    }
    
    //Array for eigenvalues
    Complex *evals = (Complex*)malloc(nEv*sizeof(Complex));
    // Compute eigenvalues
    computeEvals(mat, kSpace, residua, evals, nEv);
    for (int i = 0; i < nEv; i++) {
      printf("EigValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, evals[i].real(), evals[i].imag(),
	     residua[i]);
    }

    for (int i = 0; i < nEv; i++) {
      printf("EigenComp[%04d]: %+.16e\n", i, (evals[i].real() - eigenSolver.eigenvalues()[i])/eigenSolver.eigenvalues()[i]);
    }
    free(evals);
  }
  free(r);
}
