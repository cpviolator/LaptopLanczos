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
#include <quadmath.h>

#define Nvec 256
#include "Eigen/Eigenvalues"
using namespace std;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
Eigen::IOFormat CleanFmt(16, 0, ", ", "\n", "[", "]");
Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver;

bool verbose = false;

#define Complex complex<double>
#include "linAlgHelpers.h"
#include "lapack.h"
#include "algoHelpers.h"

int main(int argc, char **argv) {

  cout << std::setprecision(16);
  cout << scientific;  
  //Define the problem
  if (argc < 10 || argc > 10) {
    cout << "Built for matrix size " << Nvec << endl;
    cout << "./iram <nKr> <nEv> <max-restarts> <diag> <tol> <amin> <amax> <spectrum> <hermitian>" << endl;
    cout << "./iram 24 12 20 50 1e-330 1 1 1 1 " << endl;
    exit(0);
  }
  
  int nKr = atoi(argv[1]);
  int nEv = atoi(argv[2]);
  int max_restarts = atoi(argv[3]);
  double diag = atof(argv[4]);
  double tol = atof(argv[5]);
  double a_min = atof(argv[6]);
  double a_max = atof(argv[7]);
  int spectrum = atoi(argv[8]);
  bool hermitian = atoi(argv[9]) == 1 ? true : false;
  
  Complex cZero(0.0,0.0);
  
  if (!(nKr > nEv + 6)) {
    printf("nKr=%d must be greater than nEv+6=%d\n", nKr, nEv + 6);
    exit(0);
  }
  
  //Construct a matrix using Eigen.
  //---------------------------------------------------------------------  
  MatrixXcd ref = MatrixXcd::Random(Nvec, Nvec);

  // Copy to mat
  Complex **mat = (Complex**)malloc(Nvec*sizeof(Complex*));
  if(hermitian) { 
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
  } else {  
    for(int i=0; i<Nvec; i++) {
      mat[i] = (Complex*)malloc(Nvec*sizeof(Complex));
      for(int j=0; j<Nvec; j++) {
	mat[i][j] = ref(i,j);
      }
    }
  }
  
  //Eigensolve the matrix using Eigen, use as a reference.
  //---------------------------------------------------------------------  
  printf("START EIGEN SOLUTION\n");
  double t1 = clock();
  Eigen::ComplexEigenSolver<MatrixXcd> eigensolverRef(ref);
  Eigen::ComplexSchur<MatrixXcd> schurUH;
  cout << eigensolverRef.eigenvalues() << endl;
  double t2e = clock() - t1;
  printf("END EIGEN SOLUTION\n");
  printf("Time to solve problem using Eigen = %e\n", t2e/CLOCKS_PER_SEC);
  //-----------------------------------------------------------------------

  //Construct objects for Lanczos.
  //---------------------------------------------------------------------
  //Eigenvalues and their residuals
  double *residua = (double*)malloc(nKr*sizeof(double));
  Complex *evals  = (Complex*)malloc(nKr*sizeof(Complex));
  Complex *ritz_vals  = (Complex*)malloc(nKr*sizeof(Complex));
  Complex *bounds  = (Complex*)malloc(nKr*sizeof(Complex));
  for(int i=0; i<nKr; i++) {
    residua[i] = 0.0;
    evals[i]   = 0.0;
    ritz_vals[i] = cZero;
    bounds[i] = cZero;
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
  
  t1 = clock();
  double epsilon = DBL_EPSILON;
  double epsilon23 = pow(epsilon,2.0/3.0);
  bool converged = false;
  int restart_iter = 0;
  int num_converged = 0;
  int step = 0;
  int step_start = 0;
  int ops = 0;

  int np = 0;
  int np0 = 0;
  
  //%---------------------------------------------%
  //| Get a possibly random starting vector and   |
  //| force it into the range of the operator OP. |
  //%---------------------------------------------%

  MatrixXcd upperHessEigen = MatrixXcd::Zero(nKr, nKr);
  
  // Populate source
  //for(int i=0; i<Nvec; i++) r[i] = drand48();
  for(int i=0; i<Nvec; i++) r[i] = 1.0;
  
  // loop over restart iterations.
  while(!converged && restart_iter < max_restarts) {

    np = nKr - step_start;
      
    for(step = step_start; step < nKr; step++) arnoldiStep(mat, kSpace, upperHess, r, step);
    
    ops += np;
    step_start = nEv;
    upperHessEigen.Zero(nKr,nKr);
    //Construct the Upper Hessenberg matrix H_k      
    for(int i=0; i<nKr; i++) {
      for(int j=0; j<nKr; j++) {	
	upperHessEigen(i,j) = upperHess[i][j];
      }
    }    
    
    double beta_nKrm1 = dznrm2(Nvec, kSpace[nKr], 1);
    
    // Eigensolve the H_k matrix. The shifts shall be the p
    // largest eigenvalues, as those are the ones we wish to project
    // away from.
    //eigenSolverUH.compute(upperHessEigen);
    schurUH.compute(upperHessEigen);
    for (int i = 0; i < nKr; i++) {
      //cout << "schur: " << eigenSolverUH.eigenvalues()[i] << " " << schurUH.matrixT()(i,i) << endl;
      //cout << "lelem: " << eigenSolverUH.eigenvectors().col(i)[nKr-1] << " " << schurUH.matrixU()(nKr-1,i) << endl;
    }
    
    
    // Initialise Q
    MatrixXcd Q = MatrixXcd::Identity(nKr, nKr);
    for(int i=0; i<nKr; i++) {
      cout << upperHessEigen(i,i) << " ";
      if(i<nKr-1) {
	cout << upperHessEigen(i+1,i);	
      }
      cout << endl;
    }
    
    cout << "Using beta_nKrm1 = " << beta_nKrm1 << endl;    
    // Ritz estimates are updated.
    for (int i = 0; i < nKr; i++) {
      bounds[i] = beta_nKrm1*schurUH.matrixU()(nKr-1,i);
      ritz_vals[i] = schurUH.matrixT()(i,i);
    }

    //%---------------------------------------------------%
    //| Select the wanted Ritz values and their bounds    |
    //| to be used in the convergence test.               |
    //| The wanted part of the spectrum and corresponding |
    //| bounds are in the last NEV loc. of RITZ           |
    //| BOUNDS respectively.                              |
    //%---------------------------------------------------%
    
    // shift selection zngets.f
    //0  'LM' -> sort X into increasing order of magnitude.
    //1  'SM' -> sort X into decreasing order of magnitude.
    //2  'LR' -> sort X with real(X) in increasing algebraic order 
    //3  'SR' -> sort X with real(X) in decreasing algebraic order
    //4  'LI' -> sort X with imag(X) in increasing algebraic order
    //5  'SI' -> sort X with imag(X) in decreasing algebraic order
    
    int shifts = nKr - step_start;
    // Sort the eigenvalues, largest first. Unwanted are at the START of the array.
    // Ritz estimates come along for the ride
    zsortc(spectrum, true, nKr, ritz_vals, bounds);

    // We wish to shift the first nKr - step_start values.
    // Sort these so that the largest Ritz errors are first.
    zsortc(1, true, shifts, bounds, ritz_vals);
    
    for (int i = 0; i < nKr; i++) {
      cout << "residua " << i << " = " << bounds[i] << endl;
    }

    //%------------------------------------------------------------%
    //| Convergence test: currently we use the following criteria. |
    //| The relative accuracy of a Ritz value is considered        |
    //| acceptable if:                                             |
    //|                                                            |
    //| error_bounds(i) .le. tol*max(eps23, magnitude_of_ritz(i)). |
    //|                                                            |
    //%------------------------------------------------------------%
    num_converged = 0;
    int np0 = nKr - step_start;
    for (int i = 0; i < nEv; i++) {
      double condition = std::max(epsilon23, dlapy2(ritz_vals[np0 + i].real(), ritz_vals[np0 + i].imag()));
      if (dlapy2(bounds[np0 + i].real(), bounds[np0 + i].imag()) < tol * condition) {
	printf("**** Converged %d resid=%+.6e condition=%.6e ****\n",
	       i, dlapy2(bounds[np0 + i].real(), bounds[np0 + i].imag()) , tol * condition);
	num_converged++;
      }
    }
    
    if(num_converged > nEv-1) {
      converged = true;
      break;
    }    

    //%---------------------------------------------------------%
    //| Count the number of unwanted Ritz values that have zero |
    //| Ritz estimates. If any Ritz estimates are equal to zero |
    //| then a leading block of H of order equal to at least    |
    //| the number of Ritz values with zero Ritz estimates has  |
    //| split off. None of these Ritz values may be removed by  |
    //| shifting. Decrease NP the number of shifts to apply. If |
    //| no shifts may be applied, then prepare to exit          |
    //%---------------------------------------------------------%
    
    int step_start_old = step_start;
    int shifts_old = shifts;
    // do 30
    for (int i = 0; i < shifts_old; i++) {
      if (bounds[i] == cZero) {
	step_start++;
	shifts--;
      }
    }
    step_start += std::min(num_converged, shifts / 2);
    
    if(step_start == 1 && nKr >= 6) step_start = nKr / 2;
    else if(step_start == 1 && nKr > 3) step_start = 2;   
    shifts = nKr - step_start;
    
    if (step_start_old < step_start) {
      // Sort the eigenvalues, largest first. Unwanted are at the START of the array.
      // Ritz estimates come along for the ride
      zsortc(spectrum, true, nKr, ritz_vals, bounds);
      
      // We wish to shift the first nKr - step_start values.
      // Sort these so that the largest Ritz errors are first.
      zsortc(1, true, shifts, bounds, ritz_vals);      
    }
    

    // znapps.f 
    for(int j=0; j<shifts; j++) {      
      givensQRUpperHess(upperHessEigen, Q, nKr, shifts, j, step_start, ritz_vals[j], restart_iter);
    }
    
    if(upperHessEigen(step_start,step_start-1).real() > 0) {
      cout << "ZNAPPS DBL HIT" << endl;
      rotateVecsComplex(kSpace, Q, 0, step_start+1, nKr);
    } else { 
      rotateVecsComplex(kSpace, Q, 0, step_start, nKr);
    }
  
    cax(Q(nKr-1,step_start-1), r);
    if(upperHessEigen(step_start,step_start-1).real() > 0) {
      caxpy(upperHessEigen(step_start,step_start-1), kSpace[step_start], r);
    }
    
    // Update UpperHess
    for(int i=0; i<nKr; i++) {      
      for(int j=0; j<nKr; j++) {
	upperHess[i][j] = upperHessEigen(i,j);
      }
    }
    
    restart_iter++;
  }
  
  for(int i=0; i<nKr; i++) {
    cout << upperHessEigen(i,i) << " ";
    if(i<nKr-1) {
      cout << upperHessEigen(i+1,i);	
    }
    cout << endl;
  }
  
  
  
  // Post computation report  
  if (!converged) {    
    printf("IRAM failed to compute the requested %d vectors with a %d search space and %d Krylov space in %d OP*x.\n",
	   nEv, nEv, nKr, ops);
  } else {
    printf("IRAM computed the requested %d vectors in %d restart steps and %d OP*x operations in %e secs.\n",
	   nEv, restart_iter, ops, t2e/CLOCKS_PER_SEC);
  }
  
  // Dump all Ritz values and residua
  for (int i = 0; i < nEv; i++) {
    //printf("RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, alpha[i], 0.0, residua[i]);
  }
    
  // Compute eigenvalues
  computeEvals(mat, kSpace, residua, evals, nEv);
  for (int i = 0; i < nEv; i++) {
    printf("EigValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, evals[i].real(), evals[i].imag(),
	   residua[i]);
  }

  //cout << eigensolverRef.eigenvalues() << endl;

  for (int i = 0; i < nEv; i++) {
    int idx = (spectrum+1 % 2 == 1) ? Nvec-1 - i : i;
    printf("EigenComp[%04d]: (%+.16e,%+.16e) diff = (%+.16e,%+.16e)\n", i,
	   eigensolverRef.eigenvalues()[idx].real(), eigensolverRef.eigenvalues()[idx].imag(),
	   ((evals[i] - eigensolverRef.eigenvalues()[idx]).real()/eigensolverRef.eigenvalues()[idx]).real(),
	   ((evals[i] - eigensolverRef.eigenvalues()[idx]).imag()/eigensolverRef.eigenvalues()[idx]).imag()
	   );
  }
  
  free(residua);
  free(bounds);
  free(ritz_vals);
  free(evals);  
  free(r);
}
