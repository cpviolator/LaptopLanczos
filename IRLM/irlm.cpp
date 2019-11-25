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

#define Nvec 100
#include "Eigen/Eigenvalues"
using namespace std;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
Eigen::IOFormat CleanFmt(16, 0, ", ", "\n", "[", "]");
Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver;

bool verbose = false;

#define Complex complex<double>
#include "linAlgHelpers.h"
#include "algoHelpers.h"

int main(int argc, char **argv) {

  cout << std::setprecision(16);
  cout << scientific;  
  //Define the problem
  if (argc < 10 || argc > 10) {
    cout << "Built for matrix size " << Nvec << endl;
    cout << "./irlm <nKr> <nEv> <max-restarts> <diag> <tol> <amin> <amax> <QR: 0=triDiag, 1=upperHess> <eig_type: 0=lanczos, 1=arnoldi>" << endl;
    cout << "./irlm 40 20 1000 100 1e-330 1 1 0 0 " << endl;
    exit(0);
  }
  
  int nKr = atoi(argv[1]);
  int nEv = atoi(argv[2]);
  int max_restarts = atoi(argv[3]);
  double diag = atof(argv[4]);
  double tol = atof(argv[5]);
  double a_min = atof(argv[6]);
  double a_max = atof(argv[7]);
  int tridiag = atoi(argv[8]);
  int eig_type = atoi(argv[9]);
  
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
  Eigen::SelfAdjointEigenSolver<MatrixXcd> eigenSolverUH;
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
  for(int i=0; i<nKr; i++) {
    residua[i] = 0.0;
    evals[i]   = 0.0;
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
  
  //%---------------------------------------------%
  //| Get a possibly random starting vector and   |
  //| force it into the range of the operator OP. |
  //%---------------------------------------------%
  
  // Populate source
  //for(int i=0; i<Nvec; i++) r[i] = drand48();
  for(int i=0; i<Nvec; i++) r[i] = 1.0;
  
  // loop over restart iterations.
  while(!converged && restart_iter < max_restarts) {

    np = nKr - step_start;
      
    for(step = step_start; step < nKr; step++) {
      if(eig_type == 0) lanczosStep(mat, kSpace, beta, alpha, r, -1, step, a_min, a_max);
      else arnoldiStep(mat, kSpace, upperHess, r, step);
    }
    
    ops += np;
    //cout << "Start extention at " << step_start << endl;
    step_start = nEv;

    //Construct the Upper Hessenberg matrix H_k      
    MatrixXcd upperHessEigen = MatrixXcd::Zero(nKr, nKr);
    if(eig_type == 0) {
      for(int i=0; i<nKr; i++) {
	upperHessEigen(i,i).real(alpha[i]);      
	if(i<nKr-1) {
	  upperHessEigen(i+1,i).real(beta[i]);	
	  upperHessEigen(i,i+1).real(beta[i]);
	}
      }
    } else {
      for(int i=0; i<nKr; i++) {
	for(int j=0; j<nKr; j++) {	
	  upperHessEigen(i,j) = upperHess[i][j];
	}
      }
    }
    
    //cout << upperHessEigen << endl;
    
    double beta_nKrm1 = dznrm2(Nvec, r, 1);
    //if(eig_type == 0) beta_nKrm1 = beta[nKr-1];
    //else beta_nKrm1 = upperHess[nKr][nKr-1].real();
    
    // Eigensolve the H_k matrix. The shifts shall be the p
    // largest eigenvalues, as those are the ones we wish to project
    // away from.
    eigenSolverUH.compute(upperHessEigen);
    //Complex ritz_vals[nKr];
    // Initialise Q
    MatrixXcd Q = MatrixXcd::Identity(nKr, nKr);
    for(int i=0; i<nKr; i++) {
      cout << upperHessEigen(i,i) << " ";
      if(i<nKr-1) {
	cout << upperHessEigen(i+1,i);	
      }
      cout << endl;
    }
    
    //zneigh(beta_nKrm1, nKr, upperHessEigen, nKr, ritz_vals, residua, Q, nKr); 

    cout << "Using beta_nKrm1 = " << beta_nKrm1 << endl;    
    // Ritz estimates are updated.
    for (int i = 0; i < nKr; i++) {
      residua[i] = beta_nKrm1*abs(eigenSolverUH.eigenvectors().col(i)[nKr-1]);
      //cout << "bounds " << i << " = " << beta_nKrm1*eigenSolverUH.eigenvectors().col(i)[nKr-1] << endl;
      //cout << "residua " << i << " = " << residua[i] << endl;
      //cout << "Shift " << i << " = " << eigenSolverUH.eigenvalues()[i] << endl;
      //cout << "Shift " << i << " = " << ritz_vals[i] << endl;
    }

    //%---------------------------------------------------%
    //| Select the wanted Ritz values and their bounds    |
    //| to be used in the convergence test.               |
    //| The wanted part of the spectrum and corresponding |
    //| bounds are in the last NEV loc. of RITZ           |
    //| BOUNDS respectively.                              |
    //%---------------------------------------------------%
    
    // shift selection zngets.f
    // Customize me!
    int shifts = nKr - step_start;
    std::vector<std::pair<double,double>> ritz;
    ritz.reserve(nKr);
    for (int i = 0; i < nKr; i++) {
      ritz.push_back(std::make_pair(eigenSolverUH.eigenvalues()[i], residua[i]));
    }
    // Sort the eigenvalues, largest first. Unwanted are at the START of the array.
    // Ritz estimates come along for the ride
    std::sort(ritz.begin(), ritz.end());
    std::reverse(ritz.begin(), ritz.end());    
    // We wish to shift the first nKr - step_start values.
    // Sort these so that the largest Ritz errors are first.
    std::sort(ritz.begin(), ritz.begin() + shifts,
	      [](const std::pair<double,double> &left,
		 const std::pair<double,double> &right) {
		return left.second < right.second;
	      });    
    std::reverse(ritz.begin(), ritz.begin()+shifts);

    for (int i = 0; i < nKr; i++) {
      //cout << "bounds " << i << " = " << beta_nKrm1*eigenSolverUH.eigenvectors().col(i)[nKr-1] << endl;
      cout << "residua " << i << " = " << ritz[i].second << endl;
      //cout << "Shift " << i << " = " << eigenSolverUH.eigenvalues()[i] << endl;
      //cout << "Shift " << i << " = " << ritz_vals[i] << endl;
    }

    
    // Convergence check
    num_converged = 0;
    for (int i = 0; i < nEv; i++) {
      int np0 = nKr - nEv;
      double condition = std::max(epsilon23, abs(ritz[np0 + i].first));
      if (residua[i] < tol * condition) {
	printf("**** Converged %d resid=%+.6e condition=%.6e ****\n",
	       i, ritz[np0 + i].second, tol * condition);
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
      if (ritz[i].second == 0) {
	step_start++;
	shifts--;
      }
    }
    step_start += std::min(num_converged, shifts / 2);
    
    if(step_start == 1 && nKr >= 6) step_start = nKr / 2;
    else if(step_start == 1 && nKr > 3) step_start = 2;   
    //if(step_start > nKr - 1) step_start = nKr - 1;
    shifts = nKr - step_start;
    
    if (step_start_old < step_start) {
      // Sort the eigenvalues, largest first. Unwanted are at the START of the array.
      // Ritz estimates come along for the ride
      std::sort(ritz.begin(), ritz.end());
      std::reverse(ritz.begin(), ritz.end());    
      // We wish to shift the first nKr - step_start values. Sort these so that the
      // largest Ritz errors are first.
      std::sort(ritz.begin(), ritz.begin() + shifts,
		[](const std::pair<double,double> &left,
		   const std::pair<double,double> &right) {
		  return left.second < right.second;
		});    
      std::reverse(ritz.begin(), ritz.begin()+shifts);      
    }
    

    // znapps.f 
    for(int j=0; j<shifts; j++) {      
      //cout << "Shifting eigenvalue " << ritz.[j]first << endl;
      if(tridiag == 0) {
	givensQRtriDiag(upperHessEigen, Q, nKr, ritz[j].first);
      }
      else {
	Complex cShift(ritz[j].first,0.0);
	givensQRUpperHess(upperHessEigen, Q, nKr, shifts, j, step_start, cShift);
      }
    }

    rotateVecsComplex(kSpace, Q, 0, step_start+1, nKr);
    
    cax(Q(nKr-1,step_start-1)*beta_nKrm1, kSpace[nKr]);
    if(upperHessEigen(step_start,step_start-1).real() > 0) {
      caxpy(upperHessEigen(step_start,step_start-1), kSpace[step_start], kSpace[nKr]);
    }
    
    // Update UpperHess
    if(eig_type == 0) {
      for(int i=0; i<step_start; i++) {
	alpha[i] = upperHessEigen(i,i).real();
	if(i<step_start-1) {
	  beta[i] = upperHessEigen(i+1,i).real();
	}
      }
    } else {
      for(int i=0; i<nKr; i++) {      
	for(int j=0; j<nKr; j++) {
	  upperHess[i][j] = upperHessEigen(i,j);
	}
      }
    }
    
    copy(kSpace[step_start], kSpace[nKr]);
    if(eig_type == 0) beta[step_start-1] = normalise(kSpace[step_start]);
    else upperHess[step_start][step_start-1].real(normalise(kSpace[step_start]));
    
    //printf("%04d converged eigenvalues at restart iter %04d\n", num_converged, restart_iter);    
    restart_iter++;
  }
  
  // Post computation report  
  if (!converged) {    
    printf("IRLM failed to compute the requested %d vectors with a %d search space and %d Krylov space in %d OP*x.\n",
	   nEv, nEv, nKr, ops);
  } else {
    printf("IRLM computed the requested %d vectors in %d restart steps and %d OP*x operations in %e secs.\n",
	   nEv, restart_iter, ops, t2e/CLOCKS_PER_SEC);
  }
  
  // Dump all Ritz values and residua
  for (int i = 0; i < nEv; i++) {
    printf("RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, alpha[i], 0.0, residua[i]);
  }
    
  // Compute eigenvalues
  computeEvals(mat, kSpace, residua, evals, nEv);
  for (int i = 0; i < nEv; i++) {
    printf("EigValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, evals[i].real(), evals[i].imag(),
	   residua[i]);
  }

  //cout << eigensolverRef.eigenvalues() << endl;

  for (int i = 0; i < nEv; i++) {
    printf("EigenComp[%04d]: (%+.16e - %+.16e)/%+.16e = %+.16e\n", i, evals[i].real(), eigensolverRef.eigenvalues()[i], eigensolverRef.eigenvalues()[i], (evals[i].real() - eigensolverRef.eigenvalues()[i])/eigensolverRef.eigenvalues()[i]);
  }
  
  free(evals);  
  free(r);
}
