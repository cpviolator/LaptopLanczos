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

#include <arpack_interface.h>

#define Nvec 128
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
  if (argc < 9 || argc > 9) {
    cout << "Built for matrix size " << Nvec << endl;
    cout << "./iram <nKr> <nEv> <max-restarts> <diag> <tol> <spectrum> <hermitian> <transfer>" << endl;
    cout << "./iram 24 12 20 50 1e-12 1 1 1 " << endl;
    exit(0);
  }
  
  int nKr = atoi(argv[1]);
  int nEv = atoi(argv[2]);
  int max_restarts = atoi(argv[3]);
  double diag = atof(argv[4]);
  double tol = atof(argv[5]);
  int spectrum = atoi(argv[6]);
  bool hermitian = atoi(argv[7]) == 1 ? true : false;
  bool transfer = atoi(argv[8]) == 1 ? true : false;
  
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

  //Construct parameters and memory allocation
  //------------------------------------------
  
  // all FORTRAN communication uses underscored 
  int ido_;
  int info_;
  int *ipntr_ = (int*)malloc(11*sizeof(int));
  int *iparam_ = (int*)malloc(11*sizeof(int));
  int n_    = Nvec,
    nev_    = nEv,
    nkv_    = nKr,
    ldv_    = Nvec,
    lworkl_ = (3 * nKr * nKr + 5 * nKr) * 2,
    rvec_   = 1;
  int max_iter_ = max_restarts;

  double tol_ = tol;

  //ARPACK workspace
  Complex sigma_ = 0.0;
  Complex *resid_ = (Complex *) malloc(ldv_*sizeof(Complex));
  Complex *w_workd_ = (Complex *) malloc(3*ldv_*sizeof(Complex));
  Complex *w_workl_ = (Complex *) malloc(lworkl_*sizeof(Complex)); 
  Complex *w_workev_= (Complex *) malloc(3*nkv_*sizeof(Complex));
  double *w_rwork_ = (double *) malloc(nkv_*sizeof(double));    
  int *select_ = (int*)malloc(nkv_*sizeof(int));
  
  Complex *evecs_ar = (Complex *) malloc(nkv_*n_*sizeof(Complex));
  Complex *evals_ar = (Complex *) malloc(nkv_   *sizeof(Complex));

  Complex one(1.0,0.0);
  
  for(int n=0; n<nkv_; n++) {
    evals_ar[n] = 0;
    for(int i=0; i<n_; i++) {
      evecs_ar[n*n_ + i] = 0;
      if(n==0) resid_[i] = one;
    }
  }
  
  //Alias pointers
  Complex *evecs_ = nullptr;
  evecs_ = (Complex*)(evecs_ar);    
  Complex *evals_ = nullptr;
  evals_ = (Complex*)(evals_ar);
  
  //Memory checks
  if((iparam_ == nullptr) ||
     (ipntr_ == nullptr) || 
     (resid_ == nullptr) ||  
     (w_workd_ == nullptr) || 
     (w_workl_ == nullptr) ||
     (w_workev_ == nullptr) ||
     (w_rwork_ == nullptr) || 
     (select_ == nullptr) ) {
    printf("eigenSolver: not enough memory for ARPACK workspace.\n");
    exit(0);
  }    

  //Assign values to ARPACK params 
  ido_        = 0;
  info_       = 1;
  iparam_[0]  = 1;
  iparam_[2]  = max_iter_;
  iparam_[3]  = 1;
  iparam_[6]  = 1;
  
  //ARPACK problem type to be solved
  char howmany = 'A';
  char bmat = 'I';
  char spectrum_[3] = {'S','M'};
  int iter_cnt= 0;

  //Start ARPACK routines
  //---------------------------------------------------------------------------------
 
  Complex *psi1;
  Complex *psi2;

  Complex *psi1_cpy = (Complex*)malloc(n_*sizeof(Complex));
  Complex *psi2_cpy = (Complex*)malloc(n_*sizeof(Complex));
  
  for(int i=0; i<n_; i++) {
    psi1_cpy[i] = 1.0;
    psi2_cpy[i] = 1.0;
  }
  
  psi1 = w_workd_;
  psi2 = w_workd_ + n_;
  
  double time = 0.0;;
  do {
    
    t1 = -((double)clock());
    
    //Interface to arpack routines
    //----------------------------

    ARPACK(znaupd)(&ido_, &bmat, &n_, spectrum_, &nev_, &tol_, resid_, &nkv_, evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_, w_rwork_, &info_, 1, 2);
    
    if (info_ != 0) {
      printf("\nError in dsaupd info = %d. Exiting...\n",info_);
      arpackErrorHelpSAUPD(iparam_);
      exit(0);
    }
    
    if (ido_ == 99 || info_ == 1)
      break;
    
    if (ido_ == -1 || ido_ == 1) {

      //Copy from Arpack workspace
      for(int i=0; i<n_; i++) {
	psi1_cpy[i] = *(psi1 + i);
      }

      matVec(mat, psi2_cpy, psi1_cpy);
      //Copy to Arpack workspace
      for(int i=0; i<n_; i++) {
	*(psi2 + i) = psi2_cpy[i];
      }
    }



    //  for(int q = 0; q<nKr*nKr; q++) {
    //   printf("(%+.3e,%+.3e) ", w_workl_[nKr*nKr+2*nKr + q].real(), w_workl_[nKr*nKr+2*nKr+1 + q].imag());
    //   if((q+1)%nKr == 0) cout << endl;
    // }
    // cout << endl;
    // cout << endl;
    
    t1 += clock();
    time += t1;
    if((iter_cnt+1)%1000 == 0) printf("Arpack Iteration: %d (%e secs)\n", iter_cnt, time/(CLOCKS_PER_SEC));
    iter_cnt++;
    
  } while (99 != ido_ && iter_cnt < max_iter_);
  
  //Subspace calulated sucessfully. Compute nEv eigenvectors and values
  printf("ARPACK Finished in %e secs: iter=%04d  info=%d  ido=%d\n", time/(CLOCKS_PER_SEC), iter_cnt, info_, ido_);
  
  
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
  double beta_nKrm1 = 0.0;
  int np = 0;
  
  //%---------------------------------------------%
  //| Get a possibly random starting vector and   |
  //| force it into the range of the operator OP. |
  //%---------------------------------------------%

  MatrixXcd upperHess = MatrixXcd::Zero(nKr, nKr);
  
  // Populate source
  //for(int i=0; i<Nvec; i++) r[i] = drand48();
  for(int i=0; i<Nvec; i++) r[i] = 1.0;
  
  // loop over restart iterations.
  while(!converged && restart_iter < max_restarts) {

    np = nKr - step_start;
    
    for(step = step_start; step < nKr; step++) arnoldiStep(mat, kSpace, upperHess, r, step);
    
    cout << "RESTART" << endl;
    
    ops += np;
    step_start = nEv;    
    beta_nKrm1 = dznrm2(Nvec, kSpace[nKr], 1);
    
    // Schur Decompose the upper Hessenberg matrix.
    schurUH.compute(upperHess);
    
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
      if (dlapy2(bounds[np0 + i].real(), bounds[np0 + i].imag()) <= tol * condition) {
	printf("**** Converged %d resid=%+.6e condition=%.6e ****\n",
	       i, dlapy2(bounds[np0 + i].real(), bounds[np0 + i].imag()) , tol * condition);
	num_converged++;
      }
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

    if(num_converged > nEv-1 || shifts == 0) {
      converged = true;
      break;
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
    // Initialise Q
    MatrixXcd Q = MatrixXcd::Identity(nKr, nKr);    
    for(int j=0; j<shifts; j++) {      
      givensQRUpperHess(upperHess, Q, nKr, shifts, j, step_start, ritz_vals[j], restart_iter);
    }
    
    if(upperHess(step_start,step_start-1).real() > 0) {
      rotateVecsComplex(kSpace, Q, 0, step_start+1, nKr);
    } else { 
      rotateVecsComplex(kSpace, Q, 0, step_start, nKr);
    }
  
    cax(Q(nKr-1,step_start-1), r);
    if(upperHess(step_start,step_start-1).real() > 0) {
      caxpy(upperHess(step_start,step_start-1), kSpace[step_start], r);
    }
    
    restart_iter++;
  }

  if(!transfer) {
    
    /*
    //zneupd.f
    //%---------------------------------------------------%
    //| Use the temporary bounds array to store indices   |
    //| These will be used to mark the select array later |
    //%---------------------------------------------------%
    
    bool reord = false;
    
    std::vector<bool> select(nKr);
    std::vector<Complex> indices(nKr);
    for(int i=0; i<nKr; i++) {
      indices[i] = i;
      select[i] = false;
    }
    
    np = nKr - nEv;
    // Sort the eigenvalues, largest first. Unwanted are at the START of the array.
    // Ritz estimates come along for the ride
    zsortc(spectrum, true, nKr, ritz_vals, bounds);
    
    // We wish to shift the first nKr - step_start values.
    // Sort these so that the largest Ritz errors are first.
    //zsortc(1, true, np, bounds, ritz_vals);      
    
    //%-----------------------------------------------------%
  //| Record indices of the converged wanted Ritz values  |
  //| Mark the select array for possible reordering       |
  //%-----------------------------------------------------%

  int numcnv = 0;
  for(int i=0; i<nKr; i++) {

    double rtemp = std::max(epsilon23, dlapy2(ritz_vals[nKr-1-i].real(), ritz_vals[nKr-1-i].imag()));
    //cout << rtemp << endl;
    int j = indices[nKr-1 - i].real();
    //cout << " " << dlapy2(bounds[j].real(), bounds[j].imag()) << " " << tol*rtemp << endl;
    if(numcnv < num_converged &&
       dlapy2(bounds[j].real(), bounds[j].imag()) <= tol*rtemp) {
      
      select[j] = true;
      numcnv++;
      if(j > nEv) reord = true;
    }
  }
  
  if(reord) {
    cout << "Hit Reord" << endl;
    for(int i=0; i<nKr; i++) {
      cout << select[i] << " " << (int)(indices[i].real()) <<  endl;      
    }
  }

    
  //%-----------------------------------------------------------%
  //| Check the count (numcnv) of converged Ritz values with    |
  //| the number (nconv) reported by dnaupd.  If these two      |
  //| are different then there has probably been an error       |
  //| caused by incorrect passing of the dnaupd data.           |
  //%-----------------------------------------------------------%

  if(numcnv != num_converged) {
    cout << "Error in zneupd. numcnv = " << numcnv << " num_converged = " << num_converged << endl;
    exit(0);
  }

  //%-------------------------------------------------------%
  //| Call LAPACK routine zlahqr to compute the Schur form |
  //| of the upper Hessenberg matrix returned by ZNAUPD.   |
  //| Make a copy of the upper Hessenberg matrix.           |
  //| Initialize the Schur vector matrix Q to the identity. |
  //%-------------------------------------------------------%
  
  MatrixXcd upperHessCopy = MatrixXcd::Zero(nKr, nKr);
  upperHessCopy = upperHess;
  
  // Schur Decompose the upper Hessenberg matrix.
  schurUH.compute(upperHess);
  
  // Ritz estimates are updated.
  for (int i = 0; i < nKr; i++) {
    bounds[i] = beta_nKrm1*schurUH.matrixU()(nKr-1,i);
    ritz_vals[i] = schurUH.matrixT()(i,i);
  }

  //%-----------------------------------------------%
  //| Reorder the computed upper triangular matrix. |
  //%-----------------------------------------------%
  
  //call ztrsen('None'       , 'V'          , select      ,
  //&           ncv          , workl(iuptri), ldh         ,
  //&           workl(invsub), ldq          , workl(iheig),
  //&           nconv        , conds        , sep         , 
  //&           workev       , ncv          , ierr)
  //SUBROUTINE ZTRSEN( JOB, COMPQ, SELECT, N, T, LDT, Q, LDQ, W, M, S,
  //$                  SEP, WORK,  LWORK, INFO )
  

   
  MatrixXcd schurU = schurUH.matrixU();//.block(0,0,nKr,num_converged);
  MatrixXcd schurTest = schurUH.matrixU().adjoint();
  Complex tau[num_converged];
  zgeqr2(nKr, num_converged, schurU, tau);
  
  Complex cOne(1.0,0.0);
  //rotateVecsComplex(kSpace, schurU, 0, num_converged, nKr);
  for(int r=0; r<num_converged; r++) {
    //if(schurU(r,r).real() < 0.0) cax(-cOne, kSpace[r]);
    }
    */
  } else {

    //Transfer eigenvectors and eigenvalues
    for(int n=0; n<nkv_; n++) {
      evals_ar[n] = evals[n];
      for(int i=0; i<n_; i++) {
	evecs_ar[n*n_ + i] = kSpace[n][i];
	if(n==0) resid_[i] = one;
      }
    }

    //Transfer upperHess, ritz, and bounds.
    int count = 0;
    for(int j = 0; j<nKr; j++) {
      for(int i = 0; i<nKr; i++) {
	w_workl_[ipntr_[4]-1 + count] = upperHess(i,j);
	w_workl_[ipntr_[6]-1 + count] = upperHess(i,j);
	count ++;
      }
    }
    count = 0;
    for(int i = 0; i<nKr; i++) {
      w_workl_[ipntr_[7]-1 + count] = bounds[i];
      w_workl_[ipntr_[5]-1 + count] = ritz_vals[i];
      count ++;
    }
    
    printf("ARPACK Computing Eigenvlaues\n");
    ARPACK(zneupd)(&rvec_, &howmany, select_, evals_, evecs_, &n_, &sigma_, w_workev_, &bmat, &n_, spectrum_, &nev_, &tol_, resid_,
		   &nkv_, evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_, w_rwork_, &info_, 1, 1, 2);
    
    if (info_ == -15) {
      printf("\nError in dseupd info = %d. You likely need to\n"
	     "increase the maximum ARPACK iterations. Exiting...\n", info_);
      arpackErrorHelpSEUPD(iparam_);
      exit(0);
    } else if (info_ != 0) {
      printf("\nError in dseupd info = %d. Exiting...\n", info_);
      arpackErrorHelpSEUPD(iparam_);
    }
    
    // Print additional convergence information.
    if(info_ == 1){
      printf("Maximum number of iterations reached.\n");
    }
    else{
      if(info_ == 3){
	printf("Error: No shifts could be applied during implicit\n");
	printf("Error: Arnoldi update, try increasing NkV.\n");
      }
    }
    
  
    //Print Evalues  
    for(int i=0; i<nev_; i++){    
      printf("ARPACKEigVal[%04d]: (%+.16e, %+.16e)\n", i,
	     evals_[i].real(), evals[i].imag());
      
    }
    
    for (int i = 0; i < nEv; i++) {
      int idx = (spectrum+1 % 2 == 1) ? Nvec-1 - i : i;
      printf("ARPACKEigenComp[%04d]: (%+.16e,%+.16e) diff = (%+.16e,%+.16e)\n", i,
	     eigensolverRef.eigenvalues()[idx].real(), eigensolverRef.eigenvalues()[idx].imag(),
	     ((evals_[i] - eigensolverRef.eigenvalues()[idx]).real()/eigensolverRef.eigenvalues()[idx]).real(),
	     ((evals_[i] - eigensolverRef.eigenvalues()[idx]).imag()/eigensolverRef.eigenvalues()[idx]).imag()
	     );
    }
    
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
