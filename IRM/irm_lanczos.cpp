#include <vector>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <vector>
#include <cstring>
#include <random>
#include <unistd.h>
#include <omp.h>

#define Nvec 64
#include "Eigen/Eigenvalues"
#include "linAlgHelpers.h"
#include "algoHelpers.h"
#include "utils.h"

using namespace std;
using Eigen::MatrixXd;

int main(int argc, char **argv) {

  //Define the problem
  int nkv = atoi(argv[1]);
  int nev = atoi(argv[2]);
  int restartIterMax = atoi(argv[3]);
  double diag = atof(argv[4]);
  double tol = atof(argv[5]);
  int threads = atoi(argv[6]);
  omp_set_num_threads(threads);
  Eigen::setNbThreads(threads);
  
  
  //Construct a matrix using Eigen.
  //---------------------------------------------------------------------  
  using Eigen::MatrixXd;
  MatrixXd ref = MatrixXd::Random(Nvec, Nvec);
  
  //Copy ref matrix for Lanczos routine
  double **mat = (double**)malloc(Nvec*sizeof(double*));  

  //Allocate space, populate, and symmetrise
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
  
  //Eigensolve the matrix using Eigen, use as a reference.
  //---------------------------------------------------------------------  
  printf("START EIGEN SOLUTION\n");
  double t1 = clock();  
  Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolver(ref);
  double t2e = clock() - t1;
  printf("END EIGEN SOLUTION\n");
  printf("Time to solve problem using Eigen = %e\n", t2e/CLOCKS_PER_SEC);

  //-----------------------------------------------------------------------





  //Construct objects for Lanczos.
  //---------------------------------------------------------------------

  //Eigenvalues and their residuals
  double *evals_resid      = (double*)malloc(nkv*sizeof(double));
  double *evals            = (double*)malloc(nkv*sizeof(double));
  double *mod_evals_sorted = (double*)malloc(nkv*sizeof(double));
  int *evals_sorted_idx    = (int*)malloc(nkv*sizeof(int));
  for(int i=0; i<nkv; i++) {
    evals_resid[i]      = 0.0;
    evals[i]            = 0.0;
    mod_evals_sorted[i] = 0.0;
    evals_sorted_idx[i] = 0;
  }

  //Ritz vectors and Krylov Space. The eigenvectors will be stored here.
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

  //Residual vector. Also used as a temp vector
  double *r = (double*)malloc(Nvec*sizeof(double));

  // Implictly Restarted Lanczos Method for Symmetric Eigenvalue Problems
  // Algorithm 4.7
  // http://www.netlib.org/utk/people/JackDongarra/etemplates/node118.html
  // Step n in the algorithm are denoted by (n) in the code.
  //----------------------------------------------------------------------

  printf("START LANCZOS SOLUTION\n");

  t1 = clock();
  bool converged = false;
  bool update = false;
  int numConverged = 0;
  int restartIter = 0;
    
  // (1) Populate source with randoms.
  printf("Using random guess\n");
  for(int i=0; i<Nvec; i++) r[i] = drand48();
  //Normalise initial source
  double nor = norm(r);
  ax(1.0/sqrt(nor), r);
  //v_1
  copy(krylovSpace[0], r);
  
  // (2) Initial k-step factorisation.
  for(int j=0; j<nev; j++) lanczosStep(mat, krylovSpace, beta, alpha, r, j);

  // (3) loop over restart iterations.
  while(!converged && restartIter < restartIterMax) {
    
    // (2) p = m-k steps to get to the m-step factorisation
    for(int j=nev; j<nkv; j++) lanczosStep(mat, krylovSpace, beta, alpha, r, j);
    
    copy(r, krylovSpace[nkv]);
    ax(beta[nkv-1], r);
        
    //Construct the Tridiagonal matrix T_k (k = nkv)
    MatrixXd triDiag = MatrixXd::Zero(nkv, nkv);    
    for(int i=0; i<nkv; i++) {
      triDiag(i,i) = alpha[i];      
      if(i<nkv-1) {	
	triDiag(i+1,i) = beta[i];	
	triDiag(i,i+1) = beta[i];
      }
    }

    // (4) Eigensolve the T_k matrix. The shifts shall be the the p
    // largest eigenvalues, as those are the ones we wish to project
    // away from.
    Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolverTD(triDiag);
    //cout << eigenSolverTD.eigenvalues() << endl;
    
    // (5) Initialise Q
    MatrixXd Q  = MatrixXd::Identity(nkv, nkv);
    MatrixXd Qj = MatrixXd::Zero(nkv, nkv);;
    MatrixXd TmMuI;

    // (6) QR rotate the tridiag
    for(int j=nev; j<nkv; j++) {
      //printf("ROTATION %d: shifting by ev[%d]=%e\n", j, j, eigenSolverTD.eigenvalues()[j]);
      
      for(int i=0; i<nkv; i++) {
        //printf("Before rot %d: alpha[%d]=%e beta[%d]=%e\n", i, j, triDiag(j,j), j, (j < nkv-1 ? triDiag(j,j+1) : beta[nkv-1]) );
      }
      
      //Apply the shift \mu_j
      TmMuI = triDiag;
      MatrixXd Id = MatrixXd::Identity(nkv, nkv);;
      TmMuI -= eigenSolverTD.eigenvalues()[j]*Id;
      //cout << TmMuI << endl << endl;
      
      // (7) QR decomposition of Tm - \mu_i I
      Eigen::HouseholderQR<MatrixXd> QR(TmMuI);

      // (8) Retain Qj matrices as a product.
      Qj = QR.householderQ();
      Q = Q * Qj;

      // (8) Update the Tridiag
      triDiag = Qj.adjoint() * triDiag * Qj;
      //cout << triDiag << endl << endl;
      
      for(int i=0; i<nkv; i++) {
        //printf("After rot %d: alpha[%d]=%e beta[%d]=%e\n", j, i, triDiag(i,i), i, (i < nkv-1 ? triDiag(i,i+1) : beta[nkv-1]) );
      }
    }

    //Perform in-place basis rotation of the Krylov space
    //to get the Ritz vectors.
    //           y_i = V_k s_i
    // where T_k s_i = theta_i s_i
    computeRitz(krylovSpace, Q, nev, nkv);
    
    // (10) update the residual
    //      r_{nev} = r_{nkv} * \sigma_{nev}  |  \sigma_{nev} = Q(nkv,nev)
    ax(Q(nkv-1,nev-1), r);
    //      r_{nev} += v_{nev+1} * beta_k  |  beta_k = Tm(nev+1,nev)
    axpy(triDiag(nev,nev-1), krylovSpace[nev], r);
    
    double beta_k = sqrt(norm(r));
    zero(krylovSpace[nev]);
    axpy(1.0/beta_k, r, krylovSpace[nev]);
    beta[nev-1] = beta_k;

    //Update the tridiag matrix
    triDiag(nev-1,nev) = beta_k;
    triDiag(nev,nev-1) = beta_k;

    for(int i=0; i<nkv; i++) {
      //printf("alpha[%d]=%e beta[%d]=%e\n", i, triDiag(i,i), i, (i < nkv-1 ? triDiag(i,i+1) : beta[nkv-1]) );
    }
    
    MatrixXd triDiagNew = MatrixXd::Identity(nev, nev);    
    //Construct the new Tridiagonal matrix T_k (k = nkv)    
    for(int i=0; i<nev; i++) {
      triDiagNew(i,i) = triDiag(i,i);
      if(i<nev-1) {	
	triDiagNew(i+1,i) = triDiag(i+1,i);
	triDiagNew(i,i+1) = triDiag(i,i+1);
      }
    }

    Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolverTDnew(triDiagNew);
    //Copy Krylov space to compute Ritz vectors 
    for(int i=0; i<nkv; i++) copy(ritzVecs[i], krylovSpace[i]);
    computeRitz(ritzVecs, eigenSolverTDnew.eigenvectors(), nev, nev);

    //printf("Convergence Check at restart iter %04d\n", restartIter+1);
    
    //Check for convergence
    numConverged = 0;
    update = false;
    for(int i=0; i<nev; i++) {
      
      //r = A * v_i
      matVec(mat, r, ritzVecs[i]);
      
      //lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
      evals[i] = dotProd(ritzVecs[i], r)/sqrt(norm(ritzVecs[i]));
      mod_evals_sorted[i] = fabs(evals[i]);
      //Convergence check ||A * v_i - lambda_i * v_i||
      axpy(-evals[i], ritzVecs[i], r);
      evals_resid[i] = sqrt(norm(r));
      evals_sorted_idx[i] = i;
    }

    /*
    //Sort by modulus 
    sortAbs(mod_evals_sorted, nkv, true, evals_sorted_idx);
    for(int i=0; i<nkv; i++) printf("%d: %e %e %d\n", i,
				    evals[evals_sorted_idx[i]],
				    evals_resid[evals_sorted_idx[i]],		    
				    evals_sorted_idx[i]);
    */
    for(int i=0; i<nev; i++) {
      if(evals_resid[i] < tol) {
	locked[i] = true;
	mod_evals_sorted[numConverged] = fabs(evals[i]);
	evals_sorted_idx[numConverged] = i;
	numConverged++;
	printf("%04d locking converged eigenvalue = %.8e resid %+.8e idx %d\n",
	       i, fabs(evals[i]), evals_resid[i], evals_sorted_idx[i]);
      }
    }
    printf("%04d converged eigenvalues.\n", numConverged);
    
    
    //Place converged eigenpairs in first nev slots of Krylov space array,
    //check for stopping condition.
    bool test = true;
    for(int i=0; i<nev; i++) {
      if(locked[i] == true) {
	//copy(krylovSpace[i], ritzVecs[evals_sorted_idx[i]]);
	
	//alpha[i]  = alpha[evals_sorted_idx[i]];
	//beta[i]   = beta[evals_sorted_idx[i]];

	//locked[i] = true;
	//locked[evals_sorted_idx[i]] = false;
      }
      else {
	test = false;
      }
    }
    
    for(int i=0; i<nev; i++) {
      alpha[i] = triDiagNew(i,i);
      if(i < nev-1) beta[i] = triDiagNew(i,i+1);
    }

    
    //Ritz values are in ascending order if matrix is real.      
    for(int i=0; i<nev; i++) {
      std::cout << i
		<< " " << eigenSolverTD.eigenvalues()[i]
		<< " " << eigenSolverTDnew.eigenvalues()[i]
		<< " " << eigenSolver.eigenvalues()[i] << std::endl;
    }
    

    for(int i=0; i<nev; i++) {
      printf("EigValue[%04d]: EIGEN=%+.16e, LANCZOS=%+.16e, reldiff=%+.3e, "
	     "compres=%+.3e, calcres=%+.3e, ||Evec||=%+.3e\n",
	     i, eigenSolver.eigenvalues()[i], evals[i],
	     1.0-eigenSolver.eigenvalues()[i]/evals[i],
	     evals_resid[i],
	     beta[nkv-1]*eigenSolverTD.eigenvectors().col(i)[nkv-1],
	     sqrt(norm(ritzVecs[i])));
    }
    if(test == true) {
      converged = true;
      break;
    }
    
    restartIter++;
    
  }
  
  if(!converged) {
    printf("Lanczos failed to compute the requested %d vectors in %d steps. "
	   "Please either increase nkv or decrease nev\n", nev, nkv);
  }
  
  double t2l = clock() - t1;
  printf("END LANCZOS SOLUTION\n");
  printf("Time to solve (%d,%d) problem using Eigen = %e\n", Nvec, Nvec, t2e/CLOCKS_PER_SEC);
  printf("Time to solve (%d,%d) problem with %d nkv and %d nev using %d Implicit Lanczos Restarts = %e\n",
	 Nvec, Nvec, nkv, nev, restartIter+1, t2l/(CLOCKS_PER_SEC*threads));
}
