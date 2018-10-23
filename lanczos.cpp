//Apply restart with these parameters.


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

#include "/Users/deanhowarth/lanczoz/Eigen/Eigen/Eigenvalues"

using namespace std;

mt19937 rng(1235);
uniform_real_distribution<double> unif(0.0,1.0);
#define Nvec 512

static void mergeAbs(double *sort1, int *idx1, int n1, double *sort2,
		     int *idx2, int n2, bool inverse) {
  int i1=0, i2=0;
  int *ord;
  double *result;
    
  ord    = (int *)    malloc(sizeof(int)   *(n1+n2)); 
  result = (double *) malloc(sizeof(double)*(n1+n2)); 
  
  for(int i=0; i<(n1+n2); i++) {
    if((fabs(sort1[i1]) >= fabs(sort2[i2])) != inverse) { //LOGICAL XOR
      result[i] = sort1[i1];
      ord[i] = idx1[i1];
      i1++;
    } else {
      result[i] = sort2[i2];
      ord[i] = idx2[i2];
      i2++;
    }
    
    if(i1 == n1) {
      for(int j=i+1; j<(n1+n2); j++,i2++) {
	result[j] = sort2[i2];
	ord[j] = idx2[i2];
      }
      i = n1+n2;
    } else if (i2 == n2) {
      for(int j=i+1; j<(n1+n2); j++,i1++) {
	result[j] = sort1[i1];
	ord[j] = idx1[i1];
      }
      i = i1+i2;
    }
  }  
  for(int i=0;i<n1;i++) {
    idx1[i] = ord[i];
    sort1[i] = result[i];
  }
    
  for(int i=0;i<n2;i++) {
    idx2[i] = ord[i+n1];
    sort2[i] = result[i+n1];
  }  
  free (ord);
  free (result);
}

static void sortAbs(double *unsorted, int n, bool inverse, int *idx) {

  if (n <= 1)
    return;
    
  int n1,n2;
    
  n1 = n>>1;
  n2 = n-n1;
    
  double *unsort1 = unsorted;
  double *unsort2 = (double *)((char*)unsorted + n1*sizeof(double));
  int *idx1 = idx;
  int *idx2 = (int *)((char*)idx + n1*sizeof(int));
    
  sortAbs(unsort1, n1, inverse, idx1);
  sortAbs(unsort2, n2, inverse, idx2);
    
  mergeAbs(unsort1, idx1, n1, unsort2, idx2, n2, inverse);
}


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

int main(int argc, char **argv) {
  
  int nkv = atoi(argv[1]);
  int nev = atoi(argv[2]);
  int converged = 0;
  
  double *mod_h_evals_sorted = (double*)malloc(nkv*sizeof(double));
  double *h_evals_resid      = (double*)malloc(nkv*sizeof(double));
  double *h_evals            = (double*)malloc(nkv*sizeof(double));
  int *h_evals_sorted_idx    = (int*)malloc(nkv*sizeof(int));
  
  //Construct and solve a matrix using Eigen, use as a trusted reference.
  using Eigen::MatrixXd;
  MatrixXd ref = MatrixXd::Random(Nvec, Nvec);
  //symmetrise
  
  //Problem matrix
  double **mat = (double**)malloc(Nvec*sizeof(double*));  
  //Allocate space and populate
  for(int i=0; i<Nvec; i++) {
    mat[i] = (double*)malloc(Nvec*sizeof(double));
    ref(i,i) += 40;
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

  // START LANCZOS
  // Lanczos Method for Symmetric Eigenvalue Problems
  // Based on Rudy Arthur's PHD thesis
  // Link as of 06 Oct 2018: https://www.era.lib.ed.ac.uk/bitstream/handle/1842/7825/Arthur2012.pdf
  //-----------------------------------------
      
  //Abbreviated Initial Step
  
  //v_1
  copy(ritzVecs[0], r);
  
  //r = A * v_1
  //apply matrix-vector operation here:
  matVec(mat, r, ritzVecs[0]);
  
  //a_1 = v_1^dag * r
  alpha[0] = dotProd(ritzVecs[0], r);    
  
  //r = r - a_1 * v_1 
  axpy(-alpha[0], ritzVecs[0], r);
  
  //b_1 = ||r||
  beta[0] = sqrt(norm(r));
  
  //Prepare next step.
  //v_2 = r / b_1
  zero(ritzVecs[1]);      
  axpy(1.0/beta[0], r, ritzVecs[1]);    

  //Begin iteration    
  for(int j=1; j<nkv; j++) {
    
    //Compute r = A * v_j - b_{j-i} * v_{j-1}      
    //r = A * v_j
    matVec(mat, r, ritzVecs[j]);
    //r = r - b_{j-1} * v_{j-1}
    axpy(-beta[j-1], ritzVecs[j-1], r);      
    
    //a_j = v_j^dag * r
    alpha[j] = dotProd(ritzVecs[j], r);    
      
    //r = r - a_j * v_j
    axpy(-alpha[j], ritzVecs[j], r);
    
    //b_j = ||r|| 
    beta[j] = sqrt(norm(r));
    
    //Prepare next step.
    //v__{j+1} = r / b_j
    zero(ritzVecs[j+1]);
    axpy(1.0/beta[j], r, ritzVecs[j+1]);
    
    //Orthonormalise      
    double s = 0.0;
    for(int i=0; i<j; i++) {
      s += dotProd(ritzVecs[i], r);
      axpy(-s, ritzVecs[i], r);
    }
  }
  
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
  Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolverTD1(triDiag);
      
  //std::cout << eigenSolverTD.eigenvalues() << std::endl; 
  //Ritz values are in ascending order if matrix is real.
  
  //Perform Rayleigh-Ritz check for convergence    
  // y_i = V_k s_i
  // where T_k s_i = theta_i s_i
  
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
	sum += tmp[l]*eigenSolverTD1.eigenvectors().col(i)[l];
      }
      
      //Update the Ritz vector
      ritzVecs[i][j] = sum;
      sum = 0.0;
    }
  }
  
  for(int i=0; i<nkv; i++){
    printf("TriDiag EigValue[%04d] = %.16e\n",		 
	   i, eigenSolverTD1.eigenvalues()[i]);
  }
  
  for(int i=0; i<nkv; i++) {

    //r = A * v_i
    matVec(mat, r, ritzVecs[i]);

    //lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
    h_evals[i] = dotProd(ritzVecs[i], r)/sqrt(norm(ritzVecs[i]));

    //Convergence check ||A * v_i - lambda_i * v_i||
    axpy(-h_evals[i], ritzVecs[i], r);
    h_evals_resid[i] =  sqrt(norm(r));

    if(h_evals_resid[i] < 1e-6) {
      locked[i] = true;
      converged++;
    }
      
    printf("EigValue[%04d]: ref = %+.8e, comp= %+.8e, comp res = %+.8e, calc res = %+.8e, ratio = %+.8e, ||Evec|| = %+.8e\n",		 
	   i, eigenSolver.eigenvalues()[i], h_evals[i], h_evals_resid[i],
	   beta[nkv-1]*eigenSolverTD1.eigenvectors().col(i)[nkv-1],
	   h_evals_resid[i]/(beta[nkv-1]*eigenSolverTD1.eigenvectors().col(i)[nkv-1]),
	   sqrt(norm(ritzVecs[i])));
    
  }

  /*
  printf("***** Attempting a single Restart *****\n");

  //Begin restarted iteration    
  for (int k = 0; k < nkv - converged && !locked[k]; ++k) {          
    
    //q_0 = r / norm;
    if( k== 0) copy(r, ritzVecs[k]);
    //r = M * q_k
    //apply matrix-vector operation here:
    
    else matVec(mat, r, ritzVecs[k]);

    //r = r - B_{k-1} * q_{k-1}
    axpy(-beta[k-1], ritzVecs[k-1], r);      
    
    //A_k = r^dag * M * r
    alpha[k] = dotProd(ritzVecs[k], r);    
    
    //r = r - A_k * q_0 
    axpy(-alpha[k], ritzVecs[k], r);
    
    beta[k] = sqrt(norm(r));

    printf("beta %d %+e\n", k, beta[k]);
    
    //The final vector is the q_{m+1} starting vector for the
    //Thick/Block/Implicit restart
    zero(ritzVecs[k+1]);
    axpy(1.0/beta[k], r, ritzVecs[k+1]);
    
    //Orthonormalise      
    for(int i=k-1; i<k+1; i++) {
      
      
      double C = dotProd(ritzVecs[i], r); //<i,k>	
      axpy(-C, ritzVecs[i], r); // k-<i,k>i
      
      //check ortho
      C = dotProd(ritzVecs[i], r); //<i,k>
      if(fabs(C) > 1e-14) {
	printf("%d %d ortho fail %+e\n", k, i, C);
	printf("q = %f, r = %f\n", sqrt(norm(ritzVecs[i])), sqrt(norm(r)));
      } else {
	locked[i] = true;
      }
    }      
  }
  

  //Compute the Tridiagonal matrix T_k (k = nkv)
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
      
  //std::cout << eigenSolverTD.eigenvalues() << std::endl; 
  //Ritz values are in ascending order if matrix is real.
  
  //Perform Rayleigh-Ritz check for convergence    
  // y_i = V_k s_i
  // where T_k s_i = theta_i s_i
  
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
	sum += tmp[l]*eigenSolverTD.eigenvectors().col(i)[l];
      }
      
      //Update the Ritz vector
      ritzVecs[i][j] = sum;
      sum = 0.0;
    }
  }
  
  for(int i=0; i<nkv; i++){
    printf("TriDiag EigValue[%04d] = %.16e\n",		 
	   i, eigenSolverTD.eigenvalues()[i]);
  }
  
  for(int i=0; i<nkv; i++) {

    //r = A * v_i
    matVec(mat, r, ritzVecs[i]);

    //lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
    h_evals[i] = dotProd(ritzVecs[i], r)/sqrt(norm(ritzVecs[i]));

    //Convergence check ||A * v_i - lambda_i * v_i||
    axpy(-h_evals[i], ritzVecs[i], r);
    h_evals_resid[i] =  sqrt(norm(r));

    if(h_evals_resid[i] < 1e-6) {
      locked[i] = true;      
    }
      
    printf("EigValue[%04d]: ref = %+.8e, comp= %+.8e, comp res = %+.8e, calc res = %+.8e, ratio = %+.8e, ||Evec|| = %+.8e\n",		 
	   i, eigenSolver.eigenvalues()[i], h_evals[i], h_evals_resid[i],
	   beta[nkv-1]*eigenSolverTD.eigenvectors().col(i)[nkv-1],
	   h_evals_resid[i]/(beta[nkv-1]*eigenSolverTD.eigenvectors().col(i)[nkv-1]),
	   sqrt(norm(ritzVecs[i])));
    
  }

}
  */
}
