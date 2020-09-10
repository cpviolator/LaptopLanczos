#ifndef ALGOHELPERS_H
#define ALGOHELPERS_H

#include "linAlgHelpers.h"

std::vector<double> ritz_mat;

//Orthogonalise r against the j vectors in vectorSpace
void measOrthoDev(std::vector<Complex> &s, Complex *r, std::vector<Complex*> vectorSpace, int j) {  
  for(int i=0; i<j+1; i++) s[i] = cDotProd(vectorSpace[i], r);
}

void matVec(Complex **mat, Complex *out, Complex *in) {  
  Complex temp[Nvec];
  zero(temp);
  //Loop over rows of matrix
  for(int i=0; i<Nvec; i++) temp[i] = dotProd(&mat[i][0], in);    
  copy(out, temp);  
}

//Orthogonalise r against the j vectors in vectorSpace, populate with res with residua
void orthogonalise(std::vector<Complex> &s, Complex *r, std::vector<Complex*> vectorSpace, int j) {
  bool orth = false;
  double tol = j*1e-8;
  double err = 0.0;
  int count = 0;
  while (!orth && count < 1) {
    err = 0.0;
    for(int i=0; i<j+1; i++) {
      s[i] = cDotProd(vectorSpace[i], r);
      caxpy(-s[i], vectorSpace[i], r);
      err += abs(s[i]);
    }
    if(err < tol) {
      //cout << "Orthogonality at " << count << endl;
      orth = true;
    }
    count++;
  }
}

void iterRefineReal(std::vector<Complex*> &kSpace, Complex *r, double *alpha, double *beta, int j) {

  std::vector<Complex> s(j+1);
  measOrthoDev(s, r, kSpace, j);
  double err = 0.0;
  for(int i=0; i<j+1; i++) err = std::max(err, abs(s[i].real()));
  double cond = (DBL_EPSILON)*beta[j];
  
  int count = 0;
  while (count < 1 && err > cond ) {
    
    // r = r - s_{i} * v_{i}
    orthogonalise(s, r, kSpace, j);
    alpha[j] += s[j].real();
    beta[j-1] += s[j-1].real();    
    count++;
    
    measOrthoDev(s, r, kSpace, j);
    beta[j] = norm(r);
    err = 0.0;
    for(int i=0; i<j+1; i++) err = std::max(err, abs(s[i].real()));
    cond = (DBL_EPSILON)*beta[j];
    
  }   
}

void iterRefineComplex(double &rnorm,
		       std::vector<Complex*> &kSpace, Complex *r,
		       Eigen::MatrixXcd &upperHess, int j) {
  
  std::vector<Complex> s(j+1);
  //%----------------------------------------------------%
  //| Compute V_{j}^T * B * r_{j}.                       |
  //| WORKD(IRJ:IRJ+J-1) = v(:,1:J)'*WORKD(IPJ:IPJ+N-1). |
  //%----------------------------------------------------%  
  measOrthoDev(s, r, kSpace, j);

  //%---------------------------------------------%
  //| Compute the correction to the residual:     |
  //| r_{j} = r_{j} - V_{j} * WORKD(IRJ:IRJ+J-1). |
  //| The correction to H is v(:,1:J)*H(1:J,1:J)  |
  //| + v(:,1:J)*WORKD(IRJ:IRJ+J-1)*e'_j.         |
  //%---------------------------------------------%

  double rnorm1 = 0.0;
  int count = 0;
  bool orth = false;
  while (count < 5 && !orth ) {

    // r = r - s_{i} * v_{i}
    orthogonalise(s, kSpace[j+1], kSpace, j);
    for(int i=0; i<j+1; i++) upperHess(i,j) += s[i];
    count++;
    
    copy(r, kSpace[j+1]);
    rnorm1 = dznrm2(Nvec, kSpace[j+1], 1);

    if( rnorm1 > 0.717*rnorm) {

      //%---------------------------------------%
      //| No need for further refinement.       |
      //| The cosine of the angle between the   |
      //| corrected residual vector and the old |
      //| residual vector is greater than 0.717 |
      //| In other words the corrected residual |
      //| and the old residual vector share an  |
      //| angle of less than arcCOS(0.717)      |
      //%---------------------------------------%

      rnorm = rnorm1;
      orth = true;
    } else {
      
      //%------------------------------------------------%
      //| Another iterative refinement step is required. |
      //%------------------------------------------------%

      rnorm = rnorm1;
      measOrthoDev(s, r, kSpace, j);
    }    
  }
}

void lanczosStep(Complex **mat, std::vector<Complex*> &kSpace,
		 double *beta, double *alpha,
		 Complex *r, int num_keep, int j, double a_min, double a_max) {

  //Compute r = A * v_j - b_{j-i} * v_{j-1}      
  //r = A * v_j
  matVec(mat, r, kSpace[j]);
  //chebyOp(mat, r, kSpace[j], a_min, a_max);

  double wnorm = norm(r);
  
  //a_j = v_j^dag * r
  alpha[j] = cDotProd(kSpace[j], r).real();    
  
  //r = r - a_j * v_j
  axpy(-alpha[j], kSpace[j], r);

  int start = (j > num_keep && j>0) ? j - 1 : 0;
  //cout << "Start = " << start << endl;
  for (int i = start; i < j; i++) {
    
    // r = r - b_{j-1} * v_{j-1}
    axpy(-beta[i], kSpace[i], r);
  }

  // Orthogonalise r against the kSpace
  if (j > 0 && norm(r) < 0.717*wnorm) {
    
    iterRefineReal(kSpace, r, alpha, beta, j);
  }  
  //b_j = ||r|| 
  beta[j] = normalise(r);
  
  //Prepare next step.
  copy(kSpace[j+1], r);
}

//std::vector<Complex*> &upperHess,
void arnoldiStep(Complex **mat, std::vector<Complex*> &kSpace,
		 Eigen::MatrixXcd &upperHess,		 
		 Complex *r, int j) {
  
  double unfl = DBL_MIN;
  double ulp = DBL_EPSILON;
  double smlnum = unfl*( Nvec / ulp );
  double temp1 = 0.0;

  double rnorm = norm(r);
  double betaj = rnorm;
  copy(kSpace[j], r);
  
  //%---------------------------------------------------------%
  //| STEP 2:  v_{j} = r_{j-1}/rnorm and p_{j} = p_{j}/rnorm  |
  //| Note that p_{j} = B*r_{j-1}. In order to avoid overflow |
  //| when reciprocating a small RNORM, test against lower    |
  //| machine bound.                                          |
  //%---------------------------------------------------------%  
  // DMH: r{j-1} is already normalised and copied into v_{j} (kSpace[j])
  //      For the moment, B = I and so p_{j} = r_{j}.
  
  if(betaj >= unfl) {
    temp1 = 1.0/rnorm;
    zdscal(Nvec, temp1, kSpace[j], 1);    
    zdscal(Nvec, temp1, r, 1);    
  } else {

    //%-----------------------------------------%
    //| To scale both v_{j} and p_{j} carefully |
    //| use LAPACK routine zlascl               |
    //%-----------------------------------------%
    
    zlascl(rnorm, 1.0, kSpace[j]);
    zlascl(rnorm, 1.0, r);
  }
  
  //%------------------------------------------------------%
  //| STEP 3:  r_{j} = OP*v_{j}; Note that p_{j} = B*v_{j} |
  //| Note that this is not quite yet r_{j}. See STEP 4    |
  //%------------------------------------------------------%
  
  matVec(mat, r, kSpace[j]);

  //%------------------------------------------%
  //| Put another copy of OP*v_{j} into RESID. |
  //%------------------------------------------%
  // DMH: use kSpace[j+1] as space for RESID

  copy(kSpace[j+1], r);

  //%---------------------------------------%
  //| STEP 4:  Finish extending the Arnoldi |
  //|          factorization to length j.   |
  //%---------------------------------------%

  double wnorm = dznrm2(Nvec, kSpace[j+1], 1);
  
  //%-----------------------------------------%
  //| Compute the j-th residual corresponding |
  //| to the j step factorization.            |
  //| Use Classical Gram Schmidt and compute: |
  //| w_{j} <-  V_{j}^T * B * OP * v_{j}      |
  //| r_{j} <-  OP*v_{j} - V_{j} * w_{j}      |
  //%-----------------------------------------%

  //%------------------------------------------%
  //| Compute the j Fourier coefficients w_{j} |
  //| WORKD(IPJ:IPJ+N-1) contains B*OP*v_{j}.  |
  //%------------------------------------------%  
  for (int i = 0; i < j+1; i++) {
    //H_{i,j} = v_i^dag * r
    upperHess(i,j) = cDotProd(kSpace[i], kSpace[j+1]);
  }

  //%--------------------------------------%
  //| Orthogonalize r_{j} against V_{j}.   |
  //| RESID contains OP*v_{j}. See STEP 3. | 
  //%--------------------------------------%
  for (int i = 0; i < j+1; i++) {
    //r = r - v_j * H_{j,i}
    caxpy(-1.0*upperHess(i,j), kSpace[i], kSpace[j+1]);
  }
  
  if(j>0) {
    upperHess(j,j-1).real(betaj);
    upperHess(j,j-1).imag(0.0);
  }
    
  copy(r, kSpace[j+1]);

  //%------------------------------%
  //| Compute the B-norm of r_{j}. |
  //%------------------------------%

  rnorm = dznrm2(Nvec, kSpace[j+1], 1);
  
  //%-----------------------------------------------------------%
  //| STEP 5: Re-orthogonalization / Iterative refinement phase |
  //| Maximum NITER_ITREF tries.                                |
  //|                                                           |
  //|          s      = V_{j}^T * B * r_{j}                     |
  //|          r_{j}  = r_{j} - V_{j}*s                         |
  //|          alphaj = alphaj + s_{j}                          |
  //|                                                           |
  //| The stopping criteria used for iterative refinement is    |
  //| discussed in Parlett's book SEP, page 107 and in Gragg &  |
  //| Reichel ACM TOMS paper; Algorithm 686, Dec. 1990.         |
  //| Determine if we need to correct the residual. The goal is |
  //| to enforce ||v(:,1:j)^T * r_{j}|| .le. eps * || r_{j} ||  |
  //| The following test determines whether the sine of the     |
  //| angle between  OP*x and the computed residual is less     |
  //| than or equal to 0.717.                                   |
  //%-----------------------------------------------------------%
    
    
  if (rnorm < 0.717*wnorm) iterRefineComplex(rnorm, kSpace, r, upperHess, j);
  
  //%----------------------------------------------%
  //| Branch here directly if iterative refinement |
  //| wasn't necessary or after at most DMH: 5     |
  //| steps of iterative refinement.               |
  //%----------------------------------------------%
  
  if (j == (int)kSpace.size()-2 ) {
    cout << "Check for splitting and deflation" << endl;
    for(int i = 0; i < (int)kSpace.size()-2; i++) {
      
      //%--------------------------------------------%
      //| Check for splitting and deflation.         |
      //| Use a standard test as in the QR algorithm |
      //| REFERENCE: LAPACK subroutine zlahqr        |
      //%--------------------------------------------%

      double tst1 = (dlapy2(upperHess(i,i).real(),upperHess(i,i).imag()) +
		     dlapy2(upperHess(i+1,i+1).real(),upperHess(i+1,i+1).imag()));
      if( tst1 == 0.0 ) {
	cout << "TST1 HIT!" << endl;
	tst1 = zlanhs(upperHess, (int)kSpace.size()-2);
      }
      if(dlapy2(upperHess(i+1,i).real(),upperHess(i+1,i).imag()) <= max(ulp*tst1, smlnum ) ) { 
	upperHess(i+1,i) = 0.0;
      }
    }
  }
}


void reorder(std::vector<Complex*> &kSpace, double *alpha, int nKr, bool reverse) {
  int i = 0;
  Complex temp[Nvec];
  if (reverse) {
    while (i < nKr) {
      if ((i == 0) || (alpha[i - 1] >= alpha[i]))
	i++;
      else {
	double tmp = alpha[i];
	alpha[i] = alpha[i - 1];
	alpha[--i] = tmp;
	copy(temp, kSpace[i]);
	copy(kSpace[i], kSpace[i-1]);
	copy(kSpace[i-1], temp);
      }
    }
  } else {
    while (i < nKr) {
      if ((i == 0) || (alpha[i - 1] <= alpha[i])) 
	i++;
      else {
	double tmp = alpha[i];
	alpha[i] = alpha[i - 1];
	alpha[--i] = tmp;
	copy(temp, kSpace[i]);
	copy(kSpace[i], kSpace[i-1]);
	copy(kSpace[i-1], temp);
      }
    }
  }
}

void eigensolveFromArrowMat(int num_locked, int arrow_pos, int nKr, double *alpha, double *beta, double *residua, bool reverse) {
  
  int dim = nKr - num_locked;
  
  // Eigen objects
  MatrixXd A = MatrixXd::Zero(dim, dim);
  ritz_mat.resize(dim * dim);
  for (int i = 0; i < dim * dim; i++) ritz_mat[i] = 0.0;

  // Optionally invert the spectrum
  if (reverse) {
    for (int i = num_locked; i < nKr - 1; i++) {
      alpha[i] *= -1.0;
      beta[i] *= -1.0;
    }
    alpha[nKr - 1] *= -1.0;
  }
  
  // Construct arrow mat A_{dim,dim}
  for (int i = 0; i < dim; i++) {    
    // alpha populates the diagonal
    A(i,i) = alpha[i + num_locked];
  }
  
  for (int i = 0; i < arrow_pos - 1; i++) {  
    // beta populates the arrow
    A(i, arrow_pos - 1) = beta[i + num_locked];
    A(arrow_pos - 1, i) = beta[i + num_locked];
  }
  
  for (int i = arrow_pos - 1; i < dim - 1; i++) {
    // beta populates the sub-diagonal
    A(i, i + 1) = beta[i + num_locked];
    A(i + 1, i) = beta[i + num_locked];
  }
  
  // Eigensolve the arrow matrix 
  eigensolver.compute(A);
  
  // repopulate ritz matrix
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      //Place data in COLUMN major 
      ritz_mat[dim * i + j] = eigensolver.eigenvectors().col(i)[j];
    }
  }
  
  for (int i = 0; i < dim; i++) {
    residua[i + num_locked] = fabs(beta[nKr - 1] * eigensolver.eigenvectors().col(i)[dim - 1]);
    // Update the alpha array
    alpha[i + num_locked] = eigensolver.eigenvalues()[i];
    if (verbose) printf("EFAM: resid = %e, alpha = %e\n", residua[i + num_locked], alpha[i + num_locked]);
  }

  // Put spectrum back in order
  if (reverse) {
    for (int i = num_locked; i < nKr; i++) { alpha[i] *= -1.0; }
  }  
}


void eigensolveFromTriDiag(int dim, double *alpha, double *beta, double *residua, bool reverse) {
  
  // Eigen objects
  MatrixXd A = MatrixXd::Zero(dim, dim);
  ritz_mat.resize(dim * dim);
  for (int i = 0; i < dim * dim; i++) ritz_mat[i] = 0.0;

  // Optionally invert the spectrum
  if (reverse) {
    for (int i = 0; i < dim-1; i++) {
      alpha[i] *= -1.0;
      beta[i] *= -1.0;
    }
    alpha[dim - 1] *= -1.0;
  }
  
  // Construct arrow mat A_{dim,dim}
  for (int i = 0; i < dim; i++) {    
    // alpha populates the diagonal
    A(i,i) = alpha[i];
  }
  
  for (int i = 0; i < dim - 1; i++) {
    // beta populates the sub-diagonal
    A(i, i + 1) = beta[i];
    A(i + 1, i) = beta[i];
  }
  
  // Eigensolve the arrow matrix 
  eigensolver.compute(A);
  
  // repopulate ritz matrix
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      //Place data in COLUMN major 
      ritz_mat[dim * i + j] = eigensolver.eigenvectors().col(i)[j];
    }
  }
  
  for (int i = 0; i < dim; i++) {
    residua[i] = fabs(beta[dim-1] * eigensolver.eigenvectors().col(i)[dim - 1]);
    // Update the alpha array
    alpha[i] = eigensolver.eigenvalues()[i];
    if (verbose) printf("EFAM: resid = %e, alpha = %e\n", residua[i], alpha[i]);
  }

  // Put spectrum back in order
  if (reverse) {
    for (int i = 0; i < dim; i++) { alpha[i] *= -1.0; }
  }  
}

void computeEvals(Complex **mat, std::vector<Complex*> &kSpace, double *residua, Complex *evals, int nEv) {
  
  //temp vector
  Complex temp[Nvec];
  for (int i = 0; i < nEv; i++) {
    // r = A * v_i
    matVec(mat, temp, kSpace[i]);
    
    // lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
    evals[i] = cDotProd(kSpace[i], temp) / norm(kSpace[i]);
    
    // Measure ||lambda_i*v_i - A*v_i||
    Complex n_unit(-1.0, 0.0);
    caxpby(evals[i], kSpace[i], n_unit, temp);
    residua[i] = norm(temp);
  }
}

void rotateVecsReal(std::vector<Complex*> &vecs, Eigen::MatrixXd mat, int num_locked, int iter_keep, int dim) {
  
  //loop over rows of V_k
  for(int j=0; j<Nvec; j++) {    
    
    //put jth row of V_k in temp
    Complex tmp[dim];  
    for(int i=0; i<dim; i++) {
      tmp[i] = vecs[i+num_locked][j];      
    }

    //take product of jth row of V_k and ith column of mat (ith eigenvector of T_k) 
    Complex sum = 0.0;
    for(int i=0; i<iter_keep; i++) {
      
      //Loop over elements to get the y_i[j]th element 
      for(int l=0; l<dim; l++) {
	sum += tmp[l]*mat.col(i)[l];
      }
      
      //Update the Ritz vector
      vecs[i+num_locked][j] = sum;
      sum = 0.0;
    }
  }
}

void rotateVecsComplex(std::vector<Complex*> &vecs, Eigen::MatrixXcd mat, int num_locked, int iter_keep, int dim) {
  
  //loop over rows of V_k
  for(int j=0; j<Nvec; j++) {    
    
    //put jth row of V_k in temp
    Complex tmp[dim];  
    for(int i=0; i<dim; i++) {
      tmp[i] = vecs[i+num_locked][j];      
    }

    //take product of jth row of V_k and ith column of mat (ith eigenvector of T_k) 
    Complex sum = 0.0;
    for(int i=0; i<iter_keep; i++) {
      
      //Loop over elements to get the y_i[j]th element 
      for(int l=0; l<dim; l++) {
	sum += tmp[l]*mat.col(i)[l];
      }
      
      //Update the Ritz vector
      vecs[i+num_locked][j] = sum;
      sum = 0.0;
    }
  }
}

void computeKeptRitz(std::vector<Complex*> &kSpace, int nKr, int num_locked, int iter_keep, double *beta) {
  
  int dim = nKr - num_locked;
  MatrixXd mat = MatrixXd::Zero(dim, iter_keep);
  for (int j = 0; j < iter_keep; j++) 
    for (int i = 0; i < dim; i++) 
      mat(i,j) = ritz_mat[j*dim + i];  
  rotateVecsReal(kSpace, mat, num_locked, iter_keep, dim); 
  
  //Update beta and residual vector
  copy(kSpace[num_locked + iter_keep], kSpace[nKr]);
  for (int i = 0; i < iter_keep; i++)
    beta[i + num_locked] = beta[nKr - 1] * mat.col(i)[nKr-num_locked-1];  
}

void permuteVecs(std::vector<Complex*> &kSpace, Eigen::MatrixXd mat, int num_locked, int size){

  std::vector<int> pivots(size);
  for (int i=0; i<size; i++) {
    for (int j=0; j<size; j++) {
      if(mat(i,j) == 1) {
	pivots[j] = i;
      }
    }
  }

  // Identify cycles in the permutation array.
  // We shall use the sign bit as a marker. If the
  // sign is negative, the vectors has already been
  // swapped into the correct place. A positive
  // value indicates the start of a new cycle.

  Complex temp[Nvec];
  for (int i=0; i<size; i++) {
    //Cycles always start at 0, hence OR statement
    if(pivots[i] > 0 || i==0) {
      int k = i;
      // Identify vector to be placed at i
      int j = pivots[i];
      pivots[i] = -pivots[i];
      while (j > i) {
	copy(temp, kSpace[k+num_locked]);
	copy(kSpace[k+num_locked], kSpace[j+num_locked]);
	copy(kSpace[j+num_locked], temp);
	pivots[j] = -pivots[j];
	k = j;
	j = -pivots[j];
      }
    } else {
      //printf("%d already swapped\n", i);
    }
  }
  for (int i=0; i<size; i++) {
    if (pivots[i] > 0) {
      printf("Error at %d\n", i);
      exit(0);
    }
  }
}

void computeKeptRitzLU(std::vector<Complex*> &kSpace, int nKr, int num_locked, int iter_keep, int batch, double *beta, int iter) {
  
  int offset = nKr + 1;
  int dim = nKr - num_locked;

  printf("dim = %d\n", dim);
  printf("iter_keep = %d\n", iter_keep);
  printf("num_locked = %d\n", num_locked);
  printf("kspace size = %d\n", (int)kSpace.size());

  int batch_size = batch;
  int full_batches = iter_keep/batch_size;
  int batch_size_r = iter_keep%batch_size;
  bool do_batch_remainder = (batch_size_r != 0 ? true : false);
  
  printf("batch_size = %d\n", batch_size);
  printf("full_batches = %d\n", full_batches);
  printf("batch_size_r = %d\n", batch_size_r);
  
  if ((int)kSpace.size() < offset + batch_size) {
    for (int i = kSpace.size(); i < offset + batch_size; i++) {
      kSpace.push_back(new Complex[Nvec]);
    }
  }

  // Zero out the work space
  for (int i = offset; i < offset + batch_size; i++) {
    zero(kSpace[i]);
  }

  MatrixXd mat = MatrixXd::Zero(dim, iter_keep);
  for (int j = 0; j < iter_keep; j++) 
    for (int i = 0; i < dim; i++) 
      mat(i,j) = ritz_mat[j*dim + i];
  
  Eigen::FullPivLU<MatrixXd> matLU(mat);
  // RitzLU now contains the LU decomposition
  
  MatrixXd matUpper = MatrixXd::Zero(dim,iter_keep);
  matUpper = matLU.matrixLU().triangularView<Eigen::Upper>();
  MatrixXd matLower = MatrixXd::Identity(dim,dim);
  matLower.block(0,0,dim,iter_keep).triangularView<Eigen::StrictlyLower>() = matLU.matrixLU();

  permuteVecs(kSpace, matLU.permutationP().inverse(), num_locked, dim);    
  
  // Defines the column element (row index)
  // from which we reference other indicies
  int i_start, i_end, j_start, j_end;
  
  // Do L Portion
  //---------------------------------------------------------------------------
  // Loop over full batches
  for (int b = 0; b < full_batches; b++) {

    // batch triangle
    i_start = b*batch_size;
    i_end   = (b+1)*batch_size; 
    j_start = i_start;
    j_end   = i_end;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = j; i < i_end; i++) {	
	axpy(matLower.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }
    //batch pencil
    i_start = (b+1)*batch_size;
    i_end   = dim;
    j_start = b*batch_size;
    j_end   = (b+1)*batch_size;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = i_start; i < i_end; i++) {
	axpy(matLower.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }
    // copy back to correct position
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      copy(kSpace[j + num_locked], kSpace[k]);
      zero(kSpace[k]);
    }    
  }
  
  if(do_batch_remainder) {
    // remainder triangle
    i_start = full_batches*batch_size;
    i_end   = iter_keep;
    j_start = i_start;
    j_end   = i_end;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = j; i < i_end; i++) {
	axpy(matLower.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }   
    }
    //remainder pencil
    i_start = iter_keep;
    i_end   = dim;
    j_start = full_batches*batch_size;
    j_end   = iter_keep;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = i_start; i < i_end; i++) {	
	axpy(matLower.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }
    // copy back to correct position
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      copy(kSpace[j + num_locked], kSpace[k]);
      zero(kSpace[k]);
    }
  }
    
  // Do U Portion
  //---------------------------------------------------------------------------
  if(do_batch_remainder) {

    // remainder triangle
    i_start = full_batches*batch_size;
    i_end   = iter_keep;
    j_start = i_start;
    j_end   = i_end;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = i_start; i < j+1; i++) {	
	axpy(matUpper.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }
    //remainder pencil
    i_start = 0;
    i_end   = full_batches*batch_size; 
    j_start = full_batches*batch_size;
    j_end   = iter_keep;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = i_start; i < i_end; i++) {
	axpy(matUpper.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }
    // copy back to correct position
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      copy(kSpace[j + num_locked], kSpace[k]);
      zero(kSpace[k]);
    }    
  }
  
  // Loop over full batches
  for (int b = full_batches-1; b >= 0; b--) {

    // batch triangle
    i_start = b*batch_size;
    i_end   = (b+1)*batch_size; 
    j_start = i_start;
    j_end   = i_end;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = i_start; i < j+1; i++) {	
	axpy(matUpper.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }

    if(b>0) {
      //batch pencil
      i_start = 0;
      i_end   = b*batch_size; 
      j_start = b*batch_size;
      j_end   = (b+1)*batch_size;
      for (int j = j_start; j < j_end; j++) {
	int k = offset + j - j_start;
	for (int i = i_start; i < i_end; i++) {	  
	  axpy(matUpper.col(j)[i], kSpace[num_locked + i], kSpace[k]);
	}
      }
    }
    // copy back to correct position
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      copy(kSpace[j + num_locked], kSpace[k]);
      zero(kSpace[k]);
    }    
  }

  permuteVecs(kSpace, matLU.permutationQ().inverse(), num_locked, iter_keep);

  //Update residual
  copy(kSpace[num_locked + iter_keep], kSpace[nKr]);
  
  //Update beta
  for (int i = 0; i < iter_keep; i++)
    beta[i + num_locked] = beta[nKr - 1] * ritz_mat[dim * (i + 1) - 1];
  
}

void givensQRtriDiag(Eigen::MatrixXcd &triDiag, Eigen::MatrixXcd &Q, int nKr, double shift) {

  // Prep workspace
  std::vector<double> diag; // diagonal elements of T
  std::vector<double> lsub; // Lower sub diagonal elements
  std::vector<double> usub; // Upper sub diagonal elements
  std::vector<double> cos;  // cosines
  std::vector<double> sin;  // sines
  
  diag.reserve(nKr);
  cos.reserve(nKr-1);
  sin.reserve(nKr-1);
  lsub.reserve(nKr-1);
  usub.reserve(nKr-1);
    
  for(int i=0; i<nKr; i++) diag.push_back(triDiag(i,i).real() - shift);
  for(int i=0; i<nKr-1; i++) cos.push_back(0.0);
  for(int i=0; i<nKr-1; i++) sin.push_back(0.0);
  for(int i=0; i<nKr-1; i++) lsub.push_back(triDiag(i+1,i).real());
  for(int i=0; i<nKr-1; i++) usub.push_back(triDiag(i,i+1).real());

  double common, r, ratio;

  // Compute Givens Rotations
  for(int i=0; i<nKr-1; i++) {

    double diag_sign = copysign(1.0, diag[i]);
    double lsub_sign = copysign(1.0, lsub[i]);
    
    if (abs(diag[i]) > abs(lsub[i])) {
      ratio = abs(lsub[i])/abs(diag[i]);
      common = sqrt(1.0 + ratio * ratio);
      cos[i] = diag_sign / common;
      r = abs(diag[i]) * common;
      sin[i] = -lsub[i] / r;
      //cout << "Sanity["<<i<<"] " << cos[i]*cos[i] << " + " <<  abs(sin[i])*abs(sin[i]) << " = " << cos[i]*cos[i] + pow(sin[i], 2) << endl;
    } else {
      if(abs(lsub[i]) < 1e-10) {
	r = 0.0;
	cos[i] = 1.0;
	sin[i] = 0.0;
	//cout << "Sanity["<<i<<"] " << cos[i]*cos[i] << " + " <<  abs(sin[i])*abs(sin[i]) << " = " << cos[i]*cos[i] + pow(sin[i], 2) << endl;
      }
      ratio = abs(diag[i])/abs(lsub[i]);
      common = sqrt(1.0 + ratio * ratio);
      sin[i] = -lsub_sign / common;
      r =  abs(lsub[i]) * common;
      cos[i] = diag[i] / r;
      
      //cout << "Sanity["<<i<<"] " << cos[i]*cos[i] << " + " <<  abs(sin[i])*abs(sin[i]) << " = " << cos[i]*cos[i] + pow(sin[i], 2) << endl;
    }

    diag[i] = r;
    lsub[i] = 0.0;

    double tmp = usub[i];
    usub[i]   = cos[i]*tmp - sin[i]*diag[i+1];
    diag[i+1] = sin[i]*tmp + cos[i]*diag[i+1];
    
    if(i<nKr-2) {
      usub[i+1] *= cos[i];
    }
  }  

  // Update Q matrix
  double tmp;
  for(int i=0; i<nKr-1; i++) {
    for(int j=0; j<nKr; j++) {
      tmp = Q(j,i).real();
      Q(j,i).real(cos[i]*tmp - sin[i]*Q(j,i+1).real());
      Q(j,i+1).real(sin[i]*tmp + cos[i]*Q(j,i+1).real());
    }
  }

  // Update Tridiag
  triDiag.setZero();
  for(int i=0; i<nKr; i++) triDiag(i,i).real(diag[i]);
  for(int i=0; i<nKr-1; i++) {
    double tmp11 = triDiag(i,i).real();
    double tmp12 = usub[i];
    double tmp22 = diag[i+1];

    triDiag(i,i).real(cos[i]*tmp11 - sin[i]*tmp12);
    triDiag(i+1,i).real(-sin[i]*tmp22);
    triDiag(i+1,i+1).real(cos[i]*tmp22);
  }
  
  for(int i=0; i<nKr-1; i++) triDiag(i,i+1) = triDiag(i+1,i);
  for(int i=0; i<nKr; i++) {
    double tmp = triDiag(i,i).real() + shift; 
    triDiag(i,i).real(tmp);
  }  
}

void applyShift(Eigen::MatrixXcd &UH, Eigen::MatrixXcd &Q,		
		int istart, int iend, int nKr, Complex shift, int shift_num, int iter) {
  
  //%------------------------------------------------------%
  //| Construct the plane rotation G to zero out the bulge |
  //%------------------------------------------------------%

  bool ah_debug = true;
  
  std::vector<double> cos(1);
  std::vector<Complex> sin(1);
  std::vector<Complex> r(1);
  
  Complex t;
  Complex cZero(0.0,0.0);  
  Complex f = UH(istart,   istart) - shift;
  Complex g = UH(istart+1, istart);

  for(int i = istart; i < iend-1; i++) {


    if(ah_debug) {
      cout << "f= " << f << endl;
      cout << "g= " << g << endl;      
    }
    
    zlartg(f, g, cos, sin, r);
    if(ah_debug) {
      // Sanity check
      //cout << " shift " << shift << " Sanity["<<i<<"] " << cos[0]*cos[0] << " + " <<  abs(sin[0])*abs(sin[0]) << " = " << cos[0]*cos[0] + pow(sin[0].real(), 2) + pow(sin[0].imag(),2) << endl;
      cout << "r= " << r[0] << endl;
      cout << "c= " << cos[0] << endl;
      cout << "s= " << sin[0] << endl;
      cout << " shift " << shift_num << " sigma " << shift << endl;
      cout << " istart = " << istart << " iend = " << iend << endl;
    }
    if(i > istart) {
      UH(i,i-1) = r[0];
      UH(i+1,i-1) = cZero;	      
    }

    //%---------------------------------------------%
    //| Apply rotation to the left of H;  H <- G'*H |
    //%---------------------------------------------%
    //do 50
    for(int j = i; j < nKr; j++) {
      if(ah_debug) {
	cout<<"pre  h("<<i+1<<","<<j<<")="<<UH(i+1,j)<<endl;
	cout<<"pre  h("<<i<<","<<j<<")="<< UH(i,j)<<endl;
	cout<<"cos=" << cos[0] << " sin = " << sin[0] <<endl;
      }
      t         =  cos[0]       * UH(i,j) + sin[0] * UH(i+1,j);
      UH(i+1,j) = -conj(sin[0]) * UH(i,j) + cos[0] * UH(i+1,j);
      UH(i,j) = t;
      if(ah_debug) {
	cout<<"post h("<<i+1<<","<<j<<")="<<UH(i+1,j)<<endl;
	cout<<"post h("<<i<<","<<j<<")="<< UH(i,j)<<endl;
      }
    }

    //%---------------------------------------------%
    //| Apply rotation to the right of H;  H <- H*G |
    //%---------------------------------------------%
    //do 60
    if(ah_debug) cout << "min60 = min(" << i+1+2 << "," << iend << ") = " << std::min(i+1+2, iend) << endl;
    for(int j = 0; j<std::min(i+1+2, iend); j++) {
      if(ah_debug) {
	cout<<"pre  h("<<j<<","<<i+1<<")="<< UH(j,i+1)<<endl;
	cout<<"pre  h("<<j<<","<<i<<")="<< UH(j,i)<<endl;
	cout<<"cos=" << cos[0] << " sin = " << sin[0] <<endl;
      }
      t         =  cos[0] * UH(j,i) + conj(sin[0]) * UH(j,i+1);
      UH(j,i+1) = -sin[0] * UH(j,i) + cos[0]       * UH(j,i+1);
      UH(j,i) = t;
      if(ah_debug) {
	//if(abs(UH(j,i+1)) < 1e-10) UH(j,i+1) = 0.0;
	//if(abs(UH(j,i  )) < 1e-10) UH(j,i  ) = 0.0;
	cout<<"post h("<<j<<","<<i+1<<")="<< UH(j,i+1)<<endl;
	cout<<"post h("<<j<<","<<i<<")="<< UH(j,i)<<endl;
      }
    }

    //%-----------------------------------------------------%
    //| Accumulate the rotation in the matrix Q;  Q <- Q*G' |
    //%-----------------------------------------------------%
    // do 70
    if(ah_debug) cout << "min70 = " << std::min(i+1 + shift_num+1, nKr) << endl;
    for(int j = 0; j<std::min(i+1 + shift_num+1, nKr); j++) {
      if(ah_debug) {
	cout<<"pre q("<<j<<","<<i+1<<")="<< Q(j,i+1)<<endl;
	cout<<"pre q("<<j<<","<<i<<")="<< Q(j,i)<<endl;
	cout<<"cos=" << cos[0] << " sin = " << sin[0] <<endl;
      }
      t        =  cos[0] * Q(j,i) + conj(sin[0]) * Q(j,i+1);
      Q(j,i+1) = -sin[0] * Q(j,i) + cos[0]       * Q(j,i+1);
      Q(j,i) = t;
      if(ah_debug) {
	cout<<"post q("<<j<<","<<i+1<<")="<< Q(j,i+1)<<endl;
	cout<<"post q("<<j<<","<<i<<")="<< Q(j,i)<<endl;
      }
    }
    
    if(i < iend-2) {
      f = UH(i+1,i);
      g = UH(i+2,i);
    }
  }	  
}


void givensQRUpperHess(Eigen::MatrixXcd &UH, Eigen::MatrixXcd &Q, int nKr,
		       int shifts, int shift_num, int step_start, Complex shift, int iter){
  
//https://cug.org/5-publications/proceedings_attendee_lists/1997CD/S96PROC/345_349.PDF
  // This code was put in place to deal with the difference between IEEE and CRAY
  // double prec formats. Now that we use IEEE almost exclusivley, we use the IEEE
  // starndards
  double unfl = DBL_MIN;
  double ulp = DBL_EPSILON;
  double smlnum = unfl*(Nvec/ulp);
  
  //%----------------------------------------%
  //| Check for splitting and deflation. Use |
  //| a standard test as in the QR algorithm |
  //| REFERENCE: LAPACK subroutine zlahqr    |
  //%----------------------------------------%

  int istart = 0;
  int iend = -1;

  bool g_debug = true;
  
  //znapps.f line 281
  // do 30 loop
  for(int i=istart; i<nKr-1; i++) {
    
    double tst1 = abs(UH(i,i).real()) + abs(UH(i,i).imag()) + abs(UH(i+1,i+1).real()) + abs(UH(i+1,i+1).imag());
    
    if( tst1 == 0 ) {
      cout << " *** TST1 hit "<< endl; 
      tst1 = zlanhs(UH, nKr - shift_num);
    }

    if(g_debug) {
      cout << "TEST at Iter = " << iter << " loop = " << i << " shift = " << shift_num << endl;
      cout << tst1 << " " << abs(UH(i+1,i).real()) << " " << std::max(ulp*tst1, smlnum) << endl;
    }
    if (abs(UH(i+1,i).real()) <= std::max(ulp*tst1, smlnum)) {
      if(g_debug) cout << "UH split at " << i << " shift = " << shift_num << endl;
      iend = i+1;
      if(g_debug) cout << "istart = " << istart << " iend = " << iend << endl;
      UH(i+1,i) = 0.0;
      if(istart == iend){
	//if(istart == i){
	
	//%------------------------------------------------%
	//| No reason to apply a shift to block of order 1 |
	//| or if the current block starts after the point |
	//| of compression since we'll discard this stuff  |
	//%------------------------------------------------%    
	
	if(g_debug) cout << " No need for single block rotation at " << i << endl;
      } else if (istart > step_start) {
	if(g_debug) cout << " block rotation beyond " << step_start << endl;
      } else if( istart <= step_start) {
	applyShift(UH, Q, istart, iend, nKr, shift, shift_num, iter);
      }
      istart = iend + 1;
    }
  }

  iend = nKr;
  if(g_debug) {
    cout << "At End istart = " << istart << " iend = " << iend << endl;
    if(istart == iend) cout << " No need for single block rotation at " << istart << endl;
  }
  // If we finish the i loop with a istart less that step_start, we must
  // do a final set of shifts
  if(istart <= step_start && istart != iend) {
    //perform final block compression
    applyShift(UH, Q, istart, iend, nKr, shift, shift_num, iter);
  }
  
  //%---------------------------------------------------%
  //| Perform a similarity transformation that makes    |
  //| sure that the compressed H will have non-negative |
  //| real subdiagonal elements.                        |
  //%---------------------------------------------------%
  
  if( shift_num == shifts-1 ) {
    //do 120
    for(int j=0; j<step_start; j++) {
      if (UH(j+1,j).real() < 0.0 || UH(j+1,j).imag() != 0.0 ) {
	Complex t = UH(j+1,j) / dlapy2(UH(j+1,j).real(), UH(j+1,j).imag());	
	for(int i=0; i<nKr-j; i++) UH(j+1,i) *= conj(t);
	for(int i=0; i<std::min(j+1+2, nKr); i++) UH(i,j+1) *= t;
	for(int i=0; i<std::min(j+1+shifts+1,nKr); i++) Q(i,j+1) *= t;
	UH(j+1,j).imag(0.0);
      }
    }
    
    //do 130
    for(int i=0; i<step_start; i++) {

      //%--------------------------------------------%
      //| Final check for splitting and deflation.   |
      //| Use a standard test as in the QR algorithm |
      //| REFERENCE: LAPACK subroutine zlahqr.       |
      //| Note: Since the subdiagonals of the        |
      //| compressed H are nonnegative real numbers, |
      //| we take advantage of this.                 |
      //%--------------------------------------------%
      
      double tst1 = abs(UH(i,i).real()) + abs(UH(i,i).imag()) + abs(UH(i+1,i+1).real()) + abs(UH(i+1,i+1).imag());
      if( tst1 == 0 ) {
	cout << " ********* TST1 hit ********** "<< endl;
	tst1 = zlanhs(UH, step_start);
      }
      if (abs(UH(i+1,i).real()) <= std::max(ulp*tst1, smlnum)) {
	UH(i+1,i) = 0.0;
	if(g_debug) cout << "Zero out UH("<<i+1<<","<<i<<") by hand"<<endl;;
      }
    }
  }
}

void chebyOp(Complex **mat, Complex *out, Complex *in, double a, double b) {
  
  // Compute the polynomial accelerated operator.
  //double a = 15;
  //double b = 25;
  double delta = (b - a) / 2.0;
  double theta = (b + a) / 2.0;
  double sigma1 = -delta / theta;
  double sigma;
  double d1 = sigma1 / delta;
  double d2 = 1.0;
  double d3;

  // out = d2 * in + d1 * out
  // C_1(x) = x
  matVec(mat, out, in);
  caxpby(d2, in, d1, out);
  
  Complex tmp1[Nvec];
  Complex tmp2[Nvec];
  Complex tmp3[Nvec];
  
  copy(tmp1, in);
  copy(tmp2, out);
  
  // Using Chebyshev polynomial recursion relation,
  // C_{m+1}(x) = 2*x*C_{m} - C_{m-1}
  
  double sigma_old = sigma1;
  
  // construct C_{m+1}(x)
  for (int i = 2; i < 10; i++) {
    sigma = 1.0 / (2.0 / sigma1 - sigma_old);
      
    d1 = 2.0 * sigma / delta;
    d2 = -d1 * theta;
    d3 = -sigma * sigma_old;

    // mat*C_{m}(x)
    matVec(mat, out, tmp2);

    Complex d1c(d1, 0.0);
    Complex d2c(d2, 0.0);
    Complex d3c(d3, 0.0);

    copy(tmp3, tmp2);
    
    caxpby(d3c, tmp1, d2c, tmp3);
    caxpy(d1c, out, tmp3);
    copy(tmp2, tmp3);
    
    sigma_old = sigma;
  }
  copy(out, tmp2);
}

#endif

