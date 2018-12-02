#ifndef LINALGHELPERS_H
#define LINALGHELPERS_H

#include <omp.h>

//Simple Linear Algebra Helpers
void zero(double *vec) {
#pragma omp parallel for
  for(int i=0; i<Nvec; i++) vec[i] = 0.0;
}

void copy(double *vec1, double *vec2) {
#pragma omp parallel for
  for(int i=0; i<Nvec; i++) vec1[i] = vec2[i];
}

void ax(double C, double *vec) {
#pragma omp parallel for
  for(int i=0; i<Nvec; i++) vec[i] *= C;
}

void axpy(double C, double *vec1, double *vec2) {
#pragma omp parallel for
  for(int i=0; i<Nvec; i++) vec2[i] += C*vec1[i];
}

void axpby(double C, double *vec1, double D, double *vec2) {
#pragma omp parallel for
  for(int i=0; i<Nvec; i++) {
    vec2[i] *= D;
    vec2[i] += C*vec1[i];
  }
}

double dotProd(double *vec2, double *vec1) {
  double prod = 0.0;
#pragma omp parallel for reduction(+:prod) 
  for(int i=0; i<Nvec; i++) prod += vec1[i]*vec2[i];
  return prod;
}

double norm(double *vec) {
  double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
  for(int i=0; i<Nvec; i++) sum += vec[i]*vec[i];
  return sum;
}

double normalise(double *vec) {
  double sum = norm(vec);
  ax(1.0/sqrt(sum), vec); 
  return sum;
}



//Orthogonalise r against the j vectors in vectorSpace
void orthogonalise(double *r, double **vectorSpace, int j) {
  
  double s = 0.0;
  for(int i=0; i<j; i++) {
    s = dotProd(vectorSpace[i], r);
    axpy(-s, vectorSpace[i], r);
  }
}

void matVec(double **mat, double *out, double *in) {

  double tmp[Nvec];
  //Loop over rows of matrix
#pragma omp parallel for 
  for(int i=0; i<Nvec; i++) {
    tmp[i] = dotProd(mat[i], in);    
  }
#pragma omp parallel for 
  for(int i=0; i<Nvec; i++) {
    out[i] = tmp[i];
  }
}

#endif
