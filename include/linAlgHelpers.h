#ifndef LINALGHELPERS_H
#define LINALGHELPERS_H

#include <omp.h>

//Simple Complex Linear Algebra Helpers
void zero(Complex *x) {
  //#pragma omp parallel for
  for(int i=0; i<Nvec; i++) x[i] = 0.0;
}

void copy(Complex *x, Complex *y) {
  //#pragma omp parallel for
  for(int i=0; i<Nvec; i++) x[i] = y[i];
}

void ax(double a, Complex *x) {
  //#pragma omp parallel for
  for(int i=0; i<Nvec; i++) x[i] *= a;
}

void cax(Complex a, Complex *x) {
  //#pragma omp parallel for
  for(int i=0; i<Nvec; i++) x[i] *= a;
}

void axpy(double a, Complex *x, Complex *y) {
  //#pragma omp parallel for
  for(int i=0; i<Nvec; i++) y[i] += a*x[i];
}

void caxpy(Complex a, Complex *x, Complex *y) {
  //#pragma omp parallel for
  for(int i=0; i<Nvec; i++) y[i] += a*x[i];
}

void axpby(double a, Complex *x, double b, Complex *y) {
  //#pragma omp parallel for
  for(int i=0; i<Nvec; i++) {
    y[i] *= b;
    y[i] += a*x[i];
  }
}

void caxpby(Complex a, Complex *x, Complex b, Complex *y) {
  //#pragma omp parallel for
  for(int i=0; i<Nvec; i++) {
    y[i] *= b;
    y[i] += a*x[i];
  }
}

Complex dotProd(Complex *x, Complex *y) {
  Complex prod = 0.0;
  //#pragma omp parallel for reduction(+:prod) 
  for(int i=0; i<Nvec; i++) prod += x[i]*y[i];
  return prod;
}


Complex cDotProd(Complex *x, Complex *y) {
  Complex prod = 0.0;
  //#pragma omp parallel for reduction(+:prod) 
  for(int i=0; i<Nvec; i++) prod += conj(x[i])*y[i];
  return prod;
}

double norm2(Complex *x) {
  double sum = 0.0;
  //#pragma omp parallel for reduction(+:sum)
  for(int i=0; i<Nvec; i++) sum += (conj(x[i])*x[i]).real();
  return sum;
}

double norm(Complex *x) {
  double sum = 0.0;
  //#pragma omp parallel for reduction(+:sum)
  for(int i=0; i<Nvec; i++) sum += (conj(x[i])*x[i]).real();
  return sqrt(sum);
}

double normalise(Complex *x) {
  double sum = norm(x);
  ax(1.0/sum, x); 
  return sum;
}

//Orthogonalise r against the j vectors in vectorSpace
void orthogonalise(Complex *r, std::vector<Complex*> vectorSpace, int j) {
  
  Complex s = 0.0;
  for(int i=0; i<j+1; i++) {
    s = cDotProd(vectorSpace[i], r);
    s *= -1.0;
    caxpy(s, vectorSpace[i], r);
  }
}

void matVec(Complex **mat, Complex *out, Complex *in) {
  
  Complex temp[Nvec];
  zero(temp);
  //Loop over rows of matrix
  //#pragma omp parallel for 
  for(int i=0; i<Nvec; i++) {
    temp[i] = dotProd(&mat[i][0], in);    
  }
  copy(out, temp);  
  //#pragma omp parallel for 
  //for(int i=0; i<Nvec; i++) {
  //out[i] = tmp[i];
  //}
}

#endif
