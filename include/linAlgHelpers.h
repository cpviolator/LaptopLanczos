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
