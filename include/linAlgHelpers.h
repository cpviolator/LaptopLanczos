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
  double oneOnNorm = 1.0/norm(x);  
  ax(oneOnNorm, x); 
  return 1.0/oneOnNorm;
}

//Orthogonalise r against the j vectors in vectorSpace, populate with res with residua
void orthogonalise(std::vector<Complex> &s, Complex *r, std::vector<Complex*> vectorSpace, int j) {
  
  for(int i=0; i<j+1; i++) {
    s[i] = cDotProd(vectorSpace[i], r);
    caxpy(-s[i], vectorSpace[i], r);
  }
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

// OK Nov 17 12:51
void zlartg(const Complex F, const Complex G, std::vector<double> &cos,
	    std::vector<Complex> &sin, std::vector<Complex> &r) {  
  
  // Compute Givens Rotations. Adapted from zlartg.f
  // ZLARTG generates a plane rotation so that
  // 
  //      [  CS  SN  ]     [ F ]     [ R ]
  //      [  __      ]  .  [   ]  =  [   ]   where CS**2 + |SN|**2 = 1.
  //      [ -SN  CS  ]     [ G ]     [ 0 ]
  
  Complex cZero(0.0,0.0);
  Complex SS, GS, FS;
  double G2, F2, D, DI, FA, GA;
  
  if(G == cZero) {
    r[0] = F;
    cos[0] = 1.0;
    sin[0] = cZero;
    cout << "path 1";
  } else if (F == cZero) {
    double Gabs = sqrt(G.real()*G.real() + G.imag()*G.imag());  
    cos[0] = 0.0;
    sin[0] = conj(G)/Gabs;
    r[0] = Gabs;
    cout << "path 2";
  } else {
    double F1 = abs(F.real()) + abs(F.imag());
    double G1 = abs(G.real()) + abs(G.imag());    
    if (F1 >= G1) {
      GS = G / F1;
      G2 = GS.real()*GS.real() + GS.imag()*GS.imag();
      FS = F / F1;
      F2 = FS.real()*FS.real() + FS.imag()*FS.imag();
      D = sqrt(1.0 + G2/F2);
      cos[0] = 1.0/D;
      sin[0] = conj(GS) * FS * (cos[0] / F2);
      r[0] = F*D;
      cout << "path 3";
    } else {
      FS = F / G1;
      F2 = FS.real()*FS.real() + FS.imag()*FS.imag();
      FA = sqrt(F2);
      GS = G / G1;
      G2 = GS.real()*GS.real() + GS.imag()*GS.imag();
      GA = sqrt(G2);
      D = sqrt(1.0 + F2/G2);
      DI = 1.0/D;
      cos[0] = (FA/GA) * DI;
      SS = (conj(GS) * FS)/(FA*GA);
      sin[0] = SS * DI;
      r[0] = G * SS * D;
      cout << "path 4";
    }
  }
}

void applyShift(Eigen::MatrixXcd &UH, Eigen::MatrixXcd &Q,		
		int istart, int iend, int nKr, Complex shift, int shift_num) {
  
  //%------------------------------------------------------%
  //| Construct the plane rotation G to zero out the bulge |
  //%------------------------------------------------------%

  std::vector<double> cos(1);
  std::vector<Complex> sin(1);
  std::vector<Complex> r(1);
  
  Complex t;
  Complex cZero(0.0,0.0);  
  Complex f = UH(istart,   istart) - shift;
  Complex g = UH(istart+1, istart);

  cout << " Block start at " << istart << " end at " << iend - 1 << endl;
  
  for(int i = istart; i < iend-1; i++) {
    
    cout << " Applying shift " << i << endl;
    
    zlartg(f, g, cos, sin, r);
    // Sanity check
    cout << " Sanity["<<i<<"] " << cos[0]*cos[0] << " + " <<  abs(sin[0])*abs(sin[0]) << " = " << cos[0]*cos[0] + pow(sin[0].real(), 2) + pow(sin[0].imag(),2) << endl;
    if(i > istart) {
      UH(i,i-1) = r[0];
      UH(i+1,i-1) = cZero;	      
    }

    //%---------------------------------------------%
    //| Apply rotation to the left of H;  H <- G'*H |
    //%---------------------------------------------%
    //do 50
    for(int j = i; j < nKr; j++) {
      t         =  cos[0]       * UH(i,j) + sin[0] * UH(i+1,j);
      UH(i+1,j) = -conj(sin[0]) * UH(i,j) + cos[0] * UH(i+1,j);
      UH(i,j) = t;
    }

    //%---------------------------------------------%
    //| Apply rotation to the right of H;  H <- H*G |
    //%---------------------------------------------%
    //do 60
    for(int j = 0; j<std::min(i+1+2, iend); j++) {
      t         =  cos[0] * UH(j,i) + conj(sin[0]) * UH(j,i+1);
      UH(j,i+1) = -sin[0] * UH(j,i) + cos[0]       * UH(j,i+1);
      UH(j,i) = t;
    }

    //%-----------------------------------------------------%
    //| Accumulate the rotation in the matrix Q;  Q <- Q*G' |
    //%-----------------------------------------------------%
    // do 70
    for(int j = 0; j<std::min(i+1 + shift_num+1, nKr); j++) {
      t        =  cos[0] * Q(j,i) + conj(sin[0]) * Q(j,i+1);
      Q(j,i+1) = -sin[0] * Q(j,i) + cos[0]       * Q(j,i+1);
      Q(j,i) = t;
    }

    if(i < iend-2) {
      f = UH(i+1,i);
      g = UH(i+2,i);    
    }
  }	  
}


void givensQRUpperHess(Eigen::MatrixXcd &UH, Eigen::MatrixXcd &Q, int nKr,
		       int shifts, int shift_num, int step_start, Complex shift){
  
//https://cug.org/5-publications/proceedings_attendee_lists/1997CD/S96PROC/345_349.PDF
  // This code was put in place to deal with the difference between IEEE and CRAY
  // double prec formats. Now that we use IEEE almost exclusivley, we use the IEEE
  // starndards
  double unfl = DBL_MIN;
  double ulp = DBL_EPSILON;
  double smlnum = unfl*(nKr/ulp);
  cout << "Small Num = " << smlnum << endl;
  
  //%----------------------------------------%
  //| Check for splitting and deflation. Use |
  //| a standard test as in the QR algorithm |
  //| REFERENCE: LAPACK subroutine zlahqr    |
  //%----------------------------------------%

  int istart = 0;
  int iend = -1;
  
  cout << "shifts=" << shifts << " step_start=" << step_start << endl;
  /*
  //znapps.f line 281
  // do 30 loop
  for(int i=0; i<nKr-1; i++) {
    
    double tst1 = abs(UH(i,i).real()) + abs(UH(i,i).imag()) + abs(UH(i+1,i+1).real()) + abs(UH(i+1,i+1).imag());
    
    if( tst1 == 0 ) {
      cout << " *** TST1 hit "<< endl; 
      // ARPACK uses zlanhs. Finds the largest column norm
      // tst1 = zlanhs( '1', kplusp-jj+1, h, ldh, workl )
      for(int j=0; j<nKr - shift_num; j++) {
	double sum = 0.0;
	for(int k=0; k<std::min(nKr-shift_num,j+1); k++) sum += abs(UH(k,j));
	tst1 = std::max(tst1,sum);
      }
    }

    cout << "At loop " << i << " istart = " << istart << " iend = " << iend << endl;
    
    bool apply = false;
    cout << "testing UH split at " << i << " " << abs(UH(i+1,i).real()) << " " << std::max(ulp*tst1, smlnum) << endl;
    if (abs(UH(i+1,i).real()) <= std::max(ulp*tst1, smlnum)) {
      iend = i;
      UH(i+1,i) = 0.0;
      if(istart == iend){
	
	//%------------------------------------------------%
	//| No reason to apply a shift to block of order 1 |
	//| or if the current block starts after the point |
	//| of compression since we'll discard this stuff  |
	//%------------------------------------------------%    
	
	cout << " No need for single block rotation at " << i << endl;
	istart = i + 1;
      } else if( istart <= step_start) {
	cout << "Enter Apply shift at " << i << " istart = " << istart << " iend = " << iend << endl;
	applyShift(UH, Q, istart, iend, nKr, shift, shift_num);
	istart = i + 1;
      }      
    }
  }
  */
  // If we finish the i loop with a istart less that step_start, we must
  // do a final set of shifts
  if(istart <= step_start) {
    //perform final block compression
    cout << "Applying final shift " << istart << " " << nKr-2 << endl;
    applyShift(UH, Q, istart, nKr, nKr, shift, shift_num);
  }
  
  //%---------------------------------------------------%
  //| Perform a similarity transformation that makes    |
  //| sure that the compressed H will have non-negative |
  //| real subdiagonal elements.                        |
  //%---------------------------------------------------%

  //do 120
  for(int j=0; j<step_start; j++) {
    if (UH(j+1,j).real() < 0.0 || UH(j+1,j).imag() != 0.0 ) {
      Complex t = UH(j+1,j) / abs(UH(j+1,j));
      for(int i=0; i<nKr-j; i++) UH(j+1,i) *= conj(t);
      //for(int i=0; i<std::min(j+1+2, nKr); i++) UH(i,j+1) *= t;
      //for(int i=0; i<std::min(j+1+shifts+1+1,nKr); i++) Q(i,j+1) *= t;
      for(int i=0; i<std::min(nKr,nKr); i++) UH(i,j+1) *= t;
      for(int i=0; i<std::min(nKr,nKr); i++) Q(i,j+1) *= t;
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
      // ARPACK uses zlanhs. Finds the largest column norm
      // tst1 = zlanhs( '1', kplusp-jj+1, h, ldh, workl )
      for(int j=0; j<step_start; j++) {
	double sum = 0.0;
	for(int k=0; k<std::min(step_start,j+1); k++) sum += abs(UH(k,j));
	tst1 = std::max(tst1,sum);
      }
    }
    if (abs(UH(i+1,i).real()) <= std::max(ulp*tst1, smlnum)) UH(i+1,i) = 0.0;
  }
}


//Orthogonalise r against the j vectors in vectorSpace, populate with res with residua
void measOrthoDev(std::vector<Complex> &s, Complex *r, std::vector<Complex*> vectorSpace, int j) {  
  for(int i=0; i<j+1; i++) s[i] = cDotProd(vectorSpace[i], r);
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
