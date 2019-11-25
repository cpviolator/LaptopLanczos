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

void zscal(int n, Complex da, Complex *X, int incx) {
  //Scales a vector by a constant
  for(int i=0; i<n; i+=incx) {
    X[i] *= da;
  }  
}

void zdscal(int n, double da, Complex *X, int incx) {
  //Scales a vector by a constant
  for(int i=0; i<n; i+=incx) {
    X[i] *= da;
  }  
}

void zlascl(double cfrom, double cto, Complex *X) {

  //ZLASCL multiplies the M by N complex matrix A by the real scalar  
  //CTO/CFROM.  This is done without over/underflow as long as the final
  //result CTO*A(I,J)/CFROM does not over/underflow.
  
  //Get machine parameters

  double smlnum = DBL_MIN;
  double bignum = 1.0 / smlnum;

  double cfromc = cfrom, ctoc = cto, cfrom1 = 0, cto1 = 0, mul = 0;
  
  bool done = false;
  
  while(!done) {
    cfrom1 = cfromc*smlnum;
    cto1 = ctoc / bignum;
    if(abs( cfrom1 ) > abs( ctoc ) && ctoc != 0.0 ) {
      mul = smlnum;
      done = false;
      cfromc = cfrom1;
    } else if(abs( cto1 ) > abs( cfromc )) {
      mul = bignum;
      done = false;
      ctoc = cto1;
    } else {
      mul = ctoc / cfromc;
      done = true;
    }
    
    for(int j=0; j < Nvec; j++) X[j] *= mul;
  }
}

double dlapy2(double X, double Y) {
      
  //DLAPY2 returns sqrt(x**2+y**2), taking care not to cause unnecessary overflow.
  double XABS = abs( X );
  double YABS = abs( Y );
  double dlapy2;
  double W = std::max(XABS, YABS);
  double Z = std::min(XABS, YABS);
  if( Z == 0.0 ) {
    dlapy2 = W;
  } else {
    dlapy2 = W*sqrt( 1.0 + (Z/W)*(Z/W) );
  }
  return dlapy2;
}

double dlapy3(Complex X, Complex Y, Complex Z) {

  //DLAPY3 returns sqrt(x**2+y**2+z**2), taking care not to cause unnecessary overflow.
  double XABS = abs( X );
  double YABS = abs( Y );
  double ZABS = abs( Z );
  double dlapy3;
  double W = std::max(std::max(XABS, YABS), ZABS);
  if( W == 0.0 ) {
    dlapy3 = 0.0;
  } else {
    dlapy3 = W*sqrt( (XABS/W)*(XABS/W) + (YABS/W)*(YABS/W) + (ZABS/W)*(ZABS/W) );
  }
  return dlapy3;
}
  
double dznrm2(int N, Complex *X, int INCX) {
  
  //DZNRM2 returns the euclidean norm of a vector via the function
  //name, so that

  //DZNRM2 := sqrt( conjg( x' )*x )

  double NORM, SCALE, SSQ, TEMP;
    
  if( N < 1 || INCX < 1 ) {
    NORM = 0.0;
  } else {
    SCALE = 0.0;
    SSQ   = 1.0;
    
    //The following loop is equivalent to this call to the LAPACK
    //auxiliary routine:
    //CALL ZLASSQ( N, X, INCX, SCALE, SSQ )
    
    for(int i=0; i < 1 + ( N - 1 )*INCX; i += INCX) {
      if( X[i].real() != 0.0 ) {
	TEMP = abs(X[i].real());
	if(SCALE < TEMP) {	  
	  SSQ = 1.0 + SSQ*((SCALE*SCALE)/(TEMP*TEMP));
	  SCALE = TEMP;
	} else {
	  SSQ = SSQ + ((TEMP*TEMP)/(SCALE*SCALE));
	}
      }
      if(X[i].imag() != 0.0) {	
	TEMP = abs(X[i].imag());
	if(SCALE < TEMP){	  
	  SSQ = 1.0 + SSQ*((SCALE*SCALE)/(TEMP*TEMP));
	  SCALE = TEMP;
	} else {
	  SSQ = SSQ + ((TEMP*TEMP)/(SCALE*SCALE));
	}
      }
    }
    NORM = SCALE * sqrt( SSQ );
  }
  return NORM;
}
  
double zlanhs(Eigen::MatrixXcd A, int n) {
  
  //Find norm1(A)
  double value = 0.0;
  
  for(int j = 1; j<n; j++) {
    double sum = 0.0;
    for (int i = 0; i < std::min(n, j+1); i++) {
      sum += abs(A(i,j));
    }
    value = std::max(value, sum);
  }
  return value;
}

double cabs1(Complex C) {
  return abs(C.real()) + abs(C.imag());
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
    //cout << "path 1";
  } else if (F == cZero) {
    cos[0] = 0.0;
    sin[0] = conj(G)/abs(G);
    r[0] = abs(G);
    //cout << "path 2";
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
      //cout << "path 3";
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
      //cout << "path 4";
    }
  }
}

/*
void zlarfg(int N, Complex &ALPHA, Complex *X, int INCX, Complex &TAU) {
  
  
    ZLARFG generates a complex elementary reflector H of order n, such
    that
    
    H' * ( alpha ) = ( beta ),   H' * H = I.
         (   x   )   (   0  )
	 
    where alpha and beta are scalars, with beta real, and x is an
    (n-1)-element complex vector. H is represented in the form
	 
    H = I - tau * ( 1 ) * ( 1 v' ) ,
                  ( v )
		  
    where tau is a complex scalar and v is a complex (n-1)-element
    vector. Note that H is not hermitian.

    If the elements of x are all zero and alpha is real, then tau = 0
    and H is taken to be the unit matrix.
    
    Otherwise  1 <= real(tau) <= 2  and  abs(tau-1) <= 1 .
    
    Arguments
    =========
    
    N       (input) INTEGER
    The order of the elementary reflector.
    
    ALPHA   (input/output) COMPLEX*16
    On entry, the value alpha.
    On exit, it is overwritten with the value beta.*

    X       (input/output) COMPLEX*16 array, dimension
    (1+(N-2)*abs(INCX))
    On entry, the vector x.
    On exit, it is overwritten with the vector v.
    
    INCX    (input) INTEGER
    The increment between elements of X. INCX > 0.
    
    TAU     (output) COMPLEX*16
    The value tau.
    
    =====================================================================
    
  int KNT;
  double ALPHI, ALPHR, BETA, RSAFMN, SAFMIN, XNORM;

  Complex cOne(1.0,0.0);
  
  if( N <= 0 ) {
    TAU = 0.0;
    return;
  }

  XNORM = dznrm2(N-1, X, INCX);
  ALPHR = ALPHA.real();
  ALPHI = ALPHA.imag();
  
  if(XNORM == 0.0 && ALPHI == 0.0 ) {
    //H  =  I
    TAU = 0.0;
  } else {
    //general case
    int sign1 = (ALPHR > 0) - (ALPHR < 0);
    BETA = -sign1 * dlapy3(ALPHR, ALPHI, XNORM);
    SAFMIN = DBL_MIN / DBL_EPSILON;
    RSAFMN = 1.0 / SAFMIN;

    if(abs( BETA ) < SAFMIN ) {
      //XNORM, BETA may be inaccurate; scale X and recompute them      
      KNT = 0;
      while (abs( BETA ) < SAFMIN) {
	KNT = KNT + 1;
	zdscal(N-1, RSAFMN, X, INCX);
	BETA = BETA*RSAFMN;
	ALPHI = ALPHI*RSAFMN;
	ALPHR = ALPHR*RSAFMN;
      }
      
      //New BETA is at most 1, at least SAFMIN
      
      XNORM = dznrm2( N-1, X, INCX );
      ALPHA.real(ALPHR); ALPHA.imag(ALPHI);
      int sign2 = (ALPHR > 0) - (ALPHR < 0);
      BETA = -sign2 * dlapy3( ALPHR, ALPHI, XNORM );
      TAU.real( (BETA-ALPHR ) / BETA); TAU.imag(-ALPHI / BETA);
      ALPHA = 1.0 / (ALPHA-BETA);
      zscal( N-1, ALPHA, X, INCX );

      //If ALPHA is subnormal, it may lose relative accuracy
  
      ALPHA = BETA;
      for(int j = 1; j < KNT; j++) {
	ALPHA = ALPHA*SAFMIN;
      }
    } else {
      TAU.real(( BETA-ALPHR ) / BETA); TAU.imag( -ALPHI / BETA );
      ALPHA = cOne / (ALPHA-BETA);
      zscal( N-1, ALPHA, X, INCX );
      ALPHA = BETA;
    }
  }
}


//SUBROUTINE ZLAHQR( WANTT, WANTZ, N, ILO, IHI, H, LDH, W, ILOZ, IHIZ, Z, LDZ, INFO )
//zlahqr(true, true, n, 0, n, upperHessTemp, ldh, ritz, 0, n, Q, ldq);
void zlahqr(bool WANTT, bool WANTZ, int N, int ILO, int IHI,
	    Eigen::MatrixXcd &UH, int LDH, Complex *ritz,
	    int ILOZ, int IHIZ, Eigen::MatrixXcd &Q, int IDQ) {
  
  Complex cZero(0.0,0.0);
  Complex cOne(1.0,0.0);

  int I1, I2, ITN, ITS, j, k, l, m, NH, NZ;
  double H10, H21, RTEMP, S, SMLNUM, T2, TST1, ULP, UNFL;
  Complex CDUM, H11, H11S, H22, SUM, T, T1, TEMP, U, V2, X, Y;

  std::vector<double> rwork(1,0.0);
  std::vector<Complex> V(2,cZero);
  
  //Quick return if possible

  if( N == 0 ) return;
  if( ILO == IHI ) {
    ritz[ILO] = UH(ILO, ILO);
    return;
  }
  NH = IHI - ILO;   //Index change NH = IHI - ILO + 1 
  NZ = IHIZ - ILOZ; //Index change NZ = IHIZ - ILOZ + 1

  //Set machine-dependent constants for the stopping criterion.
  //If norm(H) <= sqrt(OVFL), overflow should not occur.  
  ULP = DBL_EPSILON;
  UNFL = DBL_MIN;
  SMLNUM = UNFL / ULP;

  //I1 and I2 are the indices of the first row and last column of H
  //to which transformations must be applied. If eigenvalues only are
  //being computed, I1 and I2 are set inside the main loop.  
  if( WANTT ) {
    I1 = 0; //Index change I1 = 1
    I2 = N;
  }
  
  //ITN is the total number of QR iterations allowed.  
  ITN = 30*NH;
  
  //The main loop begins here. I is the loop index and decreases from
  //IHI to ILO in steps of 1. Each iteration of the loop works
  //with the active submatrix in rows and columns L to I.
  //Eigenvalues I+1 to IHI have already converged. Either L = ILO, or
  //H(L,L-1) is negligible so that the matrix splits.
  int i = IHI-1; //Index change I = IHI
                 // Careful with this one. A matrix in f77 with
                 // M(IHI,IHI) is valid. 
  //10 continue
  cout << "ILO = " << ILO << " IHI = " << IHI << " i = " << i << endl;
  while(i >= ILO) {
    
    //for(int w=0; w<LDH; w++) cout << "zlahqr ritz = " << ritz[i] << endl;
    
    //Perform QR iterations on rows and columns ILO to I until a
    //submatrix of order 1 splits off at the bottom because a
    //subdiagonal element has become negligible.    
    l = ILO;
    // do 110
    bool lgei = false;
    for(ITS = 0; ITS < ITN && !lgei; ITS++) {

      cout << "Start iteration " << ITS << endl;
      
      //Look for a single small subdiagonal element.
      for(k = i; k >= l + 1; k--) {
	TST1 = cabs1( UH(k-1, k-1) ) + cabs1( UH(k, k));
	if(TST1 == 0.0) {
	  cout << " *** TST1 hit "<< endl; 
	  TST1 = zlanhs(UH, LDH);
	}
	if(abs(UH(k, k-1).real()) <= std::max(ULP*TST1, SMLNUM)) {
	  cout << " goto 30 hit "<< endl; 
	  continue;
	}
      }

      l = k;
      cout << "ONE k="<<k<<" m="<<m<<" l="<<l<<" i="<<i<<endl;
      if (l > ILO) {
	//UH(L,L-1) is negligible
	UH(l, l-1) = cZero;
      }

      //Exit from loop if a submatrix of order 1 has split off.      
      if( l >= i ) {
	cout << "SM split at l=" << l << " i=" << i; 
	//UH(I,I-1) is negligible: one eigenvalue has converged.	
	ritz[i] = UH(i,i);
	  
	//Decrement number of remaining iterations, and return to start of
	//the main loop with new value of I.	  
	ITN = ITN - ITS;
	i = l - 1;
	// return to start of 10 continue 
	lgei = true;
      }
      
      //cout << "l="<<l<<" k="<<k<<endl;
      //Now the active submatrix is in rows and columns L to I. If
      //eigenvalues only are being computed, only the active submatrix
      //need be transformed.
      if(!WANTT) {
	I1 = l;
	I2 = i;
      }      
      if(ITS == 9 || ITS == 19 ) {
	//Exceptional shift.	
	T = abs(UH(i,i-1).real()) + abs(UH(i-1,i-2).real());
      } else {
	//Wilkinson's shift.	
	T = UH(i,i);
	U = UH(i-1,i) * UH(i,i-1).real();
	if(U != cZero) {	  
	  X = 0.5*(UH(i-1,i-1) - T);
	  Y = sqrt(X*X+U);
	  if( (X.real() * Y.real() + X.imag() * Y.imag()) < 0.0 ) Y = -Y;
	  T -= U/( X+Y );
	}
      }

      //Look for two consecutive small subdiagonal elements.
      bool goto50 = false;
      for(m = i - 1; m >= l + 1 && !goto50; m--) {

	cout << "TWO k="<<k<<" m="<<m<<" l="<<l<<" i="<<i<<endl;
	
	//Determine the effect of starting the single-shift QR
	//iteration at row M, and see if this would make H(M,M-1)
	//negligible.

	H11 = UH(m, m);
	H22 = UH(m+1, m+1);
	H11S = H11 - T;
	//H21 = UH( M+1, M );
	H21 = UH(m+1, m).real(); //DMH guess this is what f77 means?? 
	S = cabs1( H11S ) + abs( H21 );
	H11S = H11S / S;
	H21 = H21 / S;
	V[0] = H11S;
	V[1] = H21;
	H10 = UH(m, m-1).real();//DMH guess this is what f77 means?? 
	TST1 = cabs1( H11S )*( cabs1( H11 ) + cabs1( H22 ) );
	if(abs( H10*H21 ) <= ULP*TST1 ) {
	  //cout << "goto50 hit" << endl;
	  goto50 = true;
	}
      }
      
      if(!goto50) {
	//cout << "!goto50 hit" << endl;
	//cout << "m="<<m<<" i="<<i<<" l="<<l<<endl;
	H11 = UH(l, l);
	H22 = UH(l+1, l+1);
	H11S = H11 - T;
	H21 = UH(l+1, l).real();//DMH guess this is what f77 means??
	S = cabs1( H11S ) + abs( H21 );
	H11S = H11S / S;
	H21 = H21 / S;
	V[0] = H11S;
	V[1] = H21;
      }

      cout << "THREE k="<<k<<" m="<<m<<" l="<<l<<" i="<<i<<endl;
      
      //Single-shift QR step
      for(k = m; k < i; k++) {

	cout << "FOUR k="<<k<<" m="<<m<<" l="<<l<<" i="<<i<<endl;
	
	//The first iteration of this loop determines a reflection G
	//from the vector V and applies it from left and right to H,
	//thus creating a nonzero bulge below the subdiagonal.
	
	//Each subsequent iteration determines a reflection G to
	//restore the Hessenberg form in the (K-1)th column, and thus
	//chases the bulge one step toward the bottom of the active
	//submatrix.
	
	//V[1] is always real before the call to ZLARFG, and hence
	//after the call T2 ( = T1*V[1] ) is also real.
	
	if(k > m) {
	  //CALL ZCOPY( 2, H( K, K-1 ), 1, V, 1 )
	  V[0] = UH(k, k-1);
	  V[1] = UH(k+1, k-1);
	}
	
	zlarfg(2, V[0], &V[1], 1, T1);
	if(k > m) {	    
	  UH(k, k-1) = V[0];
	  UH(k+1, k-1) = cZero;
	}
	V2 = V[1];
	Complex tempT2 = (T1*V2);
	T2 = tempT2.real();
	
	//Apply G from the left to transform the rows of the matrix
	//in columns K to I2.
	
	for (int j = k; j < I2; j++) {
	  SUM = conj(T1) * UH(k,j) + T2*UH(k+1,j);
	  UH(k,j) -= SUM;
	  UH(k+1,j) -= SUM*V2;
	}
	
	//Apply G from the right to transform the columns of the
	//matrix in rows I1 to min(K+2,I).
	
	for( int j = I1; j < std::min(k+2+1, i); j++) {
	  SUM = T1 * UH(j,k) + T2 * UH(j,k+1);
	  UH(j,k) -= SUM;
	    UH(j,k+1) -= SUM*conj(V2);
	}
	
	if( WANTZ ) {	    
	  //Accumulate transformations in the matrix Z	    
	  for (int j = ILOZ; j < IHIZ; j++) {
	    SUM = T1 * Q(j,k) + T2 * Q(j,k+1);
	    Q(j,k) -= SUM;
	    Q(j,k+1) -= SUM * conj(V2);
	  }
	}
	
	if( k == m && m > l ) {
	  //If the QR step was started at row M > L because two
	  //consecutive small subdiagonals were found, then extra
	  //scaling must be performed to ensure that H(M,M-1) remains
	  //real.
	  
	  TEMP = cOne - T1;
	  TEMP /= abs( TEMP );
	  UH(m+1, m) *= conj( TEMP );
	  if( m+2 <= i ) {
	    UH(m+1, m+1) *= TEMP;
	  }
	  for(int j = m; j < i; j++) { 
	    if( j != m+1 ) {
	      //if( I2 > j ) zscal( I2-j, TEMP, UH(j,j+1).data(), LDH );
	      if( I2 > j ) zscal( I2-j, TEMP, (UH.data() + j*LDH + j+1), LDH );
	      zscal( j-I1, conj( TEMP ), (UH.data() + I1*LDH + j), 1 );
	      if( WANTZ ) {
		zscal( NZ, conj( TEMP ), (Q.data() + ILOZ*LDH + j), 1 );
	      }
	    }
	  }
	}
      }

      cout << "FIVE k="<<k<<" m="<<m<<" l="<<l<<" i="<<i<<endl;
      //Ensure that UH(I,I-1) is real.      
      TEMP = UH(i, i-1);
      if( TEMP.imag() != 0.0 ) {
	RTEMP = abs( TEMP );
	UH(i, i-1) = RTEMP;
	TEMP /= RTEMP;
	if( I2 > i ) zscal( I2-i, conj( TEMP ), (UH.data() + i*LDH+ i+1), LDH );
	zscal( i-I1, TEMP, (UH.data() + I1*LDH + i), 1 );
	if( WANTZ ) zscal( NZ, TEMP, (Q.data() + ILOZ*LDH + i), 1 );
      }
    }
  }
}

void zneigh(double rnorm, int n, Eigen::MatrixXcd &UH, int ldh, Complex *ritz,
	    double *bounds, Eigen::MatrixXcd &Q, int ldq) {
  
  //     %----------------------------------------------------------%
  //     | 1. Compute the eigenvalues, the last components of the   |
  //     |    corresponding Schur vectors and the full Schur form T |
  //     |    of the current upper Hessenberg matrix H.             |
  //     |    zlahqr returns the full Schur form of H               | 
  //     |    in UHshur and the Schur vectors in Q.                 |
  //     %----------------------------------------------------------%

  // call zlacpy ('All', n, n, h, ldh, workl, n)
  // Copy UH to temp space
  MatrixXcd upperHessTemp = MatrixXcd::Zero(ldh, ldh);  
  upperHessTemp = UH;

  // call zlaset ('All', n, n, zero, one, q, ldq)
  // Initialise Q to the identity
  // Already done in main program.

  cout << "calling zlahqr " << endl; 
  //call zlahqr (.true., .true., n, 1, n, workl, ldh, ritz, 1, n, q, ldq, ierr)
  zlahqr(true, true, n, 0, n, upperHessTemp, ldh, ritz, 0, n, Q, ldq);
  cout << "called zlahqr " << endl; 
  //zcopy...
  
  //call ztrevc ('Right', 'Back', select, n, workl, n, vl, n, q, ldq, n, n, workl(n*n+1), rwork, ierr)
  
}
*/

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
    
    cout << " Apply shift " << i << " = " << shift << endl;
    
    f.imag(0.0);
    g.imag(0.0);
    cout << "f= " << f << endl;
    cout << "g= " << g << endl;
    zlartg(f, g, cos, sin, r);
    // Sanity check
    //cout << " Sanity["<<i<<"] " << cos[0]*cos[0] << " + " <<  abs(sin[0])*abs(sin[0]) << " = " << cos[0]*cos[0] + pow(sin[0].real(), 2) + pow(sin[0].imag(),2) << endl;
    //r[0].imag(0.0);
    //sin[0].imag(0.0);
    cout << "r= " << r[0] << endl;
    cout << "c= " << cos[0] << endl;
    cout << "s= " << sin[0] << endl;
    if(i > istart) {
      UH(i,i-1) = r[0];
      UH(i+1,i-1) = cZero;	      
    }

    //%---------------------------------------------%
    //| Apply rotation to the left of H;  H <- G'*H |
    //%---------------------------------------------%
    //do 50
    for(int j = i; j < nKr; j++) {
      cout<<"pre h("<<i+1<<","<<j<<")="<<UH(i+1,j)<<endl;
      cout<<"pre h("<<i<<","<<j<<")="<< UH(i,j)<<endl;      
      t         =  cos[0]       * UH(i,j) + sin[0] * UH(i+1,j);
      UH(i+1,j) = -conj(sin[0]) * UH(i,j) + cos[0] * UH(i+1,j);
      UH(i,j) = t;
      cout<<"post h("<<i+1<<","<<j<<")="<<UH(i+1,j)<<endl;
      cout<<"post h("<<i<<","<<j<<")="<< UH(i,j)<<endl;
    }

    //%---------------------------------------------%
    //| Apply rotation to the right of H;  H <- H*G |
    //%---------------------------------------------%
    //do 60
    cout << "min60 = " << std::min(i+1+2, iend) << endl;
    for(int j = 0; j<std::min(i+1+2, iend); j++) {
      cout<<"pre h("<<j<<","<<i+1<<")="<< UH(j,i+1)<<endl;
      cout<<"pre h("<<j<<","<<i<<")="<< UH(j,i)<<endl;
      t         =  cos[0] * UH(j,i) + conj(sin[0]) * UH(j,i+1);
      UH(j,i+1) = -sin[0] * UH(j,i) + cos[0]       * UH(j,i+1);
      UH(j,i) = t;
      cout<<"post h("<<j<<","<<i+1<<")="<< UH(j,i+1)<<endl;
      cout<<"post h("<<j<<","<<i<<")="<< UH(j,i)<<endl;
    }

    //%-----------------------------------------------------%
    //| Accumulate the rotation in the matrix Q;  Q <- Q*G' |
    //%-----------------------------------------------------%
    // do 70
    cout << "min70 = " << std::min(i+1 + shift_num+1, nKr) << endl;
    for(int j = 0; j<std::min(i+1 + shift_num, nKr); j++) {
      cout<<"pre q("<<j<<","<<i+1<<")="<< Q(j,i+1)<<endl;
      cout<<"pre q("<<j<<","<<i<<")="<< Q(j,i)<<endl;
      cout<<"cos=" << cos[0] << " sin = " << sin[0] << " conj(sin) = " << conj(sin[0]) <<endl;
      t        =  cos[0] * Q(j,i) + conj(sin[0]) * Q(j,i+1);
      Q(j,i+1) = -sin[0] * Q(j,i) + cos[0]       * Q(j,i+1);
      Q(j,i) = t;
      cout<<"post q("<<j<<","<<i+1<<")="<< Q(j,i+1)<<endl;
      cout<<"post q("<<j<<","<<i<<")="<< Q(j,i)<<endl;
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
  double smlnum = unfl*(Nvec/ulp);
  cout << "Small Num = " << smlnum << endl;
  
  //%----------------------------------------%
  //| Check for splitting and deflation. Use |
  //| a standard test as in the QR algorithm |
  //| REFERENCE: LAPACK subroutine zlahqr    |
  //%----------------------------------------%

  int istart = 0;
  int iend = -1;
  
  //cout << "shifts=" << shifts << " step_start=" << step_start << endl;
  cout << "Shift " << shift_num << " = " << shift << endl;

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
    
    if (abs(UH(i+1,i).real()) <= std::max(100*ulp*tst1, smlnum)) {
      cout << "UH split at " << i << " shift = " << shift_num << endl;
      cout << abs(UH(i+1,i).real()) << " " << std::max(ulp*tst1, smlnum) << endl;
      iend = i;
      UH(i+1,i) = 0.0;
      if(istart == iend){
	
	//%------------------------------------------------%
	//| No reason to apply a shift to block of order 1 |
	//| or if the current block starts after the point |
	//| of compression since we'll discard this stuff  |
	//%------------------------------------------------%    
	
	cout << " No need for single block rotation at " << i << endl;
	istart = iend + 1;
      } else if (istart > step_start) {
	cout << " block rotation beyond " << step_start << endl;
	istart = iend + 1;
      } else if( istart <= step_start) {
	cout << "Enter Apply shift at " << i << " istart = " << istart << " iend = " << iend << endl;
	applyShift(UH, Q, istart, iend, nKr, shift, shift_num);
	istart = iend + 1;	
      }
    }
  }

  iend = nKr;
  cout << "At End istart = " << istart << " iend = " << iend << endl;
  
  // If we finish the i loop with a istart less that step_start, we must
  // do a final set of shifts
  if(istart <= step_start) {
    //perform final block compression
    //cout << "Applying final shift " << istart << " " << nKr-2 << endl;
    applyShift(UH, Q, istart, iend, nKr, shift, shift_num);
  } else if(istart == iend-1){
    cout << " No need for single block rotation at " << istart << endl;
  }
  
  //%---------------------------------------------------%
  //| Perform a similarity transformation that makes    |
  //| sure that the compressed H will have non-negative |
  //| real subdiagonal elements.                        |
  //%---------------------------------------------------%
  
  if( shift_num == shifts-1 ) {
    //do 120
    cout << "KEV = " << step_start << endl;
    for(int j=0; j<step_start; j++) {
      if (UH(j+1,j).real() < 0.0 || UH(j+1,j).imag() != 0.0 ) {
	Complex t = UH(j+1,j) / dlapy2(UH(j+1,j).real(), UH(j+1,j).imag());	
	for(int i=0; i<nKr-j; i++) UH(j+1,i) *= conj(t);
	cout << "max1 = " << nKr-j << endl;
	for(int i=0; i<std::min(j+1+2, nKr); i++) UH(i,j+1) *= t;
	cout << "min2 = " << std::min(j+1+2, nKr) << endl;
	for(int i=0; i<std::min(j+1+shifts+1,nKr); i++) Q(i,j+1) *= t;
	cout << "min3 = " << std::min(j+1+shifts+1,nKr) << endl;
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
      if (abs(UH(i+1,i).real()) <= std::max(ulp*tst1, smlnum)) {
	UH(i+1,i) = 0.0;
        cout << "Zero out UH("<<i+1<<","<<i<<") by hand"<<endl;;
      }
    }
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
