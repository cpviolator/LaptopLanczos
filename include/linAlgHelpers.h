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

#endif

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
