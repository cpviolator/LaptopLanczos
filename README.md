# Laptop Lanczos

You'll need a copy of Eigen. To get it execute

      wget http://bitbucket.org/eigen/eigen/get/3.3.5.tar.bz2

Point to it in the Makefile and the lanczos.cpp file, compile with

      make

Run the executable with

     ./lanczos NKV NEV CHECK OFFSET TOL THREADS

where:

NKV is the size of the Kyrlov space.

NEV is the desired number of eigenpairs.

CHECK is the frequency with which to check for convergence.

OFFSET is a constant added to the diagonal of the problem matrix to ensure positive eigenvalues.

TOL is the tolernace.

THREADS is the number of OMP threads

To change the problem size,
edit line 13

     #define Nvec 1024
