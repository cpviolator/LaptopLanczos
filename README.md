# Laptop Lanczos

You'll need a copy of Eigen. To get it execute

       wget http://bitbucket.org/eigen/eigen/get/3.3.5.tar.bz2

Point to it in the Makefile and the lanczos.cpp file, compile with

      make

Run the executable with

     ./lanczos NKV NEV OFFSET

where NKV is the size of the Kyrlov space, and NEV is the
desired number of eigenpairs wanted. OFFSET is a constant added
to the diagonal of the problem matrix to ensure positive
eigenvalues.

To change the problem size,
edit line 18

     #define Nvec 512



     
