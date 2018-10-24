# Laptop Lanczos

You'll need a copy of Eigen:

       http://bitbucket.org/eigen/eigen/get/3.3.5.tar.bz2

Point to it in the Makefile and the lanczos.cpp file, compile with

      make

Run the executable with

    ./lanczos NKV NEV

where NKV is the size of the Kyrlov space, and NEV is the
desired number of eigenpairs wanted.
