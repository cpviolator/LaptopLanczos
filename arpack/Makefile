#Your path to Eigen
EIGEN=/Users/deanhowarth/LaptopLanczos/Eigen
INCLUDE=/Users/deanhowarth/LaptopLanczos/include

#Your path to ARPACK
#ARPACK_LIB='/Users/deanhowarth/LaptopLanczos/ARPACK/ARPACK/libarpack.a'
#ARPACK_LIB='/Users/deanhowarth/AdS2-Lattice/arpack-ng/SRC/.libs/libarpack.a'
ARPACK_LIB='/usr/local/Cellar/arpack/3.7.0_3/lib/libarpack.a'
#GFORTLIBS='/usr/local/Cellar/gcc/9.2.0_1/lib/gcc/9/libgfortran.a  /usr/local/Cellar/gcc/9.2.0_1/lib/gcc/9/libquadmath.a'

TARGET	 = arpack
SOURCES  = arpack_interface.cpp
OBJS     = arpack_interface.o 
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h)
LIBS     = ${ARPACK_LIB}

ERRS=-Wall

CXX = g++
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE} -I/usr/local/include 

#============================================================

all: $(TARGET)

arpack: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o arpack ${OBJS} ${ARPACK_LIB} 

arpack_interface.o: arpack_interface.cpp ${INC}
	${CXX} ${CXXFLAGS} -c arpack_interface.cpp

ALL_SOURCES = Makefile $(SOURCES) 

clean:
	rm -f $(TARGET) $(OBJS) 
