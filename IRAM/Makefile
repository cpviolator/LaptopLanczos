#Your path to Eigen
EIGEN=../Eigen
INCLUDE=../include

#Your path to ARPACK
ARPACK_LIB='/scratch/CPviolator/ARPACK_CLEAN/libarpack.a'

TARGET	 = iram
SOURCES  = iram.cpp
OBJS     = iram.o 
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h lapack.h arpack_interface.h)
LIBS     = ${ARPACK_LIB}

ERRS=-Wall

CXX = g++
CXXFLAGS = -fPIC -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE} -fcx-fortran-rules -O3

#============================================================

all: $(TARGET)

iram: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o iram $(OBJS) $(LIBS) ${ARPACK_LIB} -lgfortran

iram.o: iram.cpp ${INC}
	${CXX} ${CXXFLAGS} -c iram.cpp

ALL_SOURCES = Makefile $(SOURCES) 

clean:
	rm -f $(TARGET) $(OBJS) 
