#Your path to Eigen
EIGEN=/Users/deanhowarth/LaptopLanczos/Eigen
INCLUDE=/Users/deanhowarth/LaptopLanczos/include

TARGET	 = irlm
SOURCES  = irlm.cpp
OBJS     = irlm.o 
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h)
LIBS     = -L/usr/local/opt/libomp/lib -lomp

ERRS=-Wall

CXX = g++
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE}
#OMPFLAGS = -Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include
CXXFLAGS+=${OMPFLAGS}

#============================================================

all: $(TARGET)

irlm: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o irlm $(OBJS) $(LIBS)

irlm.o: irlm.cpp ${INC}
	${CXX} ${CXXFLAGS} -c irlm.cpp

ALL_SOURCES = Makefile $(SOURCES) 

clean:
	rm -f $(TARGET) $(OBJS) 
