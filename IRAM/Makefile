#Your path to Eigen
EIGEN=../Eigen
INCLUDE=../include

TARGET	 = iram
SOURCES  = iram.cpp
OBJS     = iram.o 
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h lapack.h)

ERRS=-Wall

CXX = g++
CXXFLAGS = -fPIC -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE} 

#============================================================

all: $(TARGET)

iram: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o iram $(OBJS) $(LIBS)

iram.o: iram.cpp ${INC}
	${CXX} ${CXXFLAGS} -c iram.cpp

ALL_SOURCES = Makefile $(SOURCES) 

clean:
	rm -f $(TARGET) $(OBJS) 
