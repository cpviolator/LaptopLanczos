#Your path to Eigen
EIGEN=../Eigen
INCLUDE=../include

TARGET	 = irlm
SOURCES  = irlm.cpp
OBJS     = irlm.o 
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h)

ERRS=-Wall

CXX = g++
CXXFLAGS = -fPIC -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE}

#============================================================

all: $(TARGET)

irlm: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o irlm $(OBJS) $(LIBS)

irlm.o: irlm.cpp ${INC}
	${CXX} ${CXXFLAGS} -c irlm.cpp

ALL_SOURCES = Makefile $(SOURCES) 

clean:
	rm -f $(TARGET) $(OBJS) 
