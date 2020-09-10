#Your path to Eigen
EIGEN=../Eigen
INCLUDE=../include

TARGET	 = trlm
SOURCES  = trlm.cpp 
OBJS     = trlm.o 
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h lapack.h)

ERRS=-Wall

CXX = g++
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE}

#============================================================

all: $(TARGET)

trlm: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o trlm $(OBJS)

trlm.o: trlm.cpp ${INC}
	${CXX} ${CXXFLAGS} -c trlm.cpp

ALL_SOURCES = Makefile $(SOURCES) 

clean:
	rm -f $(TARGET) $(OBJS) 
