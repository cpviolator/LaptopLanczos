#Your path to Eigen
EIGEN=../Eigen
INCLUDE=../include

TARGET	 = arnoldi 
SOURCES  = arnoldi.cpp 
OBJS     = arnoldi.o
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h)

ERRS=-Wall

CXX = g++
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE}
#============================================================

all: $(TARGET)

arnoldi: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o arnoldi $(OBJS) 

arnoldi.o: arnoldi.cpp ${INC}
	${CXX} ${CXXFLAGS} -c arnoldi.cpp

ALL_SOURCES = Makefile $(SOURCES) 

clean:
	rm -f $(TARGET) $(OBJS) 

tar: $(ALL_SOURCES) $(NOTES)
	tar cvf $(TARGET).tar $(ALL_SOURCES) $(NOTES)
