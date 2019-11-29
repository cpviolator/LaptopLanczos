#Your path to Eigen
EIGEN=../Eigen
INCLUDE=../include

TARGET	 = lanczos 
SOURCES  = lanczos.cpp 
OBJS     = lanczos.o
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h)

ERRS=-Wall

CXX = g++
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE}
#============================================================

all: $(TARGET)

lanczos: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o lanczos $(OBJS) $(LIBS)

lanczos.o: lanczos.cpp ${INC}
	${CXX} ${CXXFLAGS} -c lanczos.cpp

ALL_SOURCES = Makefile $(SOURCES) 

clean:
	rm -f $(TARGET) $(OBJS) 

tar: $(ALL_SOURCES) $(NOTES)
	tar cvf $(TARGET).tar $(ALL_SOURCES) $(NOTES)
