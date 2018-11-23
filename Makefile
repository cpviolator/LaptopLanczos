#Your path to Eigen
EIGEN=/Users/deanhowarth/lanczos/Eigen

TARGET	 = lanczos

SOURCES  = lanczos.cpp
OBJS     = lanczos.o 
INC      = linAlgHelpers.h
LIBS     = -L/usr/local/opt/libomp/lib -lomp

ERRS=-Wall

CXX = g++
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} 
OMPFLAGS = -Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include
CXXFLAGS+=${OMPFLAGS}

#============================================================

all: $(TARGET)

${TARGET}: ${OBJS}
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

lanczos.o: lanczos.cpp $(INC) 
	${CXX} ${CXXFLAGS} -c lanczos.cpp

ALL_SOURCES = Makefile $(SOURCES) $(INC) 

clean:
	rm -f $(TARGET) $(OBJS) 

tar: $(ALL_SOURCES) $(NOTES)
	tar cvf $(TARGET).tar $(ALL_SOURCES) $(NOTES)
