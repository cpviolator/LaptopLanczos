#Your path to Eigen
EIGEN=/Users/deanhowarth/lanczos/Eigen
INCLUDE=/Users/deanhowarth/lanczos/include

TARGET	 = irm_lanczos
SOURCES  = irm_lanczos.cpp 
OBJS     = irm_lanczos.o 
INC      = $(addprefix ${INCLUDE}/, linAlgHelpers.h algoHelpers.h)
LIBS     = -L/usr/local/opt/libomp/lib -lomp

ERRS=-Wall

CXX = g++
CXXFLAGS = -O3 -g ${ERRS} -std=c++11 -I${EIGEN} -I${INCLUDE}
OMPFLAGS = -Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include
CXXFLAGS+=${OMPFLAGS}

#============================================================

all: $(TARGET)

irm_lanczos: ${OBJS} 
	$(CXX) $(CXXFLAGS) -o irm_lanczos $(OBJS) $(LIBS)

irm_lanczos.o: irm_lanczos.cpp ${INC}
	${CXX} ${CXXFLAGS} -c irm_lanczos.cpp

ALL_SOURCES = Makefile $(SOURCES) 

clean:
	rm -f $(TARGET) $(OBJS) 

tar: $(ALL_SOURCES) $(NOTES)
	tar cvf $(TARGET).tar $(ALL_SOURCES) $(NOTES)
