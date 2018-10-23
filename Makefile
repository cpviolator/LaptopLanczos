#Your path to Eigen
EIGEN=/Users/deanhowarth/lanczoz/Eigen/Eigen

TARGET	 = lanczos

SOURCES  = lanczos.cpp
OBJS     = lanczos.o 

ERRS=-Wall

CXX = g++
CXXFLAGS = -O3 -g ${ERRS} -std=c++11

#============================================================

all: $(TARGET)

${TARGET}: ${OBJS}
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

lanczos.o: lanczos.cpp $(INCLUDES) 
	${CXX} ${CXXFLAGS} -c lanczos.cpp

ALL_SOURCES = Makefile $(SOURCES) $(INCLUDES) 

clean:
	rm -f $(TARGET) $(OBJS) 

tar: $(ALL_SOURCES) $(NOTES)
	tar cvf $(TARGET).tar $(ALL_SOURCES) $(NOTES)
