CXX = nvcc
OBJS = ex3.o
EXE  = ex3
FILES = ex3.cu
CPPFLAGS = -g -lineinfo -Xcompiler -rdynamic -std=c++11 -arch=sm_75 
LIBS = -lcublas_static -lcublasLt_static -lculibos -lcusparse_static 

all: $(EXE) 

$(EXE): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

%.o: %.cu
	$(CXX) $(CPPFLAGS) -c $<

clean:
	rm -rf $(EXE) *.o *~ *.dSYM
