CXX = nvcc
OBJS = ex2.o
EXE  = ex2
FILES = ex2.cu
CPPFLAGS = -g -lineinfo -Xcompiler -rdynamic -std=c++11 -arch=sm_75 

$(EXE): $(OBJS)
	$(CXX) $(CPPFLAGS) -o $(EXE) $(OBJS)

%.o: %.cu
	$(CXX) $(CPPFLAGS) -c $<

clean:
	rm -rf $(EXE) *.o *~ *.dSYM
