CXX = nvcc
OBJS = ex1.o
EXE  = ex1
FILES = ex1.cu
CPPFLAGS = -g -lineinfo -Xcompiler -rdynamic -std=c++11 -arch=sm_75 

$(EXE): $(OBJS)
	$(CXX) $(CPPFLAGS) -o $(EXE) $(OBJS)

%.o: %.cu
	$(CXX) $(CPPFLAGS) -c $<

clean:
	rm -rf $(EXE) *.o *~ *.dSYM
