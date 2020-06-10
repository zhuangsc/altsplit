CC=			mpicc
CFLAGS=		-Wall -Wextra -O3 
CFLAGS_LIB=	#-ansi -pedantic -Wno-long-long # ANSI C does not have inline which affects performance a little bit
CPPFLAGS=	-DHAVE_CBLAS #-D__VERIFY__
INCLUDES=   -I.  -DMKL_ILP64 -m64 -I${MKLROOT}/include
EXE=        examples/mlp-mpi examples/mlp examples/mnist-cnn examples/mlp-mpi-allsplit examples/mnist-cnn-allsplit \
			examples/vgg examples/recommender-mpi
LIBS=		-lz -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

ifdef CBLAS
	CPPFLAGS+=-DHAVE_CBLAS
	INCLUDES+=-I$(CBLAS)/include
	LIBS=-fopenmp -pthread -L$(CBLAS)/lib -lopenblas -lcblas -lz -lm
endif

.SUFFIXES:.c .o
.PHONY:all clean depend

.c.o:
		$(CC) -c $(CFLAGS) $(INCLUDES) $(CPPFLAGS) $< -o $@

all:kautodiff.o kann.o kann_extra/kann_data.o $(EXE)

kautodiff.o:kautodiff.c
		$(CC) -c $(CFLAGS) $(CFLAGS_LIB) $(INCLUDES) $(CPPFLAGS) -o $@ $<

kann.o:kann.c
		$(CC) -c $(CFLAGS) $(CFLAGS_LIB) $(INCLUDES) $(CPPFLAGS) -o $@ $<

kann_extra/kann_data.o:kann_extra/kann_data.c
		$(CC) -c $(CFLAGS) -DHAVE_ZLIB $< -o $@

examples/mlp:examples/mlp.o kautodiff.o kann.o kann_extra/kann_data.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/vgg:examples/vgg.o kautodiff.o kann.o kann_extra/kann_data.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/ae:examples/ae.o kautodiff.o kann.o kann_extra/kann_data.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/vae:examples/vae.o kautodiff.o kann.o kann_extra/kann_data.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/textgen:examples/textgen.o kautodiff.o kann.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/rnn-bit:examples/rnn-bit.o kautodiff.o kann.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/inspect:examples/inspect.o kautodiff.o kann.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/mnist-cnn:examples/mnist-cnn.o kautodiff.o kann.o kann_extra/kann_data.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/mnist-cnn-allsplit:examples/mnist-cnn-allsplit.o kautodiff.o kann.o kann_extra/kann_data.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/mlp-mpi:examples/mlp-mpi.o kautodiff.o kann.o kann_extra/kann_data.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/mlp-mpi-allsplit:examples/mlp-mpi-allsplit.o kautodiff.o kann.o kann_extra/kann_data.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/recommender-mpi:examples/recommender-mpi.o kautodiff.o kann.o kann_extra/kann_data.o examples/to_onehot.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
		rm -fr *.o */*.o a.out */a.out *.a *.dSYM */*.dSYM $(EXE)

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c kann_extra/*.c examples/*.c)

# DO NOT DELETE

kann.o: kann.h kautodiff.h
kautodiff.o: kautodiff.h
kann_extra/kann_data.o: kann_extra/kseq.h kann_extra/kann_data.h
examples/ae.o: kann.h kautodiff.h kann_extra/kann_data.h
examples/inspect.o: kann.h kautodiff.h
examples/mlp.o: kann.h kautodiff.h kann_extra/kann_data.h
examples/mnist-cnn.o: kann_extra/kann_data.h kann.h kautodiff.h
examples/rnn-bit.o: kann.h kautodiff.h
examples/textgen.o: kann.h kautodiff.h
examples/vae.o: kann.h kautodiff.h kann_extra/kann_data.h
