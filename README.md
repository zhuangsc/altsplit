Altsplit - An MPI-based distributed model parallelism technique for MLP

This repository contains the implementation of the standard 
model parallelism of an MLP and the Altsplit version. The description of 
Altsplit can be found at my PhD thesis (https://upcommons.upc.edu/handle/2117/173617)
Section 6. 

The implementation is based on the KANN deep learning framework (https://github.com/attractivechaos/kann)

You need an MPI implementation in order to compile and run the code.
This repo provides two benchmarks in the src/examples/kann-data directory:
- MNIST 
- Cifar10

Example:
1. The standard model parallelism version on MNIST

mpirun examples/mlp-mpi-allsplit -m ${EPOCHS} -t 1 -l ${NUM_LAYERS} \
                                 -n ${NUM_NEURONS} -B ${BATCH_SIZE} -d ${DROPOUT} -r ${LEARNING_RATE} 
                                 -o ${MOD} examples/kann-data/mnist-train-x.knd.gz examples/kann-data/mnist-train-y.knd.gz 

2. Altsplit version
mpirun examples/mlp-mpi -m ${EPOCHS} -t 1 -l ${NUM_LAYERS} \
                        -n ${NUM_NEURONS} -B ${BATCH_SIZE} -d ${DROPOUT} -r ${LEARNING_RATE} 
                        -o ${MOD} examples/kann-data/mnist-train-x.knd.gz examples/kann-data/mnist-train-y.knd.gz 

Sicong Zhuang
sicong.zhuang@gmail.com
