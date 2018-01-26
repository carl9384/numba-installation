import math
from numba import cuda, float32
import numpy as np
import time

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp


@cuda.jit
def fast_sum(A,C):

    x = cuda.grid(1)

    if x>=A.shape[0]:
        return

    temp = 0
    for i in range(A.shape[1]):
        temp += A[x,i]
    C[x] = temp


def run_transfer_test(rows=100,columns=100):
    A = np.ones((rows,columns))

    time_start_copy_to_device = time.time()

    A_g = cuda.to_device(A)

    time_end_copy_to_device = time.time()
    print("Total time to transfer to gpu for "+str(rows)+"x"+str(columns)+" matrix is: ")
    print("\t\t\t\t\t\t",time_end_copy_to_device - time_start_copy_to_device)
    print()


def run_fast_sum(rows=100,columns=100,runntimes=1,transferntimes=1,show_array=False):
    A = np.ones((rows,columns))

    print("Shape of A is ",A.shape)
    if show_array:
        print(A)
    time_start_copy_to_device = time.time()

    for i in range(transferntimes):
        A_g = cuda.to_device(A)
        C_g = cuda.device_array(shape=(A.shape[0]))

    time_end_copy_to_device = time.time()
    print("Time to copy to device is ",time_end_copy_to_device-time_start_copy_to_device)

    threadsperblock = (TPB,1,1)
    blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x,1,1)

    time_gpu = time.time()

    for i in range(runntimes):
        if show_array:
            print("Kernel execution ",i)
        fast_sum[blockspergrid,threadsperblock](A_g,C_g)

    C = C_g.copy_to_host()
    print("Time to compute on GPU and copy to host is ",time.time() - time_gpu)
    print()
    print("Total time to compute on GPU is ",time.time() - time_start_copy_to_device)
    print()
    if show_array:
        print("C is ",C)

    time_cpu = time.time()

    for i in range(runntimes):
        if show_array:
            print("CPU execution ",i)
        A.sum(axis=1)
    print("Total time to compute on CPU is ",time.time() - time_cpu)
    print()
    print("**********************************************************")

if __name__ == "__main__":


    # Instantiate matrices: A*B = C
    A = 2*np.ones((100,100))
    print("Shape of A is ",A.shape)
    print(A)
    B = A
    print("Shape of B is ",B.shape)
    print(B)

    time_copy_to_device = time.time()
    A_g = cuda.to_device(A)
    B_g = cuda.to_device(B)
    C_g = cuda.device_array(shape=(A.shape[0],B.shape[1]))
    print("Time to copy to device is ",time.time() - time_copy_to_device)

    threadsperblock = (TPB,1,1)
    blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x,1,1)

    time_gpu = time.time()
    fast_matmul[blockspergrid,threadsperblock](A_g,B_g,C_g)
    C = C_g.copy_to_host()
    print("Time to compute on GPU and copy to host is ",time.time() - time_gpu)
    print("Total time for GPU is ",time.time() - time_copy_to_device)

