// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:02:28 on Tue, Feb 28, 2023
//
// Description: mma naive hgemm

#include "common.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_SIZE 32
/*
* 这个mmaNaiveKernel解决: 16*8*16的矩阵乘法, warp=32,那么矩阵A为由4个 8*4 矩阵组成(row-major)且
*
*/

__global__ void mmaNaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                               size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, MMA_K); //除法，向上取整,计算出在K维度上的tiles数

    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N; // 计算出该block需要计算的tile在原来矩阵中的开始位置.

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ half A_smem[MMA_M][MMA_K]; //bf 16 (16位存储) [16,16],warp组织为[8*4]
    __shared__ half B_smem[MMA_N][MMA_K]; //bf 16 (16位存储) [8,16],warp组织为[4*8]
    __shared__ half C_smem[MMA_M][MMA_N]; //bf 16 (16位存储) [16,8],warp组织为[8*4]

    const size_t lane_id = threadIdx.x % WARP_SIZE;//计算出该线程是warp中的编号

    uint32_t RC[2] = {0, 0}; //RC寄存器，在第一轮迭代时这个RC就是空，后面就是上一轮tiles的结果

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =  //int4是4个整数组成的数据类型,每个int都是32位,因此一行A_smem代表两个thread的值
            *((int4 *)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2); // K表示row的stride,i*MMA_K表示列的offset,为什么要采用row-major的方法

        if (lane_id < MMA_N * 2) { //因为使用的是LDMATRIX_X2来load，所以只需要16个threads完成操作.
            *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =  //B_smem 一行由两个线程占据数值。一共16个Threads
                *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
        }

        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        //lane_id /2 为row-major
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]); //为什么要乘以8，因为每个线程指向128bits一行(128/16=8个)
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]); //Thread按照行优先的顺序，如果((lane_id / 8) % 2)没有%2,那么在这个矩阵下面就排了16个threads,%2后,laned_id>=16的
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

        __syncthreads();
    }

    *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0]; //行优先排序
    *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];  //写回shared_memroy

    __syncthreads();

    if (lane_id < MMA_M) { // 只要16个线程实现 C_smem的数据搬移，且每个threads搬运C_smem的每一行数据.
        *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_smem[lane_id][0])); 
    }
}

void mmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    // 对矩阵A,矩阵B,矩阵C进行分块.
    dim3 block(WARP_SIZE); //每个block的设置为一维threads，长度为32，相当于一个warp计算出D矩阵的[MMA_N,MMA_H]矩阵的结果.
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M)); //[N,M]是D=AB+C的计算结果的维度

    mmaNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
    /*
    * 总结：利用TensorCore来实现矩阵乘，我们需要考虑
    * 数据从global_memory搬运到shared_memory，
    * 然后搬运到register(利用ldmatrix),然后利用mma的指令集，
    * 然后把结果从register搬运到shared memory,然后再搬运到global memory.
    * (但是考虑问题的时候从使用哪个mma指令集开始出发,然后找到相应的ldmatrix)
    */
}
