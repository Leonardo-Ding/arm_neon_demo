#include <arm_neon.h>

#include <math.h>
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>

#define DIM 1024

//#include <stdlib.h>
void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < wB; ++i)
        for (unsigned int j = 0; j < hA; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[k * hA + j];
                double b = B[i * wA + k];
                sum += a * b;
            }

            C[i * hA + j] = (float)sum;
        }
}

void neonmult(const float *matrixA, const float *matrixB, float *matrixR)
{
#if 1	
	asm volatile(
		"ldr q0, [%[a_ptr]]\n"
		"ldr q1, [%[a_ptr], #16]\n"
		"ldr q2, [%[a_ptr], #32]\n"
		"ldr q3, [%[a_ptr], #48]\n"
		"ldr q4, [%[b_ptr]]\n"
		"ldr q5, [%[b_ptr], #16]\n"
		"ldr q6, [%[b_ptr], #32]\n"
		"ldr q7, [%[b_ptr], #48]\n"
		"movi v8.4s, #0x0\n"
		"movi v9.4s, #0x0\n"
		"movi v10.4s, #0x0\n"
		"movi v11.4s, #0x0\n"
		"fmla v8.4s, v0.4s, v4.s[0]\n"
		"fmla v9.4s, v0.4s, v5.s[0]\n"
		"fmla v10.4s, v0.4s, v6.s[0]\n"
		"fmla v11.4s, v0.4s, v7.s[0]\n"
		"fmla v8.4s, v1.4s, v4.s[1]\n"
		"fmla v9.4s, v1.4s, v5.s[1]\n"
		"fmla v10.4s, v1.4s, v6.s[1]\n"
		"fmla v11.4s, v1.4s, v7.s[1]\n"
		"fmla v8.4s, v2.4s, v4.s[2]\n"
		"fmla v9.4s, v2.4s, v5.s[2]\n"
		"fmla v10.4s, v2.4s, v6.s[2]\n"
		"fmla v11.4s, v2.4s, v7.s[2]\n"
		"fmla v8.4s, v3.4s, v4.s[3]\n"
		"fmla v9.4s, v3.4s, v5.s[3]\n"
		"fmla v10.4s, v3.4s, v6.s[3]\n"
		"fmla v11.4s, v3.4s, v7.s[3]\n"
		"str q8, [%[c_ptr]]\n"
		"str q9, [%[c_ptr], #16]\n"
		"str q10, [%[c_ptr], #32]\n"
		"str q11, [%[c_ptr], 48]"
		:
		: [a_ptr] "r"(matrixA),
		  [b_ptr] "r"(matrixB),
		  [c_ptr] "r"(matrixR)
		: "v0", "v1", "v2", "v3", 
		  "v4", "v5", "v6", "v7", 
		  "v8", "v9", "v10", "v11", 
		  "cc", "memory"
	);
#else
	float32x4_t a0, a1, a2, a3, b0, b1, b2, b3;
	float32x4_t r0, r1, r2, r3;
	a0 = vld1q_f32(matrixA);
	a1 = vld1q_f32(matrixA + 4);
	a2 = vld1q_f32(matrixA + 8);
	a3 = vld1q_f32(matrixA + 12);
	b0 = vld1q_f32(matrixB);
	b1 = vld1q_f32(matrixB + 4);
	b2 = vld1q_f32(matrixB + 8);
	b3 = vld1q_f32(matrixB + 12);

	r0 = vmulq_lane_f32(a0, vget_low_f32(b0), 0);
	r0 = vmlaq_lane_f32(r0, a1, vget_low_f32(b0), 1);
	r0 = vmlaq_lane_f32(r0, a2, vget_high_f32(b0), 0);
	r0 = vmlaq_lane_f32(r0, a3, vget_high_f32(b0), 1);
	
	r1 = vmulq_lane_f32(a0, vget_low_f32(b1), 0);
	r1 = vmlaq_lane_f32(r1, a1, vget_low_f32(b1), 1);
	r1 = vmlaq_lane_f32(r1, a2, vget_high_f32(b1), 0);
	r1 = vmlaq_lane_f32(r1, a3, vget_high_f32(b1), 1);

	r2 = vmulq_lane_f32(a0, vget_low_f32(b2), 0);
	r2 = vmlaq_lane_f32(r2, a1, vget_low_f32(b2), 1);
	r2 = vmlaq_lane_f32(r2, a2, vget_high_f32(b2), 0);
	r2 = vmlaq_lane_f32(r2, a3, vget_high_f32(b2), 1);

	r3 = vmulq_lane_f32(a0, vget_low_f32(b3), 0);
	r3 = vmlaq_lane_f32(r3, a1, vget_low_f32(b3), 1);
	r3 = vmlaq_lane_f32(r3, a2, vget_high_f32(b3), 0);
	r3 = vmlaq_lane_f32(r3, a3, vget_high_f32(b3), 1);

	vst1q_f32(matrixR, r0);
	vst1q_f32(matrixR + 4,  r1);
	vst1q_f32(matrixR + 8,  r2);
	vst1q_f32(matrixR + 12, r3);
#endif	
}

void sgemm_neon(float *C, float *A, float *B, int M, int N, int K)
{
	float32x4_t a0, a1, a2, a3;
	float32x4_t b0, b1, b2, b3;
	float32x4_t r0, r1, r2, r3;

	for (int i = 0; i < N; i += 4) 
	{
		for (int j = 0; j < M; j += 4)
		{
			r0 = vdupq_n_f32(.0f);
			r1 = vdupq_n_f32(.0f);
			r2 = vdupq_n_f32(.0f);
			r3 = vdupq_n_f32(.0f);

			// A,B,R offsets
			float *matrixA = A + j;
			float *matrixB = B + i * K;
			float *matrixR = C + j + i * M;

			for (int k = 0; k < K; k += 4)
			{
				a0 = vld1q_f32(matrixA + 0*K);
			        a1 = vld1q_f32(matrixA + 1*K);
			        a2 = vld1q_f32(matrixA + 2*K);
			        a3 = vld1q_f32(matrixA + 3*K);
			        b0 = vld1q_f32(matrixB + 0*K);
			        b1 = vld1q_f32(matrixB + 1*K);
			        b2 = vld1q_f32(matrixB + 2*K);
			        b3 = vld1q_f32(matrixB + 3*K);
 
			        r0 = vmlaq_lane_f32(r0, a0, vget_low_f32(b0), 0);
			        r0 = vmlaq_lane_f32(r0, a1, vget_low_f32(b0), 1);
			        r0 = vmlaq_lane_f32(r0, a2, vget_high_f32(b0), 0);
			        r0 = vmlaq_lane_f32(r0, a3, vget_high_f32(b0), 1);
         
			        r1 = vmlaq_lane_f32(r1, a0, vget_low_f32(b1), 0);
			        r1 = vmlaq_lane_f32(r1, a1, vget_low_f32(b1), 1);
			        r1 = vmlaq_lane_f32(r1, a2, vget_high_f32(b1), 0);
			        r1 = vmlaq_lane_f32(r1, a3, vget_high_f32(b1), 1);
 
			        r2 = vmlaq_lane_f32(r2, a0, vget_low_f32(b2), 0);
			        r2 = vmlaq_lane_f32(r2, a1, vget_low_f32(b2), 1);
			        r2 = vmlaq_lane_f32(r2, a2, vget_high_f32(b2), 0);
			        r2 = vmlaq_lane_f32(r2, a3, vget_high_f32(b2), 1);
 
			        r3 = vmlaq_lane_f32(r3, a0, vget_low_f32(b3), 0);
			        r3 = vmlaq_lane_f32(r3, a1, vget_low_f32(b3), 1);
			        r3 = vmlaq_lane_f32(r3, a2, vget_high_f32(b3), 0);
			        r3 = vmlaq_lane_f32(r3, a3, vget_high_f32(b3), 1);

				matrixA += 4 * K;
				matrixB += 4;
			}

			//printf("i: %d, j: %d\n", i, j);
			// store R
			vst1q_f32(matrixR + 0*M, r0);
			vst1q_f32(matrixR + 1*M, r1);
			vst1q_f32(matrixR + 2*M, r2);
			vst1q_f32(matrixR + 3*M, r3);
		}
	}
}

int main(int argc, char** argv)
{
	// parse input
	int nIter = 10;
	if (argc == 2)
		nIter = atoi(argv[1]);

	// variables
	float *A, *B, *C, *C1;
	A = new float[DIM * DIM];
	B = new float[DIM * DIM];
	C = new float[DIM * DIM];
	C1 = new float[DIM * DIM];

	// random init
	for (int i = 0; i < DIM * DIM; i++)
	{
		A[i] = (float)rand() / RAND_MAX;
		B[i] = (float)rand() / RAND_MAX;
	}

	// timestamp
	struct timeval start, stop;
	gettimeofday(&start, NULL);

	// execute for loop
	for (int i = 0; i < nIter; i++) {
		sgemm_neon(C, A, B, DIM, DIM, DIM);
	}

	// timestamp
	gettimeofday(&stop, NULL);
	double t = (stop.tv_sec-start.tv_sec)*1e3+(stop.tv_usec-start.tv_usec)/1e3;
	printf("Done matrixMul dim %d time used %fms!\n", DIM, t / nIter);

	// element-wise
	matrixMulCPU(C1, A, B, DIM, DIM, DIM);
	for(int i = 0; i < DIM * DIM; i++)
	{
		float diff = fabs(C1[i] - C[i]);
		if (diff > 1e-3)
		{
			printf("ERROR: index is %d, diff is %f\n", i, diff);
			break;
		}
	}

	delete []A;
	delete []B;
	delete []C;
	delete []C1;
	return 0;
}
