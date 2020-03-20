#include <stdio.h>
#include <immintrin.h>
#include <time.h>
#include <stdint.h>

#define SIZE 1048576

int main()
{
	float *a, *b, *c1, *c2;

	a = (float*) malloc(SIZE * sizeof(float));
	b = (float*) malloc(SIZE * sizeof(float));
	c1 = (float*) malloc(SIZE * sizeof(float));
	c2 = (float*) malloc(SIZE * sizeof(float));

	for(int i=0; i<SIZE; i++)
	{
		a[i] = i*2;
		b[i] = i*3;
	}

	int start, end;
	__m256 aa, bb, cc;
	__m256i aaa;

	start = clock();

	for(int i=0; i<SIZE; i++)
		c1[i] = a[i] + b[i];

	end = clock();

	printf("Execution Time: %d\n", (end-start));
	
	start = clock();

  int start_idx;
	for(int i=0; i<SIZE/8+1; i++)
	{
    start_idx = i * 8 - SIZE;
    aaa = _mm256_set_epi32(start_idx+7, start_idx+6, start_idx+5, start_idx+4, start_idx+3, start_idx+2, start_idx+1, start_idx);
		
    aa = _mm256_maskload_ps((float const*) &a[8*i], aaa);
		bb = _mm256_maskload_ps((float const*) &b[8*i], aaa);

		cc = _mm256_add_ps(aa, bb);

		_mm256_maskstore_ps((float*) &c2[8*i], aaa, cc);
}

	end = clock();

	printf("Execution Time: %d\n", (end-start));
	
	return 0;
}
