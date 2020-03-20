#include <stdio.h>
#include <immintrin.h>
#include <time.h>

#define SIZE 1048576

int main()
{
	int *a, *b, *c1, *c2;

	a = (int*) malloc(SIZE * sizeof(int));
	b = (int*) malloc(SIZE * sizeof(int));
	c1 = (int*) malloc(SIZE * sizeof(int));
	c2 = (int*) malloc(SIZE * sizeof(int));

	for(int i=0; i<SIZE; i++)
	{
		a[i] = i*2;
		b[i] = i*3;
	}

	int start, end;
	__m256 aa, bb, cc;
	__m256 aaa, bbb, ccc;

	start = clock();

	for(int i=0; i<SIZE; i++)
		c1[i] = a[i] + b[i];

	end = clock();

	printf("Execution Time: %d\n", (end-start));
	
	start = clock();

	for(int i=0; i<SIZE/16; i++)
	{
		aa = _mm256_loadu_ps((float const*) &a[16*i]);
		bb = _mm256_loadu_ps((float const*) &b[16*i]);

		cc = _mm256_add_ps(aa, bb);

		_mm256_storeu_ps((float*) &c2[16*i], cc);
	
    aaa = _mm256_loadu_ps((float const*) &a[16*i+8]);
		bbb = _mm256_loadu_ps((float const*) &b[16*i+8]);

		ccc = _mm256_add_ps(aaa, bbb);

		_mm256_storeu_ps((float*) &c2[16*i+8], ccc);
}

	end = clock();

	printf("Execution Time: %d\n", (end-start));
	
	return 0;
}
