#include <iostream>
#include <stdlib.h>
#include "util.h"

#define N 1
#define C 3
#define H 227
#define W 227

int main()
{
  AlexNet net;
  
  fmap input;
  input.dim1 = N;
  input.dim2 = C;
  input.dim3 = H;
  input.dim4 = W;
  input.data = (DATA*) malloc(N * C * H * W * sizeof(DATA));

  DATA (*temp)[C][H][W] = (DATA (*)[C][H][W])input.data;

  for(int i=0; i<N; i++)
    for(int j=0; j<C; j++)
      for(int k=0; k<H; k++)
        for(int l=0; l<W; l++)
          temp[i][j][k][l] = (i*C*H*W+j*H*W+k*W+l)%256;

  fmap* output = net.forward_pass(&input);

  for(int i=0; i<5; i++)
    std::cout << net.conv_layers[i]->exec_time << " ";
  
  for(int i=0; i<3; i++)
    std::cout << net.linear_layers[i]->exec_time << " ";

  std::cout << net.exec_time << std::endl;

  return 0;
}
