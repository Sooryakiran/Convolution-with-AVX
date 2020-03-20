#include <stdlib.h>
#include <iostream>
#include "util.h"
#include <immintrin.h>

int linear_test()
{
  int N = 4;
  int C = 128;
  int H = 1;
  int W = 1;
 

  fmap input = new_tensor(N, C, H, W);
  DATA (*temp)[C][H][W] = (DATA (*)[C][H][W])input.data;

  for(int i=0; i<N; i++)
    for(int j=0; j<C; j++)
      for(int k=0; k<H; k++)
        for(int l=0; l<W; l++)
          temp[i][j][k][l] = (i*C*H*W+j*H*W+k*W+l)%256;

  std::cout<<input.dim2<<std::endl;
  Linear *linear;
  linear = new Linear(128, 64);
  fmap* output = linear->linear(&input);
  std::cout<<output->data<<std::endl;
  std::cout<<linear->exec_time<<std::endl;
  return 0;
}

int linear_optim_test()
{
  int N = 4;
  int C = 128;
  int H = 1;
  int W = 1;
 

  fmap input = new_tensor(N, C, H, W);
  DATA (*temp)[C][H][W] = (DATA (*)[C][H][W])input.data;

  for(int i=0; i<N; i++)
    for(int j=0; j<C; j++)
      for(int k=0; k<H; k++)
        for(int l=0; l<W; l++)
          temp[i][j][k][l] = (i*C*H*W+j*H*W+k*W+l)%256;

  std::cout<<input.dim2<<std::endl;
  Linear *linear;
  linear = new Linear(128, 64);
  fmap* output = linear->linear_optimized(&input);
  std::cout<<output->data<<std::endl;
  std::cout<<linear->exec_time<<std::endl;
  return 0;
}


int conv_test()
{
  int N = 1;
  int C = 3;
  int H = 227;
  int W = 227;
 

  fmap input = new_tensor(N, C, H, W);
  DATA (*temp)[C][H][W] = (DATA (*)[C][H][W])input.data;

  for(int i=0; i<N; i++)
    for(int j=0; j<C; j++)
      for(int k=0; k<H; k++)
        for(int l=0; l<W; l++)
          temp[i][j][k][l] = (i*C*H*W+j*H*W+k*W+l)%256;

  std::cout<<input.dim2<<std::endl;
  Convolution *conv;
  conv = new Convolution(96, 3, 3, 3, 1, 1, 0, 0);
  fmap* output = conv->conv_2d(&input);
  std::cout<<output->dim1<<','<<output->dim2<<','<<output->dim3<<','<<output->dim4<<std::endl;
 
  return 0;
}

int relu_test()
{
  int N = 4;
  int C = 10;
  int H = 1;
  int W = 1;
 

  fmap input = new_tensor(N, C, H, W);
  DATA (*temp)[C][H][W] = (DATA (*)[C][H][W])input.data;

  for(int i=0; i<N; i++)
    for(int j=0; j<C; j++)
      for(int k=0; k<H; k++)
        for(int l=0; l<W; l++)
          temp[i][j][k][l] = 3.8;
  std::cout<<input.data[1]<<"\n---------"<<std::endl;
  relu(&input);
//   std::cout<<input.dim2<<std::endl;
  return 0;
}



int main(){

    linear_optim_test();
    linear_test();
    return 0;
}