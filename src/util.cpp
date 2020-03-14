#include <stdlib.h>
#include <iostream>
#include "util.h"
#include <immintrin.h>

Convolution::Convolution(int m, int c, int r, int s, int sx, int sy, int px, int py)
{
  M = m;
  C = c;
  R = r;
  S = s;
  Sx = sx;
  Sy = sy;
  Px = px;
  Py = py;
  weights = (DATA*) malloc(M * C * R * S * sizeof(DATA));
  DATA (*temp)[C][R][S] = (DATA (*)[C][R][S])weights;
  for(int i=0; i<M; i++)
    for(int j=0; j<C; j++)
      for(int k=0; k<R; k++)
        for(int l=0; l<S; l++)
          temp[i][j][k][l] = (i*C*R*S+j*R*S+k*S+l)%256;
}

Linear::Linear(int m, int l)
{
  M = m;
  L = l;
  weights = (DATA*) malloc(M * L * sizeof(DATA));
  DATA (*temp)[L] = (DATA (*)[L])weights;
  for(int i=0; i<M; i++)
    for(int j=0; j<L; j++)
      temp[i][j] = (i*L+j)%256;
}
fmap* Convolution::conv_2d(fmap* input_features)
{
  return NULL;
}

fmap* Convolution::conv2d_IS(fmap* input_features)
{
  return NULL;
}

fmap* Convolution::conv2d_OS(fmap* input_features)
{
  return NULL;
}

fmap* Convolution::conv2d_WS(fmap* input_features)
{
  return NULL;
}

fmap* Convolution::conv2d_optimized(fmap* input_features)
{
  return NULL;
}

fmap* Linear::linear(fmap* input_features)
{
  // M output size
  // L input size
  // output L = M*weights
  int N = input_features->dim1;
  fmap *output_features;
  output_features->dim1 = N;
  output_features->dim2 = output_features->dim3 = 1;
  output_features->dim4 = M;
  output_features->data = (DATA*) malloc(N*M * sizeof(DATA));

  DATA (*temp)[1][1][M] = (DATA (*)[1][1][M])output_features->data;
  for(int n=1; n<N; n++){  
    for(int m=1; m<M; m++){
      temp[n][1][1][m];
      for(int l=1; l<L; l++){
        temp[n][1][1][m] += weights[1][1][m][l]*input_features->data[n][1][1][l];
      }
    }
  }
  return output_features;
}

fmap* Linear::linear_optimized(fmap* input_features)
{
  return NULL;
}

void relu(fmap* input_features)
{
  // return NULL;
}

fmap* maxpool_2d(fmap* input_features, int R, int S, int Sx, int Sy)
{
  return NULL;
}

AlexNet::AlexNet()
{
  conv_layers = (Convolution**) malloc(5 * sizeof(Convolution*));

  Convolution *conv;
  conv = new Convolution(96, 3, 11, 11, 4, 4, 2, 2);
  conv_layers[0] = conv;
  conv = new Convolution(256, 96, 5, 5, 1, 1, 2, 2);
  conv_layers[1] = conv;
  conv = new Convolution(384, 256, 3, 3, 1, 1, 1, 1);
  conv_layers[2] = conv;
  conv = new Convolution(384, 384, 3, 3, 1, 1, 1, 1);
  conv_layers[3] = conv;
  conv = new Convolution(256, 384, 3, 3, 1, 1, 1, 1);
  conv_layers[4] = conv;

  linear_layers = (Linear**) malloc(3 * sizeof(Linear*));

  Linear *linear;
  linear = new Linear(4096, 9216);
  linear_layers[0] = linear;
  linear = new Linear(4096, 4096);
  linear_layers[1] = linear;
  linear = new Linear(1000, 4096);
  linear_layers[2] = linear;
}

fmap* AlexNet::forward_pass(fmap* input_features)
{
  clock_t start, end;
  start = clock();

  fmap* temp = input_features;
  
  temp = conv_layers[0]->conv_2d(temp);
  relu(temp);
  temp = maxpool_2d(temp, 3, 3, 2, 2);
  temp = conv_layers[1]->conv_2d(temp);
  relu(temp);
  temp = maxpool_2d(temp, 3, 3, 2, 2);
  temp = conv_layers[2]->conv_2d(temp);
  relu(temp);
  temp = conv_layers[3]->conv_2d(temp);
  relu(temp);
  temp = conv_layers[4]->conv_2d(temp);
  relu(temp);
  temp = maxpool_2d(temp, 3, 3, 2, 2);

  int lin_dim = temp->dim2 * temp->dim3 * temp->dim4;
  temp->dim2 = lin_dim;
  temp->dim3 = temp->dim4 = 1;

  temp = linear_layers[0]->linear(temp);
  relu(temp);
  temp = linear_layers[1]->linear(temp);
  relu(temp);
  temp = linear_layers[2]->linear(temp);
  relu(temp);

  end = clock();

  exec_time = double(end-start) / double(CLOCKS_PER_SEC);
  return temp;
}
