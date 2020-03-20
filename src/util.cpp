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

fmap new_tensor(int N, int C, int H, int W){
  fmap input;
  input.dim1 = N;
  input.dim2 = C;
  input.dim3 = H;
  input.dim4 = W;
  input.data = (DATA*) malloc(N * C * H * W * sizeof(DATA));
  return input;
}

// fmap* new_tensor_ptr(int N, int C, int H, int W){
//   fmap xnew_tensor(N, C, H, W)
//   return input;
// }


fmap* Convolution::conv_2d(fmap* input_features)
{
  // Start time
  clock_t start, end;
  start = clock();

  // Batches
  // M is the number of filters
  // C is the number of channels
  // R x S is the filter size
  // Sx is stride x
  // Sy is stride y
  // Px is padding x
  // Py is padding y


  
  
  // Calculate dimensions
  int N = input_features->dim1;
  int H = input_features->dim3;
  int W = input_features->dim4;
  int E = int((H + 2*Py-S)/Sy + 1);
  int F = int((W + 2*Px-R)/Sx + 1);

  // Pad inputs
  input_features = padding_2d(input_features, Px, Py);
  
  // Allocate output fmap
  fmap* output_features = (fmap*) malloc(sizeof(new_tensor(N, M, E, F)));
  *output_features = new_tensor(N, M, E, F);
  
  // cast all data into easily interpretable form
  DATA (*temp)[M][E][F] = (DATA (*)[M][E][F])output_features->data;         // N x M x E x F
  DATA (*temp_weights)[C][R][S] = (DATA (*)[C][R][S])weights;               // M x C x R x S
  DATA (*temp_inputs)[C][H][W] = (DATA (*)[C][H][W])input_features->data;   // N x C x H x W
  
  // For all batches
  for(int n=0; n<N; n++){
    // For all filters
    for(int m=0; m<M; m++){
      // For each output point
      for(int x=0; x<F; x+=1)for(int y=0; y<E; y+=1)for(int c=0; c<C; c++){
        temp[n][m][y][x] = 0;
        for(int i=0; i<R; i++){
          for(int j=0; j<S; j++){
            temp[n][m][y][x] += temp_inputs[n][c][Sy*y+j][Sx*x+i] * temp_weights[m][c][i][j];
          }
        }
      }
    }
  }
  
  // Free inputs
  input_features =  NULL;
  free(input_features);
  
  // End time
  end = clock();
  exec_time = double(end-start) / double(CLOCKS_PER_SEC);
  
  return output_features;
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
    // Start time
  clock_t start, end;
  start = clock();

  // Batches
  // M is the number of filters
  // C is the number of channels
  // R x S is the filter size
  // Sx is stride x
  // Sy is stride y
  // Px is padding x
  // Py is padding y


  
  
  // Calculate dimensions
  int N = input_features->dim1;
  int H = input_features->dim3;
  int W = input_features->dim4;
  int E = int((H + 2*Py-S)/Sy + 1);
  int F = int((W + 2*Px-R)/Sx + 1);

  // Pad inputs
  input_features = padding_2d(input_features, Px, Py);
  
  // Allocate output fmap
  fmap* output_features = (fmap*) malloc(sizeof(new_tensor(N, M, E, F)));
  *output_features = new_tensor(N, M, E, F);
  
  // cast all data into easily interpretable form
  DATA (*temp)[M][E][F] = (DATA (*)[M][E][F])output_features->data;         // N x M x E x F
  DATA (*temp_weights)[C][R][S] = (DATA (*)[C][R][S])weights;               // M x C x R x S
  DATA (*temp_inputs)[C][H][W] = (DATA (*)[C][H][W])input_features->data;   // N x C x H x W
  
  // For all batches
  for(int n=0; n<N; n++){
    // For all filters
    for(int m=0; m<M; m++){
      // For each output point
      for(int x=0; x<F; x+=1)for(int y=0; y<E; y+=1)for(int c=0; c<C; c++){
        temp[n][m][y][x] = 0;
        for(int i=0; i<R; i++){
          for(int j=0; j<S; j++){
            temp[n][m][y][x] += temp_inputs[n][c][Sy*y+j][Sx*x+i] * temp_weights[m][c][i][j];
          }
        }
      }
    }
  }
  
  // Free inputs
  input_features =  NULL;
  free(input_features);
  
  // End time
  end = clock();
  exec_time = double(end-start) / double(CLOCKS_PER_SEC);
  
  return output_features;

}

fmap* Linear::linear(fmap* input_features)
{
  clock_t start, end;
  start = clock();

  // M input size
  // L output size
  // output L = M*weights

  // Allocate the memory for outputs
  int N = input_features->dim1;
  fmap* output_features = (fmap*) malloc(sizeof(new_tensor(N, L, 1, 1)));
  *output_features = new_tensor(N, L, 1, 1);
  
  // cast all data into easily interpretable form
  DATA (*temp)[L] = (DATA (*)[L])output_features->data;
  DATA (*temp_weights)[L] = (DATA (*)[L])weights;
  DATA (*temp_inputs)[M] = (DATA (*)[M])input_features->data;

  // Naive computation
  for(int n=0; n<N; n++){  
    for(int l=0; l<L; l++){
      temp[n][l]=0;
      for(int m=0; m<M; m++){
        temp[n][l] += temp_weights[m][l]*temp_inputs[n][m];
      }
    }
  }

  // Free inputs
  input_features =  NULL;
  free(input_features);

  end = clock();
  exec_time = double(end-start) / double(CLOCKS_PER_SEC);

  return output_features;
}

fmap* Linear::linear_optimized(fmap* input_features)
{
  clock_t start, end;
  start = clock();

  // M input size
  // L output size
  // output L = M*weights

  // Allocate the memory for outputs
  int N = input_features->dim1;
  fmap* output_features = (fmap*) malloc(sizeof(new_tensor(N, L, 1, 1)));
  *output_features = new_tensor(N, L, 1, 1);
  
  // cast all data into easily interpretable form
  DATA (*temp)[L] = (DATA (*)[L])output_features->data;
  DATA (*temp_weights)[L] = (DATA (*)[L])weights;
  DATA (*temp_inputs)[M] = (DATA (*)[M])input_features->data;

  // Computation

  __m256 mm_inputs, mm_weights, partials;
  int32_t weights_array[8], input_array[8], partials_array[8];
  for(int n=0; n<N; n++){  
    for(int l=0; l<L; l++){
      temp[n][l]=0;

      for(int m=0; m>int(M/8); m++){
        // Masked load
        for(int i=0; i<8; i++){
          weights_array[i] = temp_weights[m+i][l];
          input_array[i] = temp_inputs[n][m+i];
        }
        mm_inputs = _mm256_loadu_ps((float const*) &input_array);
        mm_weights = _mm256_loadu_ps((float const*) &weights_array);
        partials = _mm256_mul_ps(mm_inputs, mm_weights);
        _mm256_storeu_ps((float*) &partials_array, partials);
        for(int i=0; i<8; i++){
          temp[n][l] += partials[i];
        }
      }

      // Manually zero masked
      int m = int(M/8);
     
      for(int i=0; i<8; i++){
        weights_array[i] = 0;
        input_array[i] = 0;
      }

      for(int i=0; i<M%8; i++){
        weights_array[i] = temp_weights[m+i][l];
        input_array[i] = temp_inputs[n][m+i];
      }
      mm_inputs = _mm256_loadu_ps((float const*) &input_array);
      mm_weights = _mm256_loadu_ps((float const*) &weights_array);
      partials = _mm256_mul_ps(mm_inputs, mm_weights);
      _mm256_storeu_ps((float*) &partials_array, partials);

      for(int i=0; i<M%8; i++){
        temp[n][l] += partials[i];
      }
    }
  }

  // Free inputs
  input_features =  NULL;
  free(input_features);

  end = clock();
  exec_time = double(end-start) / double(CLOCKS_PER_SEC);

  return output_features;
}

void relu(fmap* input_features)
{
  // Get input dimensions
  int dim1 = input_features->dim1;
  int dim2 = input_features->dim2;
  int dim3 = input_features->dim3;
  int dim4 = input_features->dim4;

  DATA (*temp) = (DATA (*))input_features->data;  
  int size = dim1*dim2*dim3*dim4;

  __m256 v_inp, v_out, cmp;
  __m256i mask;
  int start_idx = 0;

  for(int i=0; i<size; i++){
    if(temp[i]<=0){
      temp[i]=0;
    }
  }
}

fmap* padding_2d(fmap* input_features, int Px, int Py){
  
  // Padding size Px x Py
  // Calculate dimensions
  int N = input_features->dim1;
  int C = input_features->dim2;
  int H = input_features->dim3;
  int W = input_features->dim4;
  int E = int(H + Py*2);
  int F = int(W + Px*2);

  // Allocate output fmap
  fmap* output_features = (fmap*) malloc(sizeof(new_tensor(N, C, E, F)));
  *output_features = new_tensor(N, C, E, F);

  // cast all data into easily interpretable form
  DATA (*temp)[C][E][F] = (DATA (*)[C][E][F])output_features->data;         // N x M x E x F
  DATA (*temp_inputs)[C][H][W] = (DATA (*)[C][H][W])input_features->data;   // N x C x H x W
  
  for(int n=0; n<N; n++){
    for(int c=0; c<C; c++){
      for(int y=0; y<E; y++){
        for(int x=0; x<F; x++){
          if((x>Px && x<F-Px) || (y>Py && y<E-Py)){
            temp[n][c][y][x] = temp_inputs[n][c][Py+y][Px+x];
          }
          else{
            temp[n][c][y][x] = 0;
          }
        }
      }
    }
  }

  // Free inputs
  input_features = NULL;
  free(input_features);
  return output_features;
}

fmap* maxpool_2d(fmap* input_features, int R, int S, int Sx, int Sy)
{
  // Kernel size R x S
  // Strides Sx x Sy

  // Calculate output size

  int N = input_features->dim1;
  int C = input_features->dim2;
  int H = input_features->dim3;
  int W = input_features->dim4;
  int E = int((H - S)/Sy + 1);
  int F = int((W - R)/Sx + 1);

  // Allocate output fmap
  fmap* output_features = (fmap*) malloc(sizeof(new_tensor(N, C, E, F)));
  *output_features = new_tensor(N, C, E, F);
  
  // cast all data into easily interpretable form
  DATA (*temp)[C][E][F] = (DATA (*)[C][E][F])output_features->data;         // N x M x E x F
  DATA (*temp_inputs)[C][H][W] = (DATA (*)[C][H][W])input_features->data;   // N x C x H x W  

  int max_val = 0;
  for(int n=0; n<N; n++){
    for(int c=0; c<C; c++){
      for(int y=0; y<E; y++){
        for(int x=0; x<F; x++){
          max_val = temp_inputs[n][c][y][x];
          for(int i=0; i<R; i++)for(int j=0; j<S; j++){
            if(temp_inputs[n][c][y*Sy+j][x*Sx+i]>max_val){
              max_val = temp_inputs[n][c][y*Sy+j][x*Sx+i];
            }
          }
          temp[n][c][x][y] = max_val;
        }
      }
    }
  }

  return output_features;
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
