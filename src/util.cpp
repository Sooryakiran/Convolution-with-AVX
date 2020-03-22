#include <stdlib.h>
#include <iostream>
#include "util.h"
#include <immintrin.h>

Convolution::Convolution(int m, int c, int r, int s, int sx, int sy, int px, int py)
{

  // This is the class constructor for the convolution class. It allocates memory for
  // the weights of appropriate sizes and initializes them with dummy values.
  //
  // @param m: [int] Number of out channels (filters)
  // @param c: [int] Number of in channels
  // @param r: [int] Kernel size in Y dimension
  // @param s: [int] Kernel size in X dimension
  // @param sx: [int] Strides in X dimension
  // @param sy: [int] Strides in Y dimension
  // @param px: [int] Zero padding in X dimension
  // @param py: [int] Zero padding in Y dimension

  M   = m;
  C   = c;
  R   = r;
  S   = s;
  Sx  = sx;
  Sy  = sy;
  Px  = px;
  Py  = py;

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

  // This is the class constructor for the Linear/Fully connected layers. It allocates
  // memory for the weights of appropriate sizes and initializes them with dummy values.
  // 
  // @param m: [int] output vector size (number of neurons)
  // @param l: [int] input vector size

  M = m;
  L = l;

  weights = (DATA*) malloc(M * L * sizeof(DATA));
  DATA (*temp)[L] = (DATA (*)[L])weights;

  for(int i=0; i<M; i++)
    for(int j=0; j<L; j++)
      temp[i][j] = (i*L+j)%256;
}


fmap new_tensor(int N, int C, int H, int W){

  // This function creates and allocates memory for a new fmap 4D tensor of the
  // given dimensions in NCHW format.
  // 
  // @param N: [int] Batch Size
  // @param C: [int] Channel Size
  // @param H: [int] Height of the fmap
  // @param W: [int] Width of the fmap
  // @return input: [fmap] Tensor of appropriate dimensions

  fmap input;
  input.dim1 = N;
  input.dim2 = C;
  input.dim3 = H;
  input.dim4 = W;
  input.data = (DATA*) malloc(N * C * H * W * sizeof(DATA));
  return input;
}

int min(int x, int y){

  // This is a small utility function to return the min value among two integers.
  // 
  // @param x: [int] Integer 1
  // @param y: [int] Integer 2
  // @return : minimum among x and y

  return (x<y)?x:y;
}

int max(int x, int y){

  // This is a small utility function to return the max value among two integers.
  // 
  // @param x: [int] Integer 1
  // @param y: [int] Integer 2
  // @return : maximum among x and y

  return (x>y)?x:y;
}


fmap* Convolution::conv2d_tiled(fmap* input_features, int tile_size){

  // This function performs 2D convolution operation with tiling and returns the 
  // output.
  // 
  // @param input_features: [fmap] input activations for the convolutional layer
  // @param tile_size     : [int] tile size in the output space
  // @return output_features: [fmap] output of the convolutional layer

  // Start timing
  clock_t start, end;
  start = clock();

  // Calculate dimensions of output 
  int N = input_features->dim1;
  int H = input_features->dim3;
  int W = input_features->dim4;
  int E = int((H + 2*Py-S)/Sy + 1);
  int F = int((W + 2*Px-R)/Sx + 1);

  // Pad inputs
  input_features = padding_2d(input_features, Px, Py);
  
  // Allocate output fmap
  fmap* output_features = (fmap*) malloc(sizeof(new_tensor(N, M, E, F)));
  *output_features      = new_tensor(N, M, E, F);
  
  // cast all data into easily interpretable form
  DATA (*temp)[M][E][F]         = (DATA (*)[M][E][F])output_features->data;         // N x M x E x F
  DATA (*temp_weights)[C][R][S] = (DATA (*)[C][R][S])weights;               // M x C x R x S
  DATA (*temp_inputs)[C][H][W]  = (DATA (*)[C][H][W])input_features->data;   // N x C x H x W

  // Check if tiling size is greater than kernel size
  if(max(R, S)>=tile_size){
    std::cout<<"[ERROR] Tile size should be greater that the kernel size."<<std::endl;
    return NULL;
  }

  // for each tile
  for(int yt=0; yt<E/tile_size + 1; yt++){
    for(int xt=0; xt<F/tile_size + 1; xt++){
      // output maps for xt will be from xt*tile_size to min((xt+1)*tile_size, F)
      // output maps fot yt will be from yt*tile_size to min((yt+1)*tile_size, E)
      int Xmin = xt*tile_size, Xmax = min((xt+1)*tile_size, F), Ymin = yt*tile_size, Ymax = min((yt+1)*tile_size, E);

      // this maps to input as from Xmin*Sx to Xmax*Sx + S
      // this maps to input as from Ymin*Sx to Ymax*Sx + R
      int Xmin_inp = Sx*Xmin, Xmax_inp = Xmax*Sx + S, Ymin_inp = Sy*Ymin, Ymax_inp = Sy*Ymax + R;
      
      fmap* input_tile = (fmap*) malloc(sizeof(new_tensor(N, C, Ymax-Ymin, Xmax-Xmin)));
      *input_tile      = new_tensor(N, C, Ymax-Ymin, Xmax-Xmin);
      DATA (*temp_input_tile)[C][Ymax-Ymin][Xmax-Xmin] = (DATA (*)[C][Ymax-Ymin][Xmax-Xmin])input_features->data;   // N x C x H x W
      
      for(int n=0; n<N; n++){
        for(int c=0; c<C; c++){
          for(int y=Ymin_inp; y<Ymax_inp; y++){
            for(int x=Xmin_inp; x<Xmax_inp; x++){
              temp_input_tile[n][c][y-Ymin_inp][x-Xmin_inp] = temp_inputs[n][c][y][x];
            }
          }
        }
      }

      // Perform the convolution
      fmap* output_tiles = this->conv2d_tiled_aux(input_tile);
      DATA (*temp_output_tiles)[M][Ymax-Ymin][Xmax-Xmin] = (DATA (*)[M][Ymax-Ymin][Xmax-Xmin])output_tiles->data; 
      
      // Write output
      for(int n=0; n<N; n++){
        for(int m=0; m<M; m++){
          for(int y=0; y<Ymax-Ymin; y++){
            for(int x=0; x<Xmax-Xmin; x++){
              temp[n][m][yt*tile_size+y][xt*tile_size+x] = temp_output_tiles[n][m][y][x];
            }
          }
        }
      }
    }
  }

  // Free inputs
  input_features =  NULL;
  free(input_features);

  // End timing
  end = clock();
  exec_time = double(end-start) / double(CLOCKS_PER_SEC);

  return output_features;
}

fmap* Convolution::conv2d_tiled_aux(fmap* input_features)
{
  // Output Stationary without padding and time logging.
  // 
  // @param input_features: [fmap] input tile patch for optimized 2d convolutions
  // @return output_features: [fmap] output tile patch after convolution

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
  // input_features = padding_2d(input_features, Px, Py);
  
  // Allocate output fmap
  fmap* output_features = (fmap*) malloc(sizeof(new_tensor(N, M, E, F)));
  *output_features      = new_tensor(N, M, E, F);
  
  // cast all data into easily interpretable form
  DATA (*temp)[M][E][F]         = (DATA (*)[M][E][F])output_features->data;         // N x M x E x F
  DATA (*temp_weights)[C][R][S] = (DATA (*)[C][R][S])weights;               // M x C x R x S
  DATA (*temp_inputs)[C][H][W]  = (DATA (*)[C][H][W])input_features->data;   // N x C x H x W

  __m256 mm_inputs, mm_weights, partials, partials_;
  __m256i mask;

  DATA partials_array[8];
  DATA zero[8];
  for(int i=0; i<8; i++)zero[i]=0;
  
  int start_idx;

  for(int n=0; n<N; n++){
    // For all filters
    for(int m=0; m<M; m++){
      // For each output point
      for(int x=0; x<F; x+=1)for(int y=0; y<E; y+=1)for(int c=0; c<C; c++){

        temp[n][m][y][x] = 0;
        partials         = _mm256_load_ps((float const*) &zero);

        for(int i=0; i<R/8 + 1; i++){
          start_idx = i*8 - R;
          mask      = _mm256_set_epi32(start_idx+7, start_idx+6, start_idx+5, start_idx+4, start_idx+3, start_idx+2, start_idx+1, start_idx);
            
          for(int j=0; j<S; j++){
            mm_inputs  = _mm256_maskload_ps((float const*) &temp_inputs[n][c][Sy*y+j][i*8], mask);
            mm_weights = _mm256_maskload_ps((float const*) &temp_weights[m][c][j][i*8], mask);

            partials_  = _mm256_mul_ps(mm_inputs, mm_weights);
            partials   = _mm256_add_ps(partials, partials_);
          }

          _mm256_storeu_ps((float*) &partials_array, partials);
          for(int k=0; k<8; k++){
            temp[n][m][y][x] += partials_array[k];
          }

        }
      }
    }
  }
  
  // Free inputs
  input_features =  NULL;
  free(input_features);
  
  return output_features;
}

fmap* Convolution::conv_2d(fmap* input_features)
{
  // Naive implimentation of 2D convolution with 7 nested loops
  // 
  // @param input_features: [fmap] input activations for 2d convolutions
  // @return output_features: [fmap] output activations after convolution

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
  // Py is padding 

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
  *output_features      = new_tensor(N, M, E, F);
  
  // cast all data into easily interpretable form
  DATA (*temp)[M][E][F]         = (DATA (*)[M][E][F])output_features->data;         // N x M x E x F
  DATA (*temp_weights)[C][R][S] = (DATA (*)[C][R][S])weights;               // M x C x R x S
  DATA (*temp_inputs)[C][H][W]  = (DATA (*)[C][H][W])input_features->data;   // N x C x H x W
  
  // For all batches
  for(int n=0; n<N; n++){
    // For all filters
    for(int m=0; m<M; m++){
      // For each output point
      for(int x=0; x<F; x+=1)for(int y=0; y<E; y+=1)for(int c=0; c<C; c++){
        temp[n][m][y][x] = 0;
        for(int j=0; j<S; j++){
          for(int i=0; i<R; i++){
            temp[n][m][y][x] += temp_inputs[n][c][Sy*y+j][Sx*x+i] * temp_weights[m][c][j][i];
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
  // Output stationary dataflow for 2D convolution optimized using AVX
  // instructions.
  // 
  // @param input_features:   [fmap] input activations for optimized 2d convolutions
  // @return output_features: [fmap] output activations after convolution

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
  *output_features      = new_tensor(N, M, E, F);
  
  // cast all data into easily interpretable form
  DATA (*temp)[M][E][F]         = (DATA (*)[M][E][F])output_features->data;         // N x M x E x F
  DATA (*temp_weights)[C][R][S] = (DATA (*)[C][R][S])weights;               // M x C x R x S
  DATA (*temp_inputs)[C][H][W]  = (DATA (*)[C][H][W])input_features->data;   // N x C x H x W

  __m256 mm_inputs, mm_weights, partials, partials_;
  __m256i mask;
  DATA partials_array[8];
  DATA zero[8];
  for(int i=0; i<8; i++)zero[i]=0;

  int start_idx;
  
  for(int n=0; n<N; n++){
    for(int c=0; c<C; c++){
      for(int y=0; y<E; y++){
        for(int x=0; x<F; x++){
          for(int j=0; j<S; j++){
            for(int i=0; i<R/8+1; i++){
              start_idx = i*8 - R;
              mask      = _mm256_set_epi32(start_idx+7, start_idx+6, start_idx+5, start_idx+4, start_idx+3, start_idx+2, start_idx+1, start_idx);
              mm_inputs = _mm256_maskload_ps((float const*) &temp_inputs[n][c][Sy*y+j][i*8], mask);
             
              for(int m=0; m<M; m++){                
                mm_weights = _mm256_maskload_ps((float const*) &temp_weights[m][c][j][i*8], mask);
                partials   = _mm256_mul_ps(mm_inputs, mm_weights);

                _mm256_maskstore_ps((float*) &partials_array, mask, partials);
                for(int k=0; k<8; k++){
                  temp[n][m][y][x] += partials_array[k];
                }
              }
            }
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

fmap* Convolution::conv2d_OS(fmap* input_features)
{

  // Output stationary dataflow for 2D convolution optimized using AVX
  // instructions.
  // 
  // @param input_features:   [fmap] input activations for optimized 2d convolutions
  // @return output_features: [fmap] output activations after convolution


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
  *output_features      = new_tensor(N, M, E, F);
  
  // cast all data into easily interpretable form
  DATA (*temp)[M][E][F]         = (DATA (*)[M][E][F])output_features->data;         // N x M x E x F
  DATA (*temp_weights)[C][R][S] = (DATA (*)[C][R][S])weights;               // M x C x R x S
  DATA (*temp_inputs)[C][H][W]  = (DATA (*)[C][H][W])input_features->data;   // N x C x H x W

  __m256 mm_inputs, mm_weights, partials, partials_;
  __m256i mask;
  DATA partials_array[8];
  DATA zero[8];
  for(int i=0; i<8; i++)zero[i]=0;

  int start_idx;

  for(int n=0; n<N; n++){
    // For all filters
    for(int m=0; m<M; m++){
      // For each output point
      for(int x=0; x<F; x+=1)for(int y=0; y<E; y+=1)for(int c=0; c<C; c++){
        temp[n][m][y][x] = 0;
        partials         = _mm256_load_ps((float const*) &zero);

        for(int i=0; i<R/8 + 1; i++){
          start_idx = i*8 - R;
          mask      = _mm256_set_epi32(start_idx+7, start_idx+6, start_idx+5, start_idx+4, start_idx+3, start_idx+2, start_idx+1, start_idx);
            
          for(int j=0; j<S; j++){
            mm_inputs  = _mm256_maskload_ps((float const*) &temp_inputs[n][c][Sy*y+j][i*8], mask);
            mm_weights = _mm256_maskload_ps((float const*) &temp_weights[m][c][j][i*8], mask);
            partials_  = _mm256_mul_ps(mm_inputs, mm_weights);
            partials   = _mm256_add_ps(partials, partials_);
          }

          _mm256_storeu_ps((float*) &partials_array, partials);

          for(int k=0; k<8; k++){
            temp[n][m][y][x] += partials_array[k];
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

fmap* Convolution::conv2d_WS(fmap* input_features)
{
  // Weight stationary dataflow for 2D convolution optimized using AVX
  // instructions.
  // 
  // @param input_features:   [fmap] input activations for optimized 2d convolutions
  // @return output_features: [fmap] output activations after convolution

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
  *output_features      = new_tensor(N, M, E, F);
  
  // cast all data into easily interpretable form
  DATA (*temp)[M][E][F]         = (DATA (*)[M][E][F])output_features->data;         // N x M x E x F
  DATA (*temp_weights)[C][R][S] = (DATA (*)[C][R][S])weights;               // M x C x R x S
  DATA (*temp_inputs)[C][H][W]  = (DATA (*)[C][H][W])input_features->data;   // N x C x H x W

  __m256 mm_inputs, mm_weights, partials, partials_;
  __m256i mask;

  DATA partials_array[8];
 
  int start_idx;

  for(int m=0; m<M; m++){
    for(int j=0; j<S; j++){
      for(int c=0; c<C; c++){
        for(int i=0; i<R/8+1; i++){
          start_idx  = i*8 - R;
          mask       = _mm256_set_epi32(start_idx+7, start_idx+6, start_idx+5, start_idx+4, start_idx+3, start_idx+2, start_idx+1, start_idx);
          mm_weights = _mm256_maskload_ps((float const*) &temp_weights[m][c][j][i*8], mask);

          for(int n=0; n<N; n++){
            for(int y=0; y<E; y++)for(int x=0; x<F; x++){
              mm_inputs = _mm256_maskload_ps((float const*) &temp_inputs[n][c][Sy*y+j][i*8], mask);
              partials  = _mm256_mul_ps(mm_inputs, mm_weights);

              _mm256_maskstore_ps((float*) &partials_array, mask, partials);

              for(int k=0; k<8; k++){
                temp[n][m][y][x] += partials_array[k];
              }
            }
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

fmap* Convolution::conv2d_optimized(fmap* input_features)
{
  // Output stationary dataflow for 2D convolution optimized using AVX
  // instructions.
  // 
  // @param input_features:   [fmap] input activations for optimized 2d convolutions
  // @return output_features: [fmap] output activations after convolution

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
  *output_features      = new_tensor(N, M, E, F);
  
  // cast all data into easily interpretable form
  DATA (*temp)[M][E][F]         = (DATA (*)[M][E][F])output_features->data;         // N x M x E x F
  DATA (*temp_weights)[C][R][S] = (DATA (*)[C][R][S])weights;               // M x C x R x S
  DATA (*temp_inputs)[C][H][W]  = (DATA (*)[C][H][W])input_features->data;   // N x C x H x W
  
  __m256 mm_inputs, mm_weights, partials, partials_;
  __m256i mask;
  DATA partials_array[8];
  DATA zero[8];
  for(int i=0; i<8; i++)zero[i]=0;
  
  int start_idx;
  
  for(int n=0; n<N; n++){
    // For all filters
    for(int m=0; m<M; m++){
      // For each output point
      for(int x=0; x<F; x+=1)for(int y=0; y<E; y+=1)for(int c=0; c<C; c++){
        temp[n][m][y][x] = 0;
        partials         = _mm256_load_ps((float const*) &zero);

        for(int i=0; i<R/8 + 1; i++){
          start_idx = i*8 - R;
          mask      = _mm256_set_epi32(start_idx+7, start_idx+6, start_idx+5, start_idx+4, start_idx+3, start_idx+2, start_idx+1, start_idx);
            
          for(int j=0; j<S; j++){
            mm_inputs  = _mm256_maskload_ps((float const*) &temp_inputs[n][c][Sy*y+j][i*8], mask);
            mm_weights = _mm256_maskload_ps((float const*) &temp_weights[m][c][j][i*8], mask);
            partials_  = _mm256_mul_ps(mm_inputs, mm_weights);
            partials   = _mm256_add_ps(partials, partials_);          
          }

          _mm256_storeu_ps((float*) &partials_array, partials);

          for(int k=0; k<8; k++){
            temp[n][m][y][x] += partials_array[k];
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
  // Naive implimentation of a fully connected layer.
  // 
  // @param input_features:   [fmap] input activations for FC layers
  // @return output_features: [fmap] output activations 
  
  clock_t start, end;
  start = clock();

  // M input size
  // L output size
  // output L = M*weights

  // Allocate the memory for outputs
  int N                 = input_features->dim1;
  fmap* output_features = (fmap*) malloc(sizeof(new_tensor(N, L, 1, 1)));
  *output_features      = new_tensor(N, L, 1, 1);
  
  // cast all data into easily interpretable form
  DATA (*temp)[L]         = (DATA (*)[L])output_features->data;
  DATA (*temp_weights)[L] = (DATA (*)[L])weights;
  DATA (*temp_inputs)[M]  = (DATA (*)[M])input_features->data;

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
  // Implimentation of a fully connected layer optimized using AVX instructions
  // 
  // @param input_features:   [fmap] input activations for FC layers
  // @return output_features: [fmap] output activations 
  
  clock_t start, end;
  start = clock();

  // M input size
  // L output size
  // output L = M*weights

  // Allocate the memory for outputs
  int N                 = input_features->dim1;
  fmap* output_features = (fmap*) malloc(sizeof(new_tensor(N, L, 1, 1)));
  *output_features      = new_tensor(N, L, 1, 1);
  
  // cast all data into easily interpretable form
  DATA (*temp)[L]         = (DATA (*)[L])output_features->data;
  DATA (*temp_weights)[L] = (DATA (*)[L])weights;
  DATA (*temp_inputs)[M]  = (DATA (*)[M])input_features->data;

  // Computation
  __m256 mm_inputs, mm_weights, partials, partials_;
  DATA weights_array[8], input_array[8];
  DATA zero[8];
  for(int i=0; i<8; i++)zero[i]=0;
 
  float partials_array[8];
  
  for(int n=0; n<N; n++){  
    for(int l=0; l<L; l++){
      temp[n][l] = 0;
      partials   = _mm256_loadu_ps((float const*) &zero);

      for(int m=0; m<int(M/8); m++){
        // Masked load
        for(int i=0; i<8; i++){
          weights_array[i] = temp_weights[m+i][l];
          input_array[i]   = temp_inputs[n][m+i];
        }
        mm_inputs  = _mm256_loadu_ps((float const*) &input_array);
        mm_weights = _mm256_loadu_ps((float const*) &weights_array);
        partials_  = _mm256_mul_ps(mm_inputs, mm_weights);
        partials   = _mm256_add_ps(partials, partials_);
      }

      // Manually zero masked
      int m = int(M/8);
     
      for(int i=0; i<8; i++){
        weights_array[i] = 0;
        input_array[i]   = 0;
      }

      for(int i=0; i<M%8; i++){
        weights_array[i] = temp_weights[8*m+i][l];
        input_array[i]   = temp_inputs[n][8*m+i];
      }
      mm_inputs  = _mm256_loadu_ps((float const*) &input_array);
      mm_weights = _mm256_loadu_ps((float const*) &weights_array);
      partials_  = _mm256_mul_ps(mm_inputs, mm_weights);
      partials   = _mm256_add_ps(partials, partials_);
      _mm256_storeu_ps((float*) &partials_array, partials);

      for(int i=0; i<M%8; i++){
        temp[n][l] += partials_array[i];
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
  // Inplace implimentatino of ReLU function on multidimensional tensor.
  // 
  // @param input_features: [fmap] input activations for ReLU

  // Get input dimensions
  int dim1 = input_features->dim1;
  int dim2 = input_features->dim2;
  int dim3 = input_features->dim3;
  int dim4 = input_features->dim4;

  DATA (*temp) = (DATA (*))input_features->data;  
  int size     = dim1*dim2*dim3*dim4;

  for(int i=0; i<size; i++){
    if(temp[i]<=0){
      temp[i] = 0;
    }
  }
}

fmap* padding_2d(fmap* input_features, int Px, int Py){
  
  // Implimentation of 2D zero padding for convolutions
  // 
  // @param input_features: [fmap] input maps in NCHW format
  // @param Px: [int] Pad size in X dimension
  // @Param Py: [int] Pad size in Y dimension
  // @return output_features: [fmap] Padded outputs
  
  // Calculate dimensions
  int N = input_features->dim1;
  int C = input_features->dim2;
  int H = input_features->dim3;
  int W = input_features->dim4;
  int E = int(H + Py*2);
  int F = int(W + Px*2);

  // Allocate output fmap
  fmap* output_features = (fmap*) malloc(sizeof(new_tensor(N, C, E, F)));
  *output_features      = new_tensor(N, C, E, F);

  // cast all data into easily interpretable form
  DATA (*temp)[C][E][F]        = (DATA (*)[C][E][F])output_features->data;         // N x M x E x F
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

  // Implimentation of Maxpooling 2D operation.
  // 
  // @param input_features: [fmap] input activations
  // @param R: Pool window size in Y dimensions
  // @param S: Pool window size in X dimensions

  // Calculate output size
  int N = input_features->dim1;
  int C = input_features->dim2;
  int H = input_features->dim3;
  int W = input_features->dim4;
  int E = int((H - S)/Sy + 1);
  int F = int((W - R)/Sx + 1);

  // Allocate output fmap
  fmap* output_features = (fmap*) malloc(sizeof(new_tensor(N, C, E, F)));
  *output_features      = new_tensor(N, C, E, F);
  
  // cast all data into easily interpretable form
  DATA (*temp)[C][E][F]        = (DATA (*)[C][E][F])output_features->data;         // N x M x E x F
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
  temp = conv_layers[0]->conv2d_optimized(temp);
  relu(temp);
  temp = maxpool_2d(temp, 3, 3, 2, 2);
  // temp = conv_layers[1]->conv2d_tiled(temp, 28);
  temp = conv_layers[1]->conv2d_optimized(temp);
  relu(temp);
  temp = maxpool_2d(temp, 3, 3, 2, 2);
  temp = conv_layers[2]->conv2d_optimized(temp);
  relu(temp);
  temp = conv_layers[3]->conv2d_optimized(temp);
  relu(temp);
  temp = conv_layers[4]->conv2d_optimized(temp);
  relu(temp);
  temp = maxpool_2d(temp, 3, 3, 2, 2);

  int lin_dim = temp->dim2 * temp->dim3 * temp->dim4;
  temp->dim2 = lin_dim;
  temp->dim3 = temp->dim4 = 1;

  temp = linear_layers[0]->linear_optimized(temp);
  relu(temp);
  temp = linear_layers[1]->linear_optimized(temp);
  relu(temp);
  temp = linear_layers[2]->linear_optimized(temp);
  relu(temp);

  end = clock();

  exec_time = double(end-start) / double(CLOCKS_PER_SEC);
  return temp;
}
