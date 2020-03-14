#include <stdint.h>

//The datatype of each element - 8 bit signed integer
typedef int32_t DATA;

//Data structure to hold input feature map / weight filters / output feature map
typedef struct _fmap{
  DATA* data;
  int dim1, dim2, dim3, dim4;
}fmap;

//Base class for each layer
class Layer
{
  public:
  double exec_time;
  DATA* weights;
};

class Convolution : public Layer
{
  public:
  int M, C, R, S, Sx, Sy, Px, Py;

  Convolution(int m, int c, int r, int s, int sx, int sy, int px, int py);
  ~Convolution();

  /*
    Each of the following convolution function should do the following things in the given order.
    1. Allocate memory for output feature map
    2. Perform computation
    3. Free memory allocated for input feature map
    4. Return the computed output feature map
  */

  fmap* conv_2d(fmap* input_features);//Baseline Convolution using AVX
  fmap* conv2d_IS(fmap* input_features);//Input stationary
  fmap* conv2d_OS(fmap* input_features);//Output stationary
  fmap* conv2d_WS(fmap* input_features);//Weight stationary
  fmap* conv2d_optimized(fmap* input_features);//Optimized convolution
};

class Linear : public Layer
{
  public:
  int M, L;

  Linear(int m, int l);
  ~Linear();
  /*
    Each of the following Linear function should do the following things in the given order.
    1. Allocate memory for output feature map
    2. Perform computation
    3. Free memory allocated for input feature map
    4. Return the computed output feature map
  */

  fmap* linear(fmap* input_features);
  fmap* linear_optimized(fmap* input_features);
};

class AlexNet : public Layer
{
  public:
  Convolution** conv_layers;
  Linear** linear_layers;

  AlexNet();
  fmap* forward_pass(fmap* input_features);
};

//For ReLU, no new memory allocation happens. Computation happens in-place.
void relu(fmap* input_features);

//For MaxPool2D, similar to CONV and FC, allocate memory for output feature map, perform computation
//and free input feature map. Finally, return the created output feature map.
fmap* maxpool_2d(fmap* input_features, int R, int S, int Sx, int Sy);
