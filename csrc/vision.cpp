// Copyright (c) Samson Wang. All Rights Reserved.
#include "DepthWiseConv2d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("depth_wise_conv2d", &DepthWiseConv2d_forward, "DepthWiseConv2d_forward");
  m.def("depth_wise_conv2d_back_weight", &DepthWiseConv2d_backward_weight, "DepthWiseConv2d_backward_weight");
  m.def("depth_wise_conv2d_back_input", &DepthWiseConv2d_backward_input, "DepthWiseConv2d_backward_input");
}
