/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer
 *****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer
 *********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */

// modified from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
//#include <torch/library.h>
#include <THC/THCAtomics.cuh>

#include "cuda_helpers.h"
#include "deform_conv2d.h"


const int kMaxParallelImgs = 32;

inline unsigned int GET_THREADS() {
#ifdef __HIP_PLATFORM_HCC__
  return 256;
#endif
  if (at::cuda::getCurrentDeviceProperties()->major >= 6) {
    return 1024;
  }
  return 512;
}

inline unsigned int GET_BLOCKS(
    const unsigned int THREADS,
    const unsigned int N) {
  unsigned int kMaxGridNum =
      at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  return std::min(kMaxGridNum, (N + THREADS - 1) / THREADS);
}

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(
    const scalar_t* in,
    int height,
    int width,
    scalar_t h,
    scalar_t w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = in[h_low * width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = in[h_low * width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = in[h_high * width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = in[h_high * width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__global__ void deformable_im2col_cuda_kernel(
    int n,
    const scalar_t* input_ptr,
    const scalar_t* offset_ptr,
    const scalar_t* mask_ptr,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int batch_sz,
    int n_in_channels,
    int n_offset_grps,
    int out_h,
    int out_w,
    bool use_mask,
    scalar_t* columns_ptr) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int out_x = index % out_w;
    const int out_y = (index / out_w) % out_h;
    const int out_b = (index / (out_w * out_h)) % batch_sz;
    const int in_c = index / (out_w * out_h * batch_sz);
    const int out_c = in_c * weight_h * weight_w;

    int c_per_offset_grp = n_in_channels / n_offset_grps;
    const int grp_idx = in_c / c_per_offset_grp;

    columns_ptr +=
        (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) +
         out_y * out_w + out_x);

    input_ptr +=
        (out_b * (n_in_channels * height * width) + in_c * (height * width));

    offset_ptr += (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w *
        out_h * out_w;

    if (use_mask) {
      mask_ptr += (out_b * n_offset_grps + grp_idx) * weight_h * weight_w *
          out_h * out_w;
    }

    for (int i = 0; i < weight_h; ++i) {
      for (int j = 0; j < weight_w; ++j) {
        const int mask_idx = i * weight_w + j;
        const int offset_idx = 2 * mask_idx;

        scalar_t mask_value = 1;
        if (use_mask) {
          mask_value =
              mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x];
        }

        const scalar_t offset_h =
            offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
        const scalar_t offset_w = offset_ptr
            [(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];
        const scalar_t y =
            (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
        const scalar_t x =
            (out_x * stride_w - pad_w) + j * dilation_w + offset_w;
        *columns_ptr =
            mask_value * bilinear_interpolate(input_ptr, height, width, y, x);
        columns_ptr += batch_sz * out_h * out_w;
      }
    }
  }
}

void deformable_im2col_cuda_app(
    const at::Tensor& input,
    const at::Tensor& data_offset,
    const at::Tensor& data_mask,
    int n_in_channels,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int out_h,
    int out_w,
    int parallel_imgs,
    int deformable_group,
    bool use_mask,
    at::Tensor data_col) {
  int num_kernels = n_in_channels * out_h * out_w * parallel_imgs;

  const unsigned int threads = GET_THREADS();
  const unsigned int blocks = GET_BLOCKS(threads, num_kernels);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "deformable_im2col_cuda_app", ([&] {
        deformable_im2col_cuda_kernel<<<blocks, threads>>>(
            num_kernels,
            input.data_ptr<scalar_t>(),
            data_offset.data_ptr<scalar_t>(),
            data_mask.data_ptr<scalar_t>(),
            height,
            width,
            weight_h,
            weight_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            parallel_imgs,
            n_in_channels,
            deformable_group,
            out_h,
            out_w,
            use_mask,
            data_col.data_ptr<scalar_t>());
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_im2col_cuda_app: %s\n", cudaGetErrorString(err));
  }
}

static int get_greatest_divisor_below_bound(int n, int bound) {
  for (int k = bound; k > 1; --k) {
    if (n % k == 0) {
      return k;
    }
  }
  return 1;
}

at::Tensor deform_conv2d_cuda_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t n_weight_grps,
    int64_t n_offset_grps,
    bool use_mask) {
  at::Tensor input_c = input.contiguous();
  at::Tensor offset_c = offset.contiguous();
  at::Tensor weight_c = weight.contiguous();
  at::Tensor mask_c = mask.contiguous();
  at::Tensor bias_c = bias.contiguous();

  TORCH_CHECK(input_c.ndimension() == 4);
  TORCH_CHECK(offset_c.ndimension() == 4);
  TORCH_CHECK(!use_mask || mask_c.ndimension() == 4);
  TORCH_CHECK(weight_c.ndimension() == 4);
  TORCH_CHECK(input_c.is_cuda(), "input must be a CUDA tensor");

  at::DeviceGuard guard(input_c.device());

  int batch_sz = input_c.size(0);
  int in_channels = input_c.size(1);
  int in_h = input_c.size(2);
  int in_w = input_c.size(3);

  int n_parallel_imgs =
      get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

  int out_channels = weight_c.size(0);
  int weight_h = weight_c.size(2);
  int weight_w = weight_c.size(3);

  int ker_h = dilation_h * (weight_h - 1) + 1;
  int ker_w = dilation_w * (weight_w - 1) + 1;
  int out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1;
  int out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1;

  TORCH_CHECK(
      weight_h > 0 && weight_w > 0,
      "weight_h: ",
      weight_h,
      " weight_w: ",
      weight_w);
  TORCH_CHECK(
      stride_h > 0 && stride_w > 0,
      "stride_h: ",
      stride_h,
      " stride_w: ",
      stride_w);
  TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "pad_h: ", pad_h, " pad_w: ", pad_w);
  TORCH_CHECK(
      dilation_h > 0 && dilation_w > 0,
      "dilation_h: ",
      dilation_h,
      " dilation_w: ",
      dilation_w);

  TORCH_CHECK(weight_c.size(1) * n_weight_grps == input_c.size(1));
  TORCH_CHECK(weight_c.size(0) % n_weight_grps == 0);
  TORCH_CHECK(
      (offset_c.size(1) == n_offset_grps * 2 * weight_h * weight_w),
      "offset.shape[1] is not valid: got: ",
      offset_c.size(1),
      " expected: ",
      n_offset_grps * 2 * weight_h * weight_w);
  TORCH_CHECK(
      (!use_mask || mask_c.size(1) == n_offset_grps * weight_h * weight_w),
      "mask.shape[1] is not valid: got: ",
      mask_c.size(1),
      " expected: ",
      n_offset_grps * weight_h * weight_w);
  TORCH_CHECK(input_c.size(1) % n_offset_grps == 0);

  TORCH_CHECK(
      (offset_c.size(0) == input_c.size(0)), "invalid batch size of offset");
  TORCH_CHECK(
      (offset_c.size(2) == out_h && offset_c.size(3) == out_w),
      "offset output dims: (",
      offset_c.size(2),
      ", ",
      offset_c.size(3),
      ") - ",
      "computed output dims: (",
      out_h,
      ", ",
      out_w,
      ")");
  TORCH_CHECK(
      (mask_c.size(0) == input_c.size(0)), "invalid batch size of mask");
  TORCH_CHECK(
      (!use_mask || (mask_c.size(2) == out_h && mask_c.size(3) == out_w)),
      "mask output dims: (",
      mask_c.size(2),
      ", ",
      mask_c.size(3),
      ") - ",
      "computed output dims: (",
      out_h,
      ", ",
      out_w,
      ")");
  TORCH_CHECK(
      out_h > 0 && out_w > 0,
      "Calculated output size too small - out_h: ",
      out_h,
      " out_w: ",
      out_w);

  auto out =
      at::zeros({batch_sz, out_channels, out_h, out_w}, input_c.options());
  if (batch_sz == 0) {
    return out;
  }

  // Separate batches into blocks
  out = out.view(
      {batch_sz / n_parallel_imgs,
       n_parallel_imgs,
       out_channels,
       out_h,
       out_w});
  input_c = input_c.view(
      {batch_sz / n_parallel_imgs, n_parallel_imgs, in_channels, in_h, in_w});

  offset_c = offset_c.view(
      {batch_sz / n_parallel_imgs,
       n_parallel_imgs,
       n_offset_grps * 2 * weight_h * weight_w,
       out_h,
       out_w});

  if (use_mask) {
    mask_c = mask_c.view(
        {batch_sz / n_parallel_imgs,
         n_parallel_imgs,
         n_offset_grps * weight_h * weight_w,
         out_h,
         out_w});
  }

  at::Tensor out_buf = at::zeros(
      {batch_sz / n_parallel_imgs,
       out_channels,
       n_parallel_imgs * out_h,
       out_w},
      out.options());

  // Separate channels into convolution groups
  out_buf = out_buf.view(
      {out_buf.size(0),
       n_weight_grps,
       out_buf.size(1) / n_weight_grps,
       out_buf.size(2),
       out_buf.size(3)});
  weight_c = weight_c.view(
      {n_weight_grps,
       weight_c.size(0) / n_weight_grps,
       weight_c.size(1),
       weight_c.size(2),
       weight_c.size(3)});

  // Sample points and perform convolution
  auto columns = at::zeros(
      {in_channels * weight_h * weight_w, n_parallel_imgs * out_h * out_w},
      input_c.options());
  for (int b = 0; b < batch_sz / n_parallel_imgs; b++) {
    deformable_im2col_cuda_app(
        input_c[b],
        offset_c[b],
        mask_c[b],
        in_channels,
        in_h,
        in_w,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        out_h,
        out_w,
        n_parallel_imgs,
        n_offset_grps,
        use_mask,
        columns);

    columns = columns.view(
        {n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});
    for (int g = 0; g < n_weight_grps; g++) {
      out_buf[b][g] = out_buf[b][g]
                          .flatten(1)
                          .addmm_(weight_c[g].flatten(1), columns[g])
                          .view_as(out_buf[b][g]);
    }
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
  }

  out_buf = out_buf.view(
      {batch_sz / n_parallel_imgs,
       out_channels,
       n_parallel_imgs,
       out_h,
       out_w});
  out_buf.transpose_(1, 2);
  out.copy_(out_buf);
  out = out.view({batch_sz, out_channels, out_h, out_w});

  return out + bias_c.view({1, out_channels, 1, 1});
}
