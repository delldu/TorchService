#include "deform_conv2d.h"

#include <torch/types.h>

at::Tensor deform_conv2d(
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
    int64_t groups,
    int64_t offset_groups,
    bool use_mask) {
    if (input.is_cuda()) {
#ifdef WITH_CUDA
    return deform_conv2d_cuda_forward_kernel(
        input, weight, offset, mask, bias,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        groups, offset_groups, use_mask);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    } else {
        return deform_conv2d_cpu_forward_kernel(
            input, weight, offset, mask, bias,
            stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
            groups, offset_groups, use_mask);
    }
}


#include <torch/script.h>

at::Tensor dcnv2_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& offset,
    const at::Tensor& mask,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t deformable_groups)
{
    bool use_mask = true;
    int64_t groups;
    int64_t offset_groups;

    offset_groups = offset.size(1);
    offset_groups /= (2 * weight.size(2) * weight.size(3));

    // Fix Bug: set deformable_groups = 1 for video zoom !!!
    deformable_groups = 1;
    groups = input.size(1);
    groups /= weight.size(1);
    groups /= deformable_groups;

    return deform_conv2d(input, weight, offset, mask, bias,
            stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
            groups, offset_groups, use_mask);
}


static auto registry = torch::RegisterOperators("dcnv2::forward", &dcnv2_forward);

