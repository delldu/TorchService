import torch
import pdb

print("Loading libdcnv2.so ...")
torch.ops.load_library("libdcnv2.so")
print("Load libdcnv2.so OK")

print("Show torch.ops.dcnv2.forward ...")
print(torch.ops.dcnv2.forward)
print("torch.ops.dcnv2.forward OK")


# input.size(), offset.size(), mask.size()
# (torch.Size([1, 64, 68, 120]), torch.Size([1, 144, 68, 120]), torch.Size([1, 72, 68, 120]), 
# self.weight.size(), self.bias.size()
# torch.Size([64, 64, 3, 3]), torch.Size([64]))

# C++ protype
# ----------------------------------------------- #
# at::Tensor deform_conv2d(
#     const at::Tensor& input,
#     const at::Tensor& weight,
#     const at::Tensor& bias,
#     const at::Tensor& offset,
#     const at::Tensor& mask,
#     int64_t stride_h,
#     int64_t stride_w,
#     int64_t pad_h,
#     int64_t pad_w,
#     int64_t dilation_h,
#     int64_t dilation_w,
#     int64_t groups)
# ----------------------------------------------- #

def dcnv2_forward(input, weight, bias, offset, mask,  
	stride_h=1, stride_w=1, pad_h=0, pad_w=0, dilation_h=1, dilation_w=1, groups=1):

	return torch.ops.dcnv2.forward(input, weight, bias, offset, mask, 
		stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups)
	

def test_dcnv2_forward():
	print("Test torch.ops.dcnv2.forward ...")
	input = torch.rand(32, 3, 10, 10)
	kh, kw = 3, 3
	weight = torch.rand(5, 3, kh, kw)
	# bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)
	bias = torch.zeros(weight.shape[0], device=input.device, dtype=input.dtype)

	offset = torch.rand(32, 2 * kh * kw, 8, 8)
	mask = torch.rand(32, kh * kw, 8, 8)

	out = dcnv2_forward(input, weight, bias, offset, mask)
	print(out.shape)
	# Shoule be: torch.Size([32, 5, 8, 8])
	assert (out.size() == torch.Size([32, 5, 8, 8])), "Output size should be: [32, 5, 8, 8]"
	print("Test torch.ops.dcnv2.forward OK")

test_dcnv2_forward()
