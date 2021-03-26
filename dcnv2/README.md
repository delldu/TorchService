# README

## 1. C++ Source Code
Good news: for latest pytorch,  DCNv2 exists in torchvision;

Bad news: for pytorch 1.5.0 -- 1.6.0, there is only DCNv1 in torchvision, so we must develop DCNv2 !!!

This version mainly comes from https://github.com/pytorch/vision version 0.9.0, interface has been modified for video zoom.

## 2. Import Things
### 2.1 Function Limitation
We only keep `Forward`, delete `Backward` for simple.
### 2.2 Building Makefile
Makefile should be modified according your environment.
1. Fore pytorch, you must set `-D_GLIBCXX_USE_CXX11_ABI=0`
2. For libtorch, you must `-D_GLIBCXX_USE_CXX11_ABI=1`
3. GPU Setting:  sm_75 is good for RTX 2080TI.
