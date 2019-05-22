// Copyright (c) syshen. and its affiliates. All Rights Reserved.
#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


at::Tensor soft_nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold,
               float sigma,
               int mode=0) {

  if (dets.type().is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    if (dets.numel() == 0)
      return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
    return soft_nms_cuda(b, threshold, sigma, mode);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  at::Tensor result = soft_nms_cpu(dets, scores, threshold, sigma, mode);
  return result;
}