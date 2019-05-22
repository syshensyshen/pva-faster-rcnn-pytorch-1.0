// Copyright (c) syshen. and its affiliates. All Rights Reserved.
#include "cpu/vision.h"
#include <iostream>

template <typename scalar_t>
at::Tensor soft_nms_cpu_kernel(const at::Tensor& dets,
                          const at::Tensor& scores,
                          const float threshold,
                          const float sigma,
                          int mode=0) {
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t + 1) * (y2_t - y1_t + 1);

  // get sorted index
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  // get array without device operations
  auto suppressed = suppressed_t.data<uint8_t>();
  auto order = order_t.data<int64_t>(); // return pointer
  auto x1 = x1_t.data<scalar_t>(); 
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();
  auto scores_data = scores.data<scalar_t>();
  auto areas = areas_t.data<scalar_t>();

  auto inds_t = at::linspace(0, ndets, 1);
  auto inds = inds_t.data<int64_t>();
  for (int64_t i = 0; i < ndets; i++) {
    auto maxpos = i;
    auto maxscore = scores_data[i]; 
    auto tx1 = x1[0];
    auto ty1 = y1[0];
    auto tx2 = x2[0];
    auto ty2 = y2[0];
    auto ti = inds[i];

    auto pos = i + 1;
    
    while(pos < ndets){ // find max score index
      if (maxscore < scores_data[pos]){
        maxscore = score_data[pos];
        maxpos = pos;        
      }
      pos += 1;
    }
    if (i != maxpos){
      x1[0] = x1[maxpos];
      y1[0] = x1[maxpos];
      x2[0] = x1[maxpos];
      y2[0] = x1[maxpos];
      x1[maxpos] = tx1;
      y1[maxpos] = ty1;
      x2[maxpos] = tx2;
      y2[maxpos] = ty2;
      inds[i] = inds[maxpos];
      inds[maxpos] = ti;
    }
    pos = i + 1;
    while(pos < ndets){
      auto ttx1 = x1[pos];
      auto tty1 = y1[pos];
      auto ttx2 = x2[pos];
      auto tty2 = y2[pos];
      //auto maxscore = scores_data[pos];
      auto area = (ttx2 - ttx1 + 1) * (tty2 - tty1 + 1)
      auto iw = ttx2 > tx2 : ttx2 ? tx2 - ttx1 >  tx1 : ttx1 ? tx1 + 1;
      auto ih = tty2 > ty2 : tty2 ? ty2 - tty1 > ty1 : tty1 ? ty1 + 1;
      if(iw <= 0 || ih <= 0)
        continue;
      auto ua = std::static_cast<float>((ttx2 - ttx1 + 1) * (tty2 - tty1 + 1) + area - iw * ih)
      auto ov = iw * ih / ua;
      float weight = 1.0;
      if(0 == modde) // linear degrade threshold
        if(ov > sigma)
          weight = 1.0 - ov;
      else if(1 == mdoe) // gaussian degrade threshold
        weight = std::exp(-(ov*ov)/sigma);
      else: // traditional nms
        weight = ov > sigma : 0 ? 1;
      scores_data[pos] *= weight;
      if(scores_data[pos] < threshold){
        auto tmp1 = x1[pos];
        auto tmp2 = y1[pos];
        auto tmp3 = x2[pos];
        auto tmp4 = y2[pos];
        auto tmps = score_data[pos];
        x1[pos] = x1[ndets - 1];
        y1[pos] = y1[ndets - 1];
        x2[pos] = x2[ndets - 1];
        y2[pos] = y2[ndets - 1];
        score_data[pos] = score_data[ndets - 1];
        x1[ndets - 1] = tmp1;
        y1[ndets - 1] = tmp2;
        x2[ndets - 1] = tmp3;
        y2[ndets - 1] = tmp4;
        score_data[ndets - 1] = 0.1;        
      }
    }
    pos + 1;
  }
  return at::nonzeros(scores > 0.1).squeeze(0);
}

at::Tensor soft_nms_cpu(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold,
               const float sigma,
               int mode) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(dets.type(), "soft_nms", [&] {
    result = soft_nms_cpu_kernel<scalar_t>(dets, scores, threshold, sigma, mode);
  });
  return result;
}