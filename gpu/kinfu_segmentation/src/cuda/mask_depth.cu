/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "device.hpp"

using namespace pcl::device;

namespace pcl
{
  namespace device
  {
    template <bool invert> __global__ void
    maskDepthKernel(const PtrStep<ushort> depth, const PtrStepSz<unsigned int> mask, PtrStepSz<ushort> dst)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= dst.cols || y >= dst.rows)
        return;

      if (invert)
        dst.ptr (y)[x] = mask.ptr (y)[x] ? 0 : depth.ptr (y)[x];
      else
        dst.ptr (y)[x] = mask.ptr (y)[x] ? depth.ptr (y)[x] : 0;
    }
  }
}


void
pcl::device::maskDepth(const DepthMap& depth, const DeviceArray2D<unsigned int>& mask, PtrStepSz<ushort> dst, bool invert, cudaStream_t& stream)
{
  dim3 block(32, 8);
  dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

  if(invert)
  {
    maskDepthKernel<true><<<grid, block, 0>>>(depth, mask, dst);
  }
  else
  {
    maskDepthKernel<false><<<grid, block, 0>>>(depth, mask, dst);
  }

  cudaSafeCall (cudaGetLastError ());
}


namespace pcl
{
  namespace device
  {
    __global__ void
    compOverKernel(const PtrStepSz<uchar3> fg, const PtrStepSz<uchar3> bg, PtrStepSz<uchar3> dst)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= dst.cols || y >= dst.rows)
        return;

      const uchar3 pfg = fg.ptr (y)[x];
      const uchar3 pbg = bg.ptr (y)[x];

      if(0 == pfg.x + pfg.y + pfg.z)
      {
        dst.ptr (y)[x] = pbg;
      }else{
        dst.ptr (y)[x] = pfg;
      }

    }
  }
}


void
pcl::device::compOver (const DeviceArray2D<pcl::gpu::PixelRGB>& fg, const DeviceArray2D<pcl::gpu::PixelRGB>& bg, PtrStepSz<uchar3> dst)
{
  dim3 block(32, 8);
  dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

  compOverKernel<<<grid, block>>>(fg, bg, dst);

  cudaSafeCall (cudaGetLastError ());
}

