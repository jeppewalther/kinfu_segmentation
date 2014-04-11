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
#include <iostream>
namespace pcl
{
  namespace device
  {
    struct IntersectionDetection
    {
      enum { CTA_SIZE_X = 32, CTA_SIZE_Y = 8 };

      int rows, cols;

      PtrStep<float> fg_vmap;
      PtrStep<float> bg_vmap;

      float intersection_threshold;

      PtrStepSz<unsigned int> icp_outliers_filtered;
      mutable PtrStepSz<unsigned char> intersection_map;

      __device__ __forceinline__ float3
      get_vertex (const PtrStep<float>& map, const int& x, const int& y) const
      {
        return (make_float3( map.ptr (y)[x],
                            map.ptr (y + rows)[x],
                            map.ptr (y + 2 * rows)[x]));
      }

      __device__ __forceinline__ void
      operator() () const
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= cols || y >= rows)
          return;

        float3 offset = get_vertex(bg_vmap, x, y) - get_vertex(fg_vmap, x, y);

        intersection_map.ptr (y)[x] = dot (offset, offset) < (intersection_threshold) ? 2 : 0;
      }
    };

    __global__ void
    intersectionDetection (const IntersectionDetection id) {
      id();
    }
  }
}

void
pcl::device::detectIntersections (const MapArr& bg_vmap,
                                  const MapArr& fg_vmap,
                                  DeviceArray2D<unsigned int>& icp_outliers_filtered,
                                  DeviceArray2D<unsigned char>& intersection_map,
                                  float intersection_threshold,
                                  cudaStream_t& stream
                                  )
{
  IntersectionDetection id;

  id.rows = intersection_map.rows();
  id.cols = intersection_map.cols();

  id.fg_vmap = fg_vmap;
  id.bg_vmap = bg_vmap;
  id.intersection_map = intersection_map;
  id.icp_outliers_filtered = icp_outliers_filtered;
  id.intersection_threshold = intersection_threshold*intersection_threshold;

  dim3 block (IntersectionDetection::CTA_SIZE_X, IntersectionDetection::CTA_SIZE_Y);
  dim3 grid (divUp (id.cols, block.x), divUp (id.rows, block.y));

  intersectionDetection<<<grid, block, 0, stream>>>(id);
  cudaSafeCall (cudaGetLastError ());
  cudaSafeCall (cudaDeviceSynchronize ());
}

namespace pcl
{
  namespace device
  {
    __global__ void calculateCenterOfMassKernel(
      const PtrStep<unsigned int> index_map,
      const PtrStep<float> vmap,
      PtrSz<float> dst,
      unsigned int max_blob_count,
      unsigned int rows,
      unsigned int cols
      )
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= cols || y >= rows) return;

      unsigned int index = index_map.ptr (y)[x];

      if(index > 0 && index <= max_blob_count)
      {
        float3 v;
        v.x = vmap.ptr (y)[x];
        v.y = vmap.ptr (y + rows)[x];
        v.z = vmap.ptr (y + 2 * rows)[x];

        if (!isnan (v.x))
        {
          atomicAdd(&dst.data[(index - 1) * 3], v.x);
          atomicAdd(&dst.data[(index - 1) * 3 + 1], v.y);
          atomicAdd(&dst.data[(index - 1) * 3 + 2], v.z);
        }
      }
    }
  }
}


void
pcl::device::calculateCenterOfMass(const DeviceArray2D<unsigned int>& index_map,
                                   const MapArr& vmap,
                                   const DeviceArray<uint2>& device_blob_sizes,
                                   DeviceArray<float>& device_dst,
                                   unsigned int blob_count,
                                   std::vector<float3>& intersections_
                                   )
{
  device_dst.create(blob_count*3);
  std::vector<float> host_dst(blob_count*3, 0.0f);
  device_dst.upload(host_dst);


  dim3 block (32,8);
  dim3 grid (divUp (index_map.cols(), block.x), divUp (index_map.rows(), block.y));
  calculateCenterOfMassKernel<<<grid, block>>>(index_map, vmap, device_dst, blob_count, index_map.rows(), index_map.cols());

  cudaSafeCall (cudaGetLastError ());
  cudaSafeCall (cudaDeviceSynchronize ());

  std::vector<uint2> host_blob_sizes(blob_count);
  device_blob_sizes.download(host_blob_sizes);
  device_dst.download(host_dst);

  intersections_.clear();

  {
    std::vector<uint2> host_blob_sizes(device_blob_sizes.size());
    device_blob_sizes.download(host_blob_sizes);

    for (unsigned int i = 0; i < device_blob_sizes.size(); ++i)
    {
      unsigned int size = host_blob_sizes[i].y;

      if (0 == size)
        continue;

      float3 x = make_float3(host_dst[3*i] / size, host_dst[3*i + 1] / size, host_dst[3*i + 2] / size);
      intersections_.push_back(x);

      //std::cout << i + 1 << ": [" << x.x << ", " << x.y << ", " << x.z << "]" << std::endl;
    }
  }

}




