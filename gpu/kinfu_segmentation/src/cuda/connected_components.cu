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
#include <algorithm>
#include <numeric>
//#include <pcl/gpu/utils/device/block.hpp>

using namespace pcl::device;


namespace pcl
{
  namespace device
  {
    __global__ void
    initializeConnectedComponentsKernel(
        const PtrStep<ushort> depth,
        const PtrStep<unsigned char> icp_outliers,
        PtrStep<unsigned int> blob_sizes,
        PtrStepSz<unsigned int> dst
      )
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= dst.cols || y >= dst.rows)
        return;

      unsigned char i = icp_outliers.ptr (y)[x];
      unsigned int d = depth.ptr (y)[x];

      // Filter error values from depth input
      if (0 == d || 0 == i)
      {
        dst.ptr (y)[x] = 0;
        blob_sizes.ptr (y)[x] = 0;
      }
      else
      {
        // Label entries with position in 2dArray
        dst.ptr (y)[x] = (y*dst.cols + x);
        blob_sizes.ptr (y)[x] = 1;
      }
    }
  }
}


namespace pcl
{
  namespace device
  {
    __global__ void
    linkConnectedComponentsKernel (
      unsigned int depth_threshold,
      const PtrStep<ushort> depth,
      PtrStepSz<unsigned int> dst,
      PtrSz<unsigned char> is_not_done
    )
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x < 1 || y < 1 || x >= dst.cols -1 || y >= dst.rows -1)
        return;

      unsigned int l = dst.ptr (y)[x];
      int d = depth.ptr (y)[x];

      //Scan for min neighbour - 4 connectivity
      if (l) // Skip if l = 0
      {
        unsigned int minl = UINT_MAX;

        //West
        unsigned int lw = dst.ptr (y)[x-1];
        int dw = depth.ptr (y)[x-1];
        if (lw && abs(d - dw) < depth_threshold) minl = lw;

        //East
        unsigned int le = dst.ptr (y)[x+1];
        int de = depth.ptr (y)[x+1];
        if(le && le < minl && abs(d - de) < depth_threshold) minl = le;

        //South
        unsigned int ls = dst.ptr (y-1)[x];
        int ds = depth.ptr (y-1)[x];
        if(ls && ls < minl && abs(d - ds) < depth_threshold) minl = ls;

        //North
        unsigned int ln = dst.ptr (y+1)[x];
        int dn = depth.ptr (y+1)[x];
        if(ln && ln < minl && abs(d - dn) < depth_threshold) minl = ln;

        if(minl<l)
        {
          unsigned int ll = dst.data[l];
          dst.data[l] = min(ll,minl);
          is_not_done.data[0] = 1;
        }
      }
    }
  }
}



namespace pcl
{
  namespace device
  {
    __global__ void
    relabelConnectedComponentsKernel(
      PtrStepSz<unsigned int> dst,
      PtrStepSz<unsigned int> blob_sizes
      )
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x < 1 || y < 1 || x >= dst.cols -1 || y >= dst.rows -1)
        return;

      unsigned int label = dst.ptr (y)[x];

      if (label)
      {
        unsigned int r = dst.data[label];
        while (r != label)
        {
          label = dst.data[r];
          r = dst.data[label];
        }
        dst.ptr (y)[x] = label;

        //Size Calculation
        unsigned int n = blob_sizes.ptr (y)[x];
        if(n && label != (y*dst.cols + x) )
        {
          atomicAdd(&blob_sizes.data[label],n);
          blob_sizes.ptr (y)[x] = 0;
        }
      }
    }
  }
}


namespace pcl
{
  namespace device
  {
    __global__ void
    filterConnectedComponentsKernel(PtrStepSz<unsigned int> input, PtrSz<uint2> targets)
    {
      extern __shared__ uint2 sdata[];
      unsigned int tid = threadIdx.x;
      unsigned int g_tid = blockDim.x * blockIdx.x + threadIdx.x;

      if (g_tid >= input.cols * input.rows)
        return;

      unsigned int blobId = input.data[g_tid];

      if (0 == tid)
      {
        for (unsigned int i = 0; i < targets.size; ++i)
        {
            sdata[i] = targets[i];
        }
      }

      __syncthreads();

      input.data[g_tid] = 0;

      if (!blobId)
        return;

      for (unsigned int i = 0; i < targets.size; ++i)
      {
        if (blobId == sdata[i].x)
        {
          input.data[g_tid] = i + 1;
        }
      }
    }

    template <typename T, typename R>
    struct FindLargestBlob
    {
      enum { CTA_SIZE_TOTAL = 640 };

      PtrSz<T> input;

      mutable PtrSz<R> intermediate_result;
      mutable PtrSz<R> result;
      mutable R blob_max;

      PtrSz<R> previous_result;

      __host__ __device__ __forceinline__  R
      max (const R& lhs, const R& rhs) const
      {
        return (rhs.y > lhs.y && rhs.y < blob_max.y ? rhs : lhs.y < blob_max.y ? lhs : make_pair(0, 0));
      }

      __host__ __device__ __forceinline__  void
      max (const volatile R& lhs, const volatile R& rhs, volatile R& result) const
      {
        result.x = (rhs.y > lhs.y && rhs.y < blob_max.y ? rhs.x : lhs.y < blob_max.y ? lhs.x : 0);
        result.y = (rhs.y > lhs.y && rhs.y < blob_max.y ? rhs.y : lhs.y < blob_max.y ? lhs.y : 0);
      }

      static __device__ __forceinline__ R
      make_pair(const unsigned int& tid, const unsigned int& value)
      {
        return (make_uint2(tid, value));
      }

      static __device__ __forceinline__ R
      make_pair(const unsigned int& tid, const uint2& value)
      {
        return (value);
      }

      template <unsigned int BS> __device__ __forceinline__ void
      reduce6 () const
      {
        unsigned int tid = threadIdx.x;
        extern __shared__ R __sdata[];

        R* sdata = (R*) __sdata;
        R max_id = make_pair(tid, 0);

        for (unsigned int i = blockIdx.x * (BS * 2) + tid; i < input.size;
            i += BS * 2 * gridDim.x)
        {
          max_id = max(max_id, max(
              make_pair(i, input.data[i]), make_pair(i + BS, input.data[i + BS])));
        }

        sdata[tid] = max_id;

        __syncthreads();

        if (BS >= 512) { if (tid < 256) { sdata[tid] = max_id =
            max(sdata[tid + 256], max_id); } __syncthreads(); }
        if (BS >= 256) { if (tid < 128) { sdata[tid] = max_id =
            max(sdata[tid + 128], max_id); } __syncthreads(); }
        if (BS >= 128) { if (tid <  64) { sdata[tid] = max_id =
            max(sdata[tid +  64], max_id); } __syncthreads(); }

        if (tid < 32)
        {
          volatile R* v_sdata = sdata;

          if (BS >=  64) max(v_sdata[tid], v_sdata[tid + 32], v_sdata[tid]);
          if (BS >=  32) max(v_sdata[tid], v_sdata[tid + 16], v_sdata[tid]);
          if (BS >=  16) max(v_sdata[tid], v_sdata[tid +  8], v_sdata[tid]);
          if (BS >=   8) max(v_sdata[tid], v_sdata[tid +  4], v_sdata[tid]);
          if (BS >=   4) max(v_sdata[tid], v_sdata[tid +  2], v_sdata[tid]);
          if (BS >=   2) max(v_sdata[tid], v_sdata[tid +  1], v_sdata[tid]);
        }

        __syncthreads();

        if (tid == 0) intermediate_result.data[blockIdx.x] = sdata[0];
      }

      __device__ __forceinline__ void
      operator() (unsigned int idx) const
      {
        //blob_max = previous_result.data[idx];

        reduce6<CTA_SIZE_TOTAL>();
      }

      __device__ __forceinline__ void
      reduce (R min, unsigned int idx) const
      {
        blob_max = make_pair(0, UINT_MAX);

        reduce6<CTA_SIZE_TOTAL>();

        if (1 == gridDim.x && 0 == blockIdx.x && 0 == threadIdx.x)
        {
          extern __shared__ R __sdata[];

          if (__sdata[0].y > min.y)
            result.data[idx] = __sdata[0];
          else
            result.data[idx] = make_pair(0, 0);
        }
      }
    };

    template <typename T, typename R> __global__ void
    findLargestBlob(const FindLargestBlob<T, R> flb, unsigned int idx)
    {
      flb(idx);
    }

    template <typename T, typename R> __global__ void
    findLargestBlobReduce(const FindLargestBlob<T, R> flb, R min, unsigned int idx)
    {
      flb.reduce(min, idx);
    }
  }
}

void
pcl::device::depthAwareConnectedComponents (
                          const DepthMap& depth,
                          const DeviceArray2D<unsigned char>& src,
                          DeviceArray2D<unsigned int> dst,
                          DeviceArray2D<unsigned int>& blobSizes,
                          DeviceArray<uint2>& largestBlobs,
                          DeviceArray<unsigned char>& isNotDone,
                          DeviceArray<uint2>& reductionTemporary,
                          unsigned int depthThreshold,
                          unsigned int minBlobSize,
                          unsigned int numBlobs
  )
{
  const unsigned int size = dst.rows() * dst.cols();

  if (!reductionTemporary.size())
  {
    reductionTemporary.create (divUp(size, FindLargestBlob<unsigned int, uint2>::CTA_SIZE_TOTAL));
  }

  if (!(numBlobs == largestBlobs.size()))
  {
    largestBlobs.release();
    largestBlobs.create(numBlobs);

    sync();
  }

  // Init labels and blobSizes
  dim3 block (32,8);
  dim3 grid (divUp (dst.cols(), block.x), divUp (dst.rows(), block.y));

  initializeConnectedComponentsKernel<<<grid, block>>>(depth, src, blobSizes, dst);
  cudaSafeCall (cudaGetLastError ());
  //cudaSafeCall (cudaDeviceSynchronize ());

  std::vector<unsigned char> isNotDoneHost(1); //Used to terminate
  isNotDoneHost[0] = 1;

  std::vector<uint2> cpu_temporary(1, make_uint2(0, UINT_MAX));
  DeviceArray<uint2> temp(largestBlobs.ptr(), 1);
  temp.upload(cpu_temporary);

  // Pre-run algorithm a number of times
  const unsigned int prerun_steps = 6;
  for (unsigned int i = 0; i < prerun_steps; ++i)
  {
    linkConnectedComponentsKernel<<<grid, block>>>(depthThreshold, depth, dst, isNotDone);
    relabelConnectedComponentsKernel<<<grid, block>>>(dst, blobSizes);
  }

  // Loop until isNotDone = 0;
  for (unsigned int counter = 0; counter < 50 && isNotDoneHost[0]; ++counter)
  {
    isNotDoneHost[0] = 0;
    isNotDone.upload(isNotDoneHost);

    linkConnectedComponentsKernel<<<grid, block>>>(depthThreshold, depth, dst, isNotDone);
    relabelConnectedComponentsKernel<<<grid, block>>>(dst, blobSizes);

    //cudaSafeCall (cudaDeviceSynchronize ());
    cudaSafeCall (cudaGetLastError ());

    isNotDone.download(isNotDoneHost);
  }

  typedef FindLargestBlob<unsigned int, uint2> FLB;

  unsigned int flb_block = FLB::CTA_SIZE_TOTAL;
  const size_t shared_mem = (FLB::CTA_SIZE_TOTAL > 32 ? FLB::CTA_SIZE_TOTAL : 64) * sizeof(uint2);

  uint2 winner = make_uint2(0, 0);

  std::vector<uint2> cpu_results(numBlobs, make_uint2(0, 0));

  unsigned int previous_max = UINT_MAX;
  bool minimum_reached = false;
  for (unsigned int i = 0; i < numBlobs && !minimum_reached ; ++i)
  {
    unsigned int flb_grid = divUp(size, flb_block);

    {
      FLB flb;

      flb.input = PtrSz<unsigned int>(blobSizes.ptr(), size);
      flb.intermediate_result = reductionTemporary;
      flb.previous_result = largestBlobs;
      flb.blob_max = make_uint2(0, previous_max);

      findLargestBlob<<<flb_grid, flb_block, shared_mem>>>(flb, 0 == i ? 0 : i - 1);
    }

    {
      DeviceArray<uint2> partial_gpu_result(reductionTemporary.ptr(), flb_grid);

      cpu_temporary.resize(partial_gpu_result.size());

      partial_gpu_result.download(cpu_temporary);


      uint2 result = make_uint2(0, 0);
      for (unsigned int j = 0; j < cpu_temporary.size(); ++j)
      {
        if (cpu_temporary[j].y > result.y)
        {
          result = cpu_temporary[j];
        }
      }

      if (result.y < minBlobSize)
      {
        result = make_uint2(0, 0);
        minimum_reached = true;
      }

      previous_max = result.y;
      cpu_results[i] = result;
    }

//    {
//      FindLargestBlob<uint2, uint2> flb;
//
//      flb.input = reductionTemporary;
//      flb.intermediate_result = reductionTemporary;
//      flb.result = DeviceArray<uint2>(largestBlobs.ptr(), 1);
//
//      while (flb_grid > 1)
//      {
//        flb.input.size = flb_grid;
//        flb_grid = divUp(flb_grid, flb_block);
//
//        findLargestBlobReduce<<<flb_grid, flb_block, shared_mem>>>(flb, make_uint2(0, minBlobSize), i);
//
//      }
//    }
  }

  largestBlobs.upload(cpu_results);

  filterConnectedComponentsKernel<<<divUp(size, flb_block), flb_block, numBlobs * sizeof(uint2)>>>(dst, largestBlobs);

  cudaSafeCall (cudaDeviceSynchronize ());
  cudaSafeCall (cudaGetLastError ());

//  {
//    DeviceArray<uint2> temp2 (largestBlobs.ptr(), numBlobs);
//
//    cpu_result.resize(temp2.size());
//
//    temp2.download(cpu_result);
//
//
//    winner = cpu_results[0];

//    if (winner.y < minBlobSize)
//      winner.x = 0;

//    // Find blob id for blob with greatest area
//    int cols;
//    std::vector<unsigned int> blobSizesHost;
//    blobSizes.download(blobSizesHost,cols);

//    unsigned int maxvalue = 0;
//    unsigned int winnerid = 0;
//    for (unsigned int i = 0; i < blobSizesHost.size(); ++i) {
//      if ( blobSizesHost[i] > maxvalue )
//      {
//        maxvalue = blobSizesHost[i];
//        winnerid = i;
//      }
//    }

//    if (maxvalue < minBlobSize)
//      winnerid = 0;

//    if (!(winnerid == winner.x))
//      std::cerr << "Difference in values: " << winnerid << " " << winner.x << " " << maxvalue << " " << winner.y << std::endl;
//  }


}
