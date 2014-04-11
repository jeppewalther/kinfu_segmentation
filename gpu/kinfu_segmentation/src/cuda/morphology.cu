
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
    __global__ void mmErodeKernel(
      PtrStep<unsigned char> src,
      PtrStepSz<unsigned char> dst
      )
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= dst.cols || y >= dst.rows) return;

      if( src.ptr (y)[x] == 0 || ( src.ptr (y+1)[x] + src.ptr (y-1)[x] + src.ptr (y)[x+1] + src.ptr (y)[x-1] ) < 4)
      {
        dst.ptr (y)[x] = 0;
        return;
      }

      dst.ptr (y)[x] = 1;
    }
  }
}


void pcl::device::mmErode(
    const DeviceArray2D<unsigned char>& src,
    DeviceArray2D<unsigned char>& temp,
    DeviceArray2D<unsigned char>& dst,
    unsigned int repeats
  )
{
  dim3 block (32,8);
  dim3 grid (divUp (dst.cols(), block.x), divUp (dst.rows(), block.y));

  if(repeats > 0)
  {
    mmErodeKernel<<<grid, block>>>(src, temp);

    bool to_copy = true;
    for (unsigned int i = 1; i < repeats; ++i)
    {
      if ((i & 1) == 0) {
        //Even
        mmErodeKernel<<<grid, block>>>(dst, temp);
        to_copy = true;
      } else {
        //Odd
        mmErodeKernel<<<grid, block>>>(temp, dst);
        to_copy = false;
      }
    }
    //Copy to dst if we ended on tmp
    if (to_copy) {
       temp.copyTo(dst);
    }
  }

}
