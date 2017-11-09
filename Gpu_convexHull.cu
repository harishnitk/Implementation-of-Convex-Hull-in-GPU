#include "wb.h"
#include<bits/stdc++.h>
#include<vector>
#include<fstream>
#include<string.h>
#include<sstream>
#include<stdio.h>
#include<thrust/device_vector.h>
#include<thrust/copy.h>
#include<thrust/scan.h>
#include <thrust/sort.h>
#include<vector>
#include<climits>

using namespace std;

typedef pair<long int,long int> Point;

struct convexHull
{
   Point point;
   long int label;
   long int distance;
   int mark;
};

struct assignMax
{
   long int max;
   long int index;  
};

Point *hull;

#define CUDA_CHECK(ans)                                                   \
{ gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}


/*
@ region Definition of global kernel function
*/

__global__ void lowerHull(convexHull *input,Point *devHull,long int size)
{
   long int Idx = blockIdx.x*blockDim.x+threadIdx.x;
   
   /*
   @ calculate perpendicular point
   */
   if((Idx)<size)
   {

     Point P = input[Idx].point;
     long int lb = input[Idx].label;
     Point min = devHull[lb];
     Point max = devHull[lb+1];
     input[Idx].distance = (P.second-min.second)*(max.first-min.first)-(max.second-min.second)*(P.first-min.first);
     if(input[Idx].distance<0)
     {
       input[Idx].mark = -1;
     }else
     {
       input[Idx].mark = 1;
     }
   }

}

__global__ void scan(convexHull *input,assignMax *store,int size)
{
   int Idx = blockIdx.x*blockDim.x+threadIdx.x;
   long int itr = 0;
  
   for(;itr<size;itr++)
   {
      if(Idx==input[itr].label)
      {
        while(itr<size&&Idx==input[itr].label)
        {
           if(store[Idx].max<input[itr].distance)
           {
              store[Idx].max = input[itr].distance;
              store[Idx].index = itr;
              //printf("%ld\n",store[Idx].max);
           }
           itr++;
        }
        break;
      }
   }
}


struct labelbased
{
  __host__ __device__ bool operator()(convexHull &x, convexHull &y)
  {
     return x.label<y.label;
  }
};

/*
@endregion
*/


int main(int argc, char *argv[]) {
  
  long int inputLength,itr;
  Point leftmost_point{INT_MAX,0},rightmost_point{INT_MIN,0};
  
  wbTime_start(Generic, "Importing data and creating memory on host");
  ifstream file;
  file.open(argv[2]);
  file>>inputLength;
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  /*
  @region declaration
  */
  convexHull *hostInput = new convexHull[inputLength];
  convexHull *deviceInput = new convexHull[inputLength];
  convexHull *original = new convexHull[inputLength];
  hull = new Point[inputLength];
  Point *deviceHull = new Point[inputLength];
  long int hull_length = 2;
  /*
  @endregion
  */
 
  for(itr=0;itr<inputLength;itr++)
  {
     file>>hostInput[itr].point.first;
     file>>hostInput[itr].point.second;
     hostInput[itr].label = 0;
     original[itr].label = 0;
     original[itr].distance = 0;
     hostInput[itr].distance = 0;
     original[itr].point.first = hostInput[itr].point.first;
     original[itr].point.second = hostInput[itr].point.second;
  }
  file.close();

  int threads_per_block = 512;
  dim3 blocks(ceil(inputLength/threads_per_block)+1,1,1);
  
  /*
  @ param find the leftmost and rightmost Point in x direction
  */
  for(itr=0;itr<inputLength;itr++)
  {
     if(leftmost_point.first>hostInput[itr].point.first)
     {
        leftmost_point = hostInput[itr].point;
     }
     if(rightmost_point.first<hostInput[itr].point.first)
     {
        rightmost_point = hostInput[itr].point;
     }
  }

  /*
  @region Memory-Allocation
  */
  cudaMalloc((void **)&deviceInput,inputLength*sizeof(convexHull));
  cudaMemcpy(deviceInput,hostInput,inputLength*sizeof(convexHull),cudaMemcpyHostToDevice);
  
  //--------------- insert point in hull -----------//
  hull[0] = leftmost_point;
  hull[1] = rightmost_point;
  

  cudaMalloc((void **)&deviceHull,inputLength*sizeof(Point));
  cudaMemcpy(deviceHull,hull,inputLength*sizeof(Point),cudaMemcpyHostToDevice);
  
  /*
  @endregion
  */

  /*
  @ param calculate the LowerHull
  */
  
  lowerHull<<<blocks,threads_per_block>>>(deviceInput,deviceHull,inputLength);
  cudaMemcpy(hostInput,deviceInput,inputLength*sizeof(convexHull),cudaMemcpyDeviceToHost);
  
  thrust::device_vector<convexHull> devI(hostInput,hostInput+inputLength);
  thrust::sort(devI.begin(),devI.end(),labelbased());
  thrust::copy(devI.begin(),devI.end(),hostInput);

  
  int label_thread = hostInput[inputLength-1].label+1;

  assignMax *devMax = new assignMax[label_thread];
  assignMax *hostMax = new assignMax[label_thread];
  for(int m=0;m<label_thread;m++)
  {
    hostMax[m].max = INT_MIN;
    hostMax[m].index = -1;
  }

  cudaMalloc((void **)&devMax,label_thread*sizeof(assignMax));
  cudaMemcpy(devMax,hostMax,label_thread*sizeof(assignMax),cudaMemcpyHostToDevice);

  cudaMalloc((void **)&deviceInput,inputLength*sizeof(convexHull));
  cudaMemcpy(deviceInput,hostInput,inputLength*sizeof(convexHull),cudaMemcpyHostToDevice);
  
  scan<<<1,label_thread>>>(deviceInput,devMax,inputLength);
  cudaMemcpy(hostMax,devMax,label_thread*sizeof(assignMax),cudaMemcpyDeviceToHost);  
  
  /*
   @method update hull
   @description []
  */
  
  for(int k=0;k<label_thread;k++)
  {
    hull[hull_length] = hostInput[hostMax[k].index].point;
    hull_length++;
  }
   
  /*
  @endregion
  */

  cudaDeviceSynchronize();
  

  return 0;
}
