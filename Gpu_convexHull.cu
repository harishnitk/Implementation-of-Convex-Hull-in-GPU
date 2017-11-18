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
#include<cuda.h>
#include<curand.h>
#include<curand_kernel.h>
#include<cuda_runtime.h>
#include<thrust/extrema.h>
#include<algorithm>
//#include<Timer.h>

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
   Point p;
   int l;
   int index;  
};

Point *hull;
Point *lhull;
long int inputLength,itr;
Point leftmost_point{INT_MAX,0},rightmost_point{INT_MIN,0};
convexHull *hostInput;
convexHull *deviceInput;
convexHull *original;
Point *deviceHull;
long int hull_length = 2;
long int lhull_length = 2;
convexHull *appendPoint;
long int append_point_len = 0;
assignMax *devMax;
assignMax *hostMax;
bool flag = false;
int maxlabel = 1;

bool comparision(Point a,Point b)
{
    return (a.first<b.first);
}

bool labelsort(convexHull a, convexHull b)
{
   return a.label<b.label;
}

/*
 @region kernel functions
*/


__global__ void calculate_perpendicularDistance_And_markNegDistance(convexHull *input,Point *devHull,long int size)
{
   long int Idx = threadIdx.x+blockIdx.x*blockDim.x;

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

__global__ void scan(convexHull *input,assignMax *store,int size,bool upper)
{
   int Idx = blockIdx.x*blockDim.x+threadIdx.x;
   long int itr = 0;
   
   if(upper)
   {
     for(;itr<size;)
     {
        if(Idx==input[itr].label)
        {
          while(itr<size&&Idx==input[itr].label)
          {
             if((input[itr].distance>0)&&(store[Idx].max<input[itr].distance))
             {
                store[Idx].max = input[itr].distance;
                store[Idx].p.first = input[itr].point.first;
                store[Idx].p.second = input[itr].point.second;
                store[Idx].l = input[itr].label;
                store[Idx].index = itr;
             }
             itr++;
          }
          break;
        }
        itr++;
     }
    }
    else
    {
       for(;itr<size;)
       {
          if(Idx==input[itr].label)
          {
            while(itr<size&&Idx==input[itr].label)
            {
               if((input[itr].distance<0)&&(store[Idx].max>input[itr].distance))
               {
                  store[Idx].max = input[itr].distance;
                  store[Idx].p.first = input[itr].point.first;
                  store[Idx].p.second = input[itr].point.second;
                  store[Idx].l = input[itr].label;
                  store[Idx].index = itr;
               }
               itr++;
            }
            break;
          }
          itr++;
       }
    }
}

__global__ void update_label(convexHull *ptr, assignMax *M,long int size)
{
  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  int l = M[idx].l;
  long int i=0;
  for(;i<size;i++)
  {
      if(l==ptr[i].label&&M[idx].p.first<=ptr[i].point.first)
      {
        ptr[i].label = ptr[i].label+1;
      }
  }

}

void initialize(convexHull ptr[],long int n)
{
   long int i=0;
   for(;i<n;i++)
   {
      if(leftmost_point.first>ptr[i].point.first)
      {
        leftmost_point = ptr[i].point;
      }
      if(rightmost_point.first<ptr[i].point.first)
      {
        rightmost_point = ptr[i].point;
      }
   }
}

void initialize_Max(assignMax ptr[],long int n,bool upper)
{
   long int i=0;
   for(;i<n;i++)
   {
     if(upper)
     {
       ptr[i].max = INT_MIN;
     }
     else
     {
       ptr[i].max = INT_MAX;
     }
     ptr[i].p = {INT_MIN,INT_MIN};
     ptr[i].l = -1;
     ptr[i].index = -1;
   }
}

void update_Hull(int labels,bool upper)
{
  flag = true;

  if(upper)
  {
      for(int k=0;k<labels;k++)
      {
        flag = true;
        for(long int hull_itr=0;hull_itr<hull_length;hull_itr++)
        {
            if((hostInput[hostMax[k].index].point.first==hull[hull_itr].first)&&(hostInput[hostMax[k].index].point.second==hull[hull_itr].second))
            {
               // for distinct point
               flag = false;
               break;
            }
        }

        if(flag&&hostMax[k].l!=-1)
        {
            hull[hull_length] = hostInput[hostMax[k].index].point;
            hull_length++;
        }
      }
   }else
   {
      for(int k=0;k<labels;k++)
      {
        flag = true;
        for(long int hull_itr=0;hull_itr<lhull_length;hull_itr++)
        {
            if((hostInput[hostMax[k].index].point.first==lhull[hull_itr].first)&&(hostInput[hostMax[k].index].point.second==lhull[hull_itr].second))
            {
               // for distinct point
               flag = false;
               break;
            }
        }

        if(flag&&hostMax[k].l!=-1)
        {
            lhull[lhull_length] = hostInput[hostMax[k].index].point;
            lhull_length++;
        }
      }
   }
}

void update_And_Remove_MarkPoints(convexHull p[],long int &p_len,convexHull ap[],long int &ap_len)
{
    long int i=0;
    for(long int k=0;k<p_len;k++)
    {
       if(p[k].mark==-1)
       {
         ap[ap_len].point.first = p[k].point.first;
         ap[ap_len].point.second = p[k].point.second;
         ap[ap_len].label = p[k].label;
         ap[ap_len].distance = p[k].distance;
         ap[ap_len].mark = p[k].mark;
         ap_len++;
       }else
       {
         p[i] = p[k];
         i++;
       }
    }
    p_len = i;
}

void printOutput()
{
   int it = 0;
   for(;it<hull_length;it++)
   {
      cout<<hull[it].first<<" "<<hull[it].second<<endl;
   }
}

int main(int argc, char *argv[]) {
  
  wbTime_start(Generic, "Importing data and creating memory on host");
  ifstream file;
  file.open(argv[2]);
  file>>inputLength;
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  /*
  @region initialization
  */
  
  hostInput = new convexHull[inputLength];
  deviceInput = new convexHull[inputLength];
  original = new convexHull[inputLength];
  hull = new Point[inputLength];
  deviceHull = new Point[inputLength];
  appendPoint = new convexHull[inputLength];
  
  /*
  @endregion
  */
   
  //Assigining Value

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
  

  //find the left and righ most point form region
  initialize(hostInput,inputLength);
  
  hull[0] = leftmost_point;
  hull[1] = rightmost_point;
  
  flag = false;
  
  //GpuTimer time;
  //time.Start();

  /*
   @params [upper convexHull]
   @descriptor {finding the point which is the part of lower hull}
  */
  
  bool lower_flag = false; 

  int prev_len = 2;
  
  clock_t t = clock();

  do
  {
      //compute upperhull
      
      cudaMalloc((void **)&deviceInput,inputLength*sizeof(convexHull));
      cudaMemcpy(deviceInput,hostInput,inputLength*sizeof(convexHull),cudaMemcpyHostToDevice);

      cudaMalloc((void **)&deviceHull,inputLength*sizeof(Point));
      cudaMemcpy(deviceHull,hull,inputLength*sizeof(Point),cudaMemcpyHostToDevice);

      calculate_perpendicularDistance_And_markNegDistance<<<blocks,threads_per_block>>>(deviceInput,deviceHull,inputLength);
      cudaMemcpy(hostInput,deviceInput,inputLength*sizeof(convexHull),cudaMemcpyDeviceToHost);
      
      //sort based on the label
      
      std::sort(hostInput,hostInput+inputLength,labelsort);
      
      int label_thread = hostInput[inputLength-1].label+1;

      devMax = new assignMax[label_thread];
      hostMax = new assignMax[label_thread];

      initialize_Max(hostMax,label_thread,true);
      
      cudaMalloc((void **)&devMax,label_thread*sizeof(assignMax));
      cudaMemcpy(devMax,hostMax,label_thread*sizeof(assignMax),cudaMemcpyHostToDevice);

      cudaMalloc((void **)&deviceInput,inputLength*sizeof(convexHull));
      cudaMemcpy(deviceInput,hostInput,inputLength*sizeof(convexHull),cudaMemcpyHostToDevice);
      
      scan<<<1,label_thread>>>(deviceInput,devMax,inputLength,true);
      cudaMemcpy(hostMax,devMax,label_thread*sizeof(assignMax),cudaMemcpyDeviceToHost);
      /*
       @method update hull
       @description [which have distinct points in hull]
      */

      prev_len = hull_length;

      update_Hull(label_thread,true);
      
      if(lower_flag==false){
        //update_And_Mark
        update_And_Remove_MarkPoints(hostInput,inputLength,appendPoint,append_point_len);
        lower_flag = true;
      }
      
      //sort label_partition
      
      cudaMalloc((void **)&deviceInput,inputLength*sizeof(convexHull));
      cudaMemcpy(deviceInput,hostInput,inputLength*sizeof(convexHull),cudaMemcpyHostToDevice);

      update_label<<<1,label_thread>>>(deviceInput,devMax,inputLength);
      
      cudaMemcpy(hostInput,deviceInput,sizeof(convexHull)*inputLength,cudaMemcpyDeviceToHost);
      
      maxlabel = label_thread;

      std::sort(hull,hull+hull_length,comparision);
      
      cudaDeviceSynchronize();
  
  }while(prev_len!=hull_length);
  
  //time.Stop();
  /*
  @params [upper hull]
  @descriptor {finding the point which is the part of upper hull, having -ve perpendicular distance}
  */
  
  appendPoint[append_point_len].point = leftmost_point;
  appendPoint[append_point_len].label = 0;
  append_point_len++;
  appendPoint[append_point_len].point = rightmost_point;
  appendPoint[append_point_len].label = 0;
  append_point_len++;
  inputLength = append_point_len;
  
  thrust::device_vector<convexHull> temp(appendPoint,appendPoint+append_point_len);
  thrust::copy(temp.begin(),temp.end(),hostInput);
  

  lhull = new Point[inputLength];

  lhull[0] = leftmost_point;
  lhull[1] = rightmost_point;

  // finding the lower hull
  
  maxlabel = 1;
  
  do
  {
      
      cudaMalloc((void **)&deviceInput,inputLength*sizeof(convexHull));
      cudaMemcpy(deviceInput,hostInput,inputLength*sizeof(convexHull),cudaMemcpyHostToDevice);

      cudaMalloc((void **)&deviceHull,inputLength*sizeof(Point));
      cudaMemcpy(deviceHull,lhull,inputLength*sizeof(Point),cudaMemcpyHostToDevice);

      calculate_perpendicularDistance_And_markNegDistance<<<blocks,threads_per_block>>>(deviceInput,deviceHull,inputLength);
      cudaMemcpy(hostInput,deviceInput,inputLength*sizeof(convexHull),cudaMemcpyDeviceToHost);
    

      //sort based on the label

      std::sort(hostInput,hostInput+inputLength,labelsort);//findlabel(inputLength)+1;
      
      int label_thread = hostInput[inputLength-1].label+1;

      devMax = new assignMax[label_thread];
      hostMax = new assignMax[label_thread];

      initialize_Max(hostMax,label_thread,false);

      cudaMalloc((void **)&devMax,label_thread*sizeof(assignMax));
      cudaMemcpy(devMax,hostMax,label_thread*sizeof(assignMax),cudaMemcpyHostToDevice);

      cudaMalloc((void **)&deviceInput,inputLength*sizeof(convexHull));
      cudaMemcpy(deviceInput,hostInput,inputLength*sizeof(convexHull),cudaMemcpyHostToDevice);
      
      scan<<<1,label_thread>>>(deviceInput,devMax,inputLength,false);
      cudaMemcpy(hostMax,devMax,label_thread*sizeof(assignMax),cudaMemcpyDeviceToHost);
      
      /*
       @method update hull
       @description [which have distinct points in hull]
      */
 
      prev_len = lhull_length;

      update_Hull(label_thread,false);
      
      //sort
      
      cudaMalloc((void **)&deviceInput,inputLength*sizeof(convexHull));
      cudaMemcpy(deviceInput,hostInput,inputLength*sizeof(convexHull),cudaMemcpyHostToDevice);

      update_label<<<1,label_thread>>>(deviceInput,devMax,inputLength);
      cudaMemcpy(hostInput,deviceInput,sizeof(convexHull)*inputLength,cudaMemcpyDeviceToHost);
      
      maxlabel = label_thread;

      std::sort(lhull,lhull+lhull_length,comparision);

      cudaDeviceSynchronize();
  
  }while(prev_len!=lhull_length);
  
  t = clock() - t;

  cout<<"Total execution time is "<<(double)t/(double)CLOCKS_PER_SEC<<endl;
  /*
  @ param update the upperhull and lower hull
  */
  

  for(int j=0;j<lhull_length;j++)
  {
    bool check = true;
    for(int k=0;k<hull_length;k++){
       if(hull[k].first==lhull[j].first&&hull[k].second==lhull[j].second)
       {
         check = false;
         break;
       }
    }
    if(check)
    {
       hull[hull_length++] = lhull[j];
    }
  }

  printOutput();

  cudaDeviceSynchronize();
  

  return 0;
}
