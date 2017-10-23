/*
@classname Serial_QuickHull
*/

#include<bits/stdc++.h>
#include<vector>
#include<fstream>
#include<string.h>
#include<sstream>
#include "wb.h"

using namespace std;

static char *base_dir;

typedef pair<long int,long int> Point;
set<Point> hull;

long int findSide(Point p1, Point p2, Point p)
{
    long int val = (p.second - p1.second)*(p2.first - p1.first)-(p2.second - p1.second)*(p.first - p1.first);

    if (val>0)
        return 1;
    if (val<0)
        return -1;
    return 0;
}

long int lineDist(Point p1, Point p2, Point p)
{
    return abs((p.second - p1.second)*(p2.first - p1.first)-(p2.second - p1.second)*(p.first - p1.first));
}

void quickHull(vector<Point> p, long int n, Point p1, Point p2, long int side)
{
    long int ind = -1;
    long int max_dist = 0;

    for(long int i=0; i<n; i++)
    {
        long int temp = lineDist(p1, p2, p[i]);
        if(findSide(p1, p2, p[i])==side&&temp>max_dist)
        {
            ind = i;
            max_dist = temp;
        }
    }

    if(ind==-1)
    {
        hull.insert(p1);
        hull.insert(p2);
        return;
    }

    quickHull(p, n, p[ind], p1, -findSide(p[ind], p1, p2));
    quickHull(p, n, p[ind], p2, -findSide(p[ind], p2, p1));
}

void DesignHull(vector<Point> p, long int n)
{
    if(n<3)
    {
        cout << "Convex hull not possible\n";
        return;
    }

    long int min_x = 0, max_x = 0;
    for(long int i=1;i<n;i++)
    {
        if(p[i].first<p[min_x].first)
            min_x = i;
        if(p[i].first>p[max_x].first)
            max_x = i;
    }

    quickHull(p, n, p[min_x], p[max_x], 1);

    quickHull(p, n, p[min_x], p[max_x], -1);

}

vector<Point> getInputPoint(long int datasetNum)
{
  long int length;
  ifstream file;

  string stringname = "../ConvexHull/Dataset/";
  stringstream genericstr;
  genericstr<<datasetNum;
  stringname.append(genericstr.str()).append("/input.raw");

  file.open(stringname);
  file>>length;
  vector<Point> input;

  for(long int i=0;i<length;i++)
  {
     long int x,y;
     file>>x;
     file>>y;
     input.push_back(make_pair(x,y));
  }
  
  file.close();

  return input;
  
}

void write_output_data(long int datasetNum)
{
   const char *dir_name = wbDirectory_create(wbPath_join(base_dir, datasetNum));
   char *file_name = wbPath_join(dir_name, "expected.raw");

   FILE *handle = fopen(file_name, "w");

   set<Point> :: iterator it;
   fprintf(handle,"%d",hull.size());
   for(it=hull.begin();it!=hull.end();it++)
   {
       fprintf(handle, "\n%d %d", it->first,it->second);
       hull.erase(it);
   }
   
   fflush(handle);
   fclose(handle);
}

int main(int argc, char *argv[])
{
    base_dir = wbPath_join(wbDirectory_current(),"../ConvexHull","Dataset");
	  vector<Point> p = getInputPoint(0);
	  DesignHull(p,p.size());
    write_output_data(0);
    p = getInputPoint(1);
    DesignHull(p,p.size());
    write_output_data(1);
    p = getInputPoint(2);
    DesignHull(p,p.size());
    write_output_data(2);
    p = getInputPoint(3);
    DesignHull(p,p.size());
    write_output_data(3);
    p = getInputPoint(4);
    DesignHull(p,p.size());
    write_output_data(4);
    p = getInputPoint(5);
    DesignHull(p,p.size());
    write_output_data(5);      
    return 0;
}
