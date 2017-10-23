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

long int orientation(Point p, Point q, Point r)
{
    long int val = (q.second - p.second)*(r.first-q.first)-(q.first-p.first)*(r.second-q.second);
 
    if(val == 0) 
    	return 0;
    return (val > 0)? 1: 2;
}

void DesignHull(vector<Point> points, long int n)
{
    if (n < 3) return;
 
    long int l = 0;
    for(long int i=1;i<n;i++)
       if(points[i].first<points[l].first)
           l = i;
 
    long int p = l, q;
    do
    {
        hull.insert(points[p]);
 
        q = (p+1)%n;
        for(long int i = 0;i<n;i++)
        {
           if(orientation(points[p],points[i],points[q])==2)
               q = i;
        }
 
        p = q;
 
    }while (p!=l);
 
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
   char *file_name = wbPath_join(dir_name, "expserial.raw");

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
