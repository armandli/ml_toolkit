#ifndef ML_UTIL
#define ML_UTIL

#include <ctime>
#include <random>
#include <map>
#include <Eigen/Dense>

using namespace std;

default_random_engine& get_default_random_engine(){
  static default_random_engine eng(time(0));
  return eng;
}

template <typename T>
struct CatStat {
  map<T, size_t> types;
  T avg;
};

template <typename T>
struct NumStat {
  T avg;
  T sdv;
  size_t count;
};

//TODO: convert each into using the stats struct above
template <typename T>
void update_categorical_column_stat(const string&, map<T, size_t>&);
template <>
void update_categorical_column_stat(const string& val, map<int, size_t>& smap){
  if (not val.empty()){
    int type = atoi(val.c_str());
    if (smap.find(type) != smap.end())
      smap[type]++;
    else
      smap[type] = 1;
  }
}
template <>
void update_categorical_column_stat(const string& val, map<string, size_t>& smap){
  if (not val.empty()){
    if (smap.find(val) != smap.end())
      smap[val]++;
    else
      smap[val] = 1;
  }
}

template <typename T>
void update_numerical_avg(const string& val, T& avg, size_t& count);
template <>
void update_numerical_avg(const string& val, double& avg, size_t& count){
  if (not val.empty()){
    avg += atof(val.c_str());
    count++;
  }
}

template <typename T>
void update_numerical_sdv(const string& val, const T& avg, T& sdv);
template <>
void update_numerical_sdv(const string& val, const double& avg, double& sdv){
  if (not val.empty())
    sdv += pow(atof(val.c_str()) - avg, 2);
}

template <typename T>
void compute_categorical_avg(const map<T, size_t>& m, T& avg){
  size_t tmp = 0;
  for (typename map<T, size_t>::const_iterator it = m.cbegin(); it != m.cend(); ++it)
    if ((*it).second > tmp){
      avg = (*it).first;
      tmp = (*it).second;
    }   
}

#endif
