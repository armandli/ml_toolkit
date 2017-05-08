#ifndef CSV_READER
#define CSV_READER

#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <limits>
#include <sstream>

using namespace std;

class csv_reader {
  struct Buffer {
    char* m_buff;
    int m_sz;

    Buffer(int sz) : m_buff(new (nothrow) char[sz]), m_sz(sz) {}
    ~Buffer(){
      if (m_buff) delete[] m_buff;
    }

    operator char*(){
      return m_buff;
    }
  };

  FILE* m_fd;
  const char* m_delim;
  const char* m_escape;
  int m_max_sz;
  vector<map<string, const char*>::iterator> m_columns;
  map<string, const char*> m_linemap;
  string m_error;
  char* m_strtok_register;
  Buffer m_buffer;

  void close(){
    if (m_fd){
      fclose(m_fd);
      m_fd = NULL;
    }
  }

  char* get_line(char* buffer, int max_size, FILE* fd){
    char* ret = fgets(buffer, max_size, fd);
    char* el = strrchr(buffer, '\n');
    if (el) *el = '\0';
    el = std::strchr(buffer, '\r');
    if (el) *el = '\0';
    return ret;
  }

  char* strtok(char* __restrict__ str, const char* const __restrict__ delim, const char* const __restrict__ escape){
    char* start = str ? str : m_strtok_register;
    if (not start) return NULL;

    char* ret = start;

    //handling escape quotes
    if (std::strchr(escape, *start)){
      start = ++ret;
      for (; *start && not std::strchr(escape, *start); ++start)
        if (*start == '\\' && 1[start]){
          ++start;
          continue;
        }
      if (*start && std::strchr(escape, *start)){
        *start = '\0';
        ++start;
      }
    }

    for (; *start && not strchr(delim, *start); ++start);

    if (*start){
      m_strtok_register = &1[start];
      *start = '\0';
    } else 
      m_strtok_register = NULL;

    return ret;
  }

  void read_header(){
    Buffer buffer(m_max_sz);
    if (buffer.m_buff == 0){
      m_error = "buffer Bad alloc";
      close();
      return;
    }
  
    get_line(buffer, m_max_sz, m_fd);
  
    if (buffer.m_buff == NULL){
      m_error = "File is empty";
      close();
      return;
    }
  
    map<const char*, int> found;
  
    int col_count = 0;
    for (char* pc = (*this).strtok(buffer, m_delim, m_escape); pc; ++col_count, pc = (*this).strtok(NULL, m_delim, m_escape)){
      found.insert(make_pair(pc, col_count));
      m_linemap.insert(make_pair(string(pc), (const char*)0));
    }

    m_columns = vector<map<string, const char*>::iterator>(col_count + 1);
  
    for (map<const char*, int>::iterator it = found.begin(); it != found.end(); ++it)
      m_columns[(*it).second] = m_linemap.find(string((*it).first));
  }

  void read(Buffer& buffer){
    char* pc = (*this).strtok(buffer, m_delim, m_escape);
    for (int col = 0; pc; ++col, pc = (*this).strtok(NULL, m_delim, m_escape)){
      map<string, const char*>::iterator it = m_columns[col];
      if (it != m_linemap.end()){
        (*it).second = pc;
      }
    }
  }

public:
  csv_reader(const char* filename, const char* delimiter = ",", const char* escape = "\"", int max_size = 4096) :
      m_fd(fopen64(filename, "r")), m_delim(delimiter), m_escape(escape), m_max_sz(max_size), m_strtok_register(NULL), m_buffer(max_size) {
    if (m_fd == NULL){
      stringstream ss;
      ss << "Failed to open file " << filename;
      m_error = ss.str();
      return;
    }
    if (m_max_sz <= 0){
      m_error = "Maximum line size <= 0";
      return;
    }
    read_header();
  }

  ~csv_reader(){ close(); }

  bool is_open() const { return m_fd != NULL; }

  bool read_line(){
    if (m_fd == NULL) return false;
  
    if (m_buffer.m_buff == NULL){
      m_error = "Buffer bad alloc";
      close();
      return false;
    }
  
    char* p = get_line(m_buffer, m_max_sz, m_fd);

    if (p == NULL) return false;
  
    read(m_buffer);
    return true;
  }

  const map<string, const char*>& get_map() const { return m_linemap; }

  template <typename F>
  void process(F& func, int offset = -1, int length = -1){
    if (m_fd == NULL) return;
  
    Buffer buffer(m_max_sz);
    if (buffer.m_buff == NULL){
      m_error = "Buffer bad alloc";
      close();
      return;
    }
  
    if (offset < 0) offset = 0;
    if (length < 0) length = numeric_limits<int>::max();
  
    for (int lineno = 0; lineno - offset < length && get_line(buffer, m_max_sz, m_fd); ++lineno){
      if (lineno < offset) continue;
      read(buffer);
      if (not func(get_map(), lineno)) break;
    }
  }
};

#endif
