#ifndef CSV_BLOCKREADER
#define CSV_BLOCKREADER

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace std;

class csv_blockreader {
  struct Buffer {
    char* m_buff;
    int m_sz;

    Buffer(int sz) : m_buff(new (std::nothrow) char[sz]), m_sz(sz) {}
    ~Buffer(){
      if (m_buff) delete[] m_buff;
    }   

    operator char*(){
      return m_buff;
    }   
  };

  FILE* m_fd;
  const char* m_delim;
  int m_max_sz;
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
    char* el = std::strrchr(buffer, '\n');
    if (el) *el = '\0';
    return ret;
  }

  char* strtok(char* __restrict__ str, const char* const __restrict__ delim){
    char* start = str ? str : m_strtok_register;
    if (not start) return NULL;

    char* ret = start;

    for (; *start && not std::strchr(delim, *start); ++start);

    if (*start){
      m_strtok_register = &1[start];
      *start = '\0';
    } else 
      m_strtok_register = NULL;

    return ret;
  }

  void read_header(){
    if (m_buffer.m_buff == 0){
      m_error = "buffer Bad alloc";
      close();
      return;
    }

    get_line(m_buffer, m_max_sz, m_fd);

    if (m_buffer.m_buff == NULL){
      m_error = "File is empty";
      close();
    }
  }
public:
  csv_blockreader(const char* filename, const char* delimiter = ",", int max_size = 4096, bool header = true):
    m_fd(fopen64(filename, "r")), m_delim(delimiter), m_max_sz(max_size), m_buffer(max_size) {
    if (m_fd == NULL){
      std::stringstream ss;
      ss << "Failed to open file " << filename;
      m_error = ss.str();
      return;
    }
    if (header) read_header();
  }
  ~csv_blockreader(){ close(); }

  bool is_open() const { return m_fd != NULL; }

  bool read_line(vector<char*>& columns){
    columns.clear();
    if (not get_line(m_buffer, m_max_sz, m_fd))
      return false;

    for (char* pc = (*this).strtok(m_buffer, m_delim); pc; pc = (*this).strtok(NULL, m_delim))
      columns.push_back(pc);
    return true;
  }
};

#endif
