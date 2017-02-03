#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <limits>

struct csv_reader {
  struct Column {
    const char* m_name;
    bool m_required;

    bool operator<(const Column& b) const {
      return strcmp(m_name, b.m_name) < 0;
    }
    bool operator==(const Column& b) const {
      return strcmp(m_name, b.m_name) == 0;
    }
  };
private:
  FILE* m_fd;
  std::string m_delim;
  int m_max_sz;
  std::vector<std::map<std::string, const char*>::iterator> m_columns;
  std::map<std::string, const char*> m_linemap;
  std::string m_error;
  char* m_strtok_register;

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

  void read_header(const std::set<Column>& columns);
  void read(Buffer& buffer);
public:
  csv_reader(const char* filename, const std::set<Column>& columns, const char* delimiter = ",", int max_size = 4096);
  ~csv_reader(){ close(); }

  bool is_open() const { return m_fd != NULL; }
  const char* error() const { return m_error.c_str(); }

  bool read_line();
  const std::map<std::string, const char*>& map() const { return m_linemap; }

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
    if (length < 0) length = std::numeric_limits<int>::max();
  
    for (int lineno = 0; lineno - offset < length && get_line(buffer, m_max_sz, m_fd); ++lineno){
      if (lineno < offset) continue;
      read(buffer);
      if (not func(map(), lineno)) break;
    }
  }
};
