#include <sstream>
#include <limits>

#include "csv_reader.h"

csv_reader::csv_reader(const char* filename, const std::set<Column>& columns, const char* delimiter, int max_size) :
    m_fd(fopen64(filename, "r")), m_delim(delimiter), m_max_sz(max_size), m_strtok_register(NULL) {
  if (m_fd == NULL){
    std::stringstream ss;
    ss << "Failed to open file " << filename;
    m_error = ss.str();
    return;
  }
  if (m_max_sz <= 0){
    m_error = "Maximum line size <= 0";
    return;
  }
  read_header(columns);
}

void csv_reader::read_header(const std::set<Column>& columns){
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

  std::map<Column, int> found;

  int col_count = 0;
  for (char* pc = (*this).strtok(buffer, m_delim.c_str()); pc; ++col_count, pc = (*this).strtok(NULL, m_delim.c_str())){
    Column c = {pc, false};
    std::set<Column>::iterator it = columns.find(c);
    if (columns.size() == 0 || it != columns.end()){
      found.insert(std::make_pair(c, col_count));
      m_linemap.insert(std::make_pair(std::string(c.m_name), (const char*)0));
    }
  }

  m_columns.reserve(col_count + 1);

  for (int i = 0; i < col_count; ++i)
    m_columns.push_back(m_linemap.end());
  for (std::map<Column, int>::iterator it = found.begin(); it != found.end(); ++it)
    m_columns[(*it).second] = m_linemap.find(std::string((*it).first.m_name));

  if (columns.size() == 0) return;

  for (std::set<Column>::iterator it = columns.begin(); it != columns.end(); ++it)
    if ((*it).m_required && found.find(*it) == found.end()){
      std::stringstream ss;
      ss << "Column " << (*it).m_name << " not found";
      m_error = ss.str();
      close();
      return;
    }
}

void csv_reader::read(Buffer& buffer){
  char* pc = (*this).strtok(buffer, m_delim.c_str());
  for (int col = 0; pc; ++col, pc = (*this).strtok(NULL, m_delim.c_str())){
    std::map<std::string, const char*>::iterator it = m_columns[col];
    if (it != m_linemap.end()){
      (*it).second = pc;
    }
  }
}

bool csv_reader::read_line(){
  if (m_fd == NULL) return false;

  Buffer buffer(m_max_sz);
  if (buffer.m_buff == NULL){
    m_error = "Buffer bad alloc";
    close();
    return false;
  }

  get_line(buffer, m_max_sz, m_fd);

  read(buffer);
  return true;
}
