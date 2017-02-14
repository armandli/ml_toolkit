#include <gtest/gtest.h>

#include <iostream>

#include <vector>
#include <csv_blockreader.h>
#include <csv_reader.h>

using namespace std;

TEST(TestCSVBlockreader, TestCSVBlockreader){
  csv_blockreader reader("test.csv");

  ASSERT_EQ(true, reader.is_open());

  vector<char*> columns;
  int expect_val = 0;
  while (reader.read_line(columns)){
    EXPECT_EQ(5, columns.size());
    for (int i = 0; i < columns.size(); ++i, ++expect_val){
      EXPECT_EQ(expect_val, atoi(columns[i]));
    }
  }
}

TEST(TestCSVReader, TestCSVReader){
  csv_reader reader("test.csv");

  ASSERT_EQ(true, reader.is_open());

  const map<string, const char*>& m = reader.get_map();
  ASSERT_EQ(true, reader.read_line());
  EXPECT_EQ(0, strcmp("0", m.at(string("a"))));
  EXPECT_EQ(0, strcmp("1", m.at(string("b"))));
  EXPECT_EQ(0, strcmp("2", m.at(string("c"))));
  EXPECT_EQ(0, strcmp("3", m.at(string("d"))));
  EXPECT_EQ(0, strcmp("4", m.at(string("e"))));

  ASSERT_EQ(true, reader.read_line());
  EXPECT_EQ(0, strcmp("5", m.at(string("a"))));
  EXPECT_EQ(0, strcmp("6", m.at(string("b"))));
  EXPECT_EQ(0, strcmp("7", m.at(string("c"))));
  EXPECT_EQ(0, strcmp("8", m.at(string("d"))));
  EXPECT_EQ(0, strcmp("9", m.at(string("e"))));

  ASSERT_EQ(true, reader.read_line());
  EXPECT_EQ(0, strcmp("10", m.at(string("a"))));
  EXPECT_EQ(0, strcmp("11", m.at(string("b"))));
  EXPECT_EQ(0, strcmp("12", m.at(string("c"))));
  EXPECT_EQ(0, strcmp("13", m.at(string("d"))));
  EXPECT_EQ(0, strcmp("14", m.at(string("e"))));
}
