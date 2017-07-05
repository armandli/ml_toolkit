#!/bin/sh

if [ "`which optirun | grep -v "no optirun"`" == "" ]; then
  LD_LIBRARY_PATH=/opt/clblast/lib:/opt/clblas/lib64 ./test_clblas
else
  LD_LIBRARY_PATH=/opt/clblast/lib:/opt/clblas/lib64 optirun ./test_clblas
fi
