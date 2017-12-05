cd mex_files

mex -largeArrayDims mysetsparseentries.c
mex -largeArrayDims myspmaskmult.c
mex -largeArrayDims getValsAtIndex_mex.c

cd ..

