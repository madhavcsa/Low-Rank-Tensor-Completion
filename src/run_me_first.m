% Add folders to path.

addpath(pwd);

cd proposed;
addpath(genpath(pwd));
cd ..;

cd mex_files;
addpath(genpath(pwd));
cd ..;

cd auxiliary_files;
addpath(genpath(pwd));
cd ..;

cd thirdpartytools;
addpath(genpath(pwd));
cd ..;

cd datasets;
addpath(genpath(pwd));
cd ..;
