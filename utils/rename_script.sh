#!/usr/bin/env bash


# replace the file extension with new extension
# usage: cd /to/dir
# copy the following to terminal to run

for file in *.TIF
do
  mv "$file" "${file%.TIF}.tif"
done

# rename the file
# usage: cd /to/dir

for file in *.jpg
do
#    mv "$file" "centromere_$file"
#    mv "$file" "coarse_$file"
#    mv "$file" "cytoplasmatic_$file"
#     mv "$file" "fine_$file"
#      mv "$file" "homogeneous_$file"
      mv "$file" "nucleolar_$file"
done