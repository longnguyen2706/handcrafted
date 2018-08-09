#!/usr/bin/env bash


# replace the file extension with new extension
# usage: cd /to/dir
# copy the following to terminal to run

for file in *.TIF
do
  mv "$file" "${file%.TIF}.tif"
done