#!/usr/bin/env bash


#!/bin/bash
#for x in {1..10}; do (python script.py > /tmp/$x.log ) & done
echo
echo "<<<<<----- $(date) Running ${0} script file ----->>>>>"

# Setting dir
HOME_DIR="/home/long/Desktop/"
CODE_DIR=$HOME_DIR"handcrafted/"
LOG_DIR=$CODE_DIR"log/automated_script/"
#if [ ! -d $LOG_DIR ]; then
#  mkdir -p $LOG_DIR
#fi

# Setting hyper param array
knn_arr=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
pyramid_arr=("1" "1,2" "1,2,4" "1,2,4,8")

echo "Current time: $CURRENT_TIME"

# Write to logfile
cat <<EOF >$log_file

Date created: $(date)

knn_arr: ${knn_arr[@]}
pyramid_arr: ${pyramid_arr[@]}

################# End of Setting ##################

EOF

# Loop through hyper-param arrays
for knn in ${knn_arr[@]}
do  echo "knn: $knn"
	for pyramid in ${pyramid_arr[@]}
	do
		echo "pyramid: $pyramid"
		logfile=$LOG_DIR"PAP_sift_cnn_knn_"$knn"_pyramid_"$pyramid".txt"
		echo "logfie: $logfile"
            python3 ${CODE_DIR}main.py --knn ${knn} --pyramid ${pyramid} >>${logfile}
    done
done
