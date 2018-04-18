#! /bin/bash
set -e 


echo "Come again, when this will be publicly released..."

exit 1


prefix=""

echo "Downloading raw data and results, 4.6Gb in total..."


function download {
    set -e 
    for f in $@;do
        if [ ! -e ${f} ];then
        curl "${prefix}/${f}" -o ${f}
        fi
    done
}

download raw_data_01.tar.xz raw_data_02.tar.xz results_01.tar.xz

echo "Unpacking..."

tar xJf raw_data_01.tar.xz -C data/
tar xJf raw_data_02.tar.xz -C data/

mkdir -p results
tar xJf results_01.tar.xz -C results/

