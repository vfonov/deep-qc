#! /bin/bash
set -e 
prefix=""

echo "Downloading minimal results, to run pretrained model 91MB in total..."


function download {
    set -e 
    for f in $@;do
        if [ ! -e ${f} ];then
        curl "${prefix}/${f}" -o ${f}
        fi
    done
}

download results_minimal_01.tar.xz

echo "Unpacking..."

mkdir -p results
tar xJf results_minimal_01.tar.xz -C results/

