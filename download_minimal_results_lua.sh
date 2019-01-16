#! /bin/bash
set -e 
prefix="https://github.com/vfonov/deep-qc/releases/download/v0.2/"

echo "Downloading minimal lua results, to run pretrained model 182MB in total..."

function download {
    set -e 
    for f in $@;do
        if [ ! -e ${f} ];then
        curl --location "${prefix}/${f}" -o ${f}
        fi
    done
}

download models_minimal_lua_02.tar.xz

echo "Unpacking..."

mkdir -p results
tar xJf models_minimal_lua_02.tar.xz
