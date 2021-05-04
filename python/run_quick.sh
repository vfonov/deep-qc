#! /bin/bash

python aqc_training.py --net r18 \
    --adam --n_epochs 2 --batch_size 196 model_r18_quick  \
    --folds 8 --fold 0

