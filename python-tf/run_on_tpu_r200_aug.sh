#! /bin/bash


python aqc_training.py --tpu $TPU_NAME --use_tpu --batch_size 128 \
  --training_data gs://deep-qc-training-data/deep-qc-aug_20190817_train.tfrecord --validation_data gs://deep-qc-training-data/deep-qc-aug_20190817_val.tfrecord --model_dir gs://deep-qc-training-data/model/r200_aug --learning_rate 0.00001 --learning_rate_decay_epochs 10 --flavor r200 --mult 10 --train_epochs 900 --n_samples 213574 --n_val_samples 5463 --n_test_samples 20757




