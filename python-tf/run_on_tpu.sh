#! /bin/bash

python aqc_training.py --tpu $TPU_NAME --use_tpu --batch_size 1024 \
	--training_data gs://deep-qc-training-data/deep_qc_data_shuffled_20190805_train.tfrecord \
	--validation_data gs://deep-qc-training-data/deep_qc_data_shuffled_20190805_val.tfrecord \
	--model_dir gs://deep-qc-training-data/model \
	--learning_rate 0.00001 \
	--learning_rate_decay_epochs 4


