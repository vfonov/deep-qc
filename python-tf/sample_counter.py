import tensorflow as tf
import sys
tf.logging.set_verbosity('WARN')
tf.compat.v1.enable_eager_execution()

files=sys.argv

for f in sys.argv[1:]:
    print(f, sum(1 for i in tf.data.TFRecordDataset(f)))
