import tensorflow as tf
import sys
tf.compat.v1.enable_eager_execution()
tf.logging.set_verbosity('WARN')

files=sys.argv

for f in sys.argv[1:]:
    print(f, sum(1 for i in tf.data.TFRecordDataset(f)))
