from utils import *
from ed_model import *
import numpy as np
import tensorflow as tf
import os, datetime, time, pickle
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.python.tools import inspect_checkpoint as chkp

path = os.path.join(os.path.curdir, "runs", "1553519498", "checkpoints", "final-7343")
chkp.print_tensors_in_checkpoint_file(file_name=path, tensor_name='', all_tensors=False)

# meta_model_path = os.path.abspath(os.path.join(os.path.curdir, "runs", "1553519498", "checkpoints", "final-7343.meta"))
global_step = tf.get_variable(name="global_step", dtype=tf.int32, shape=[])
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(meta_model_path)
    saver.restore(sess, os.path.join(os.path.curdir, "runs", "1553519498", "checkpoints", "final-7343"))
    print(tf.get_collection("LOCAL_VARIABLES"))
    print(global_step.eval())
    print(1)

