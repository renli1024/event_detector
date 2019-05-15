from utils import *
from ed_model import *
from script.encode_window import *
import script.data_script as dt
import script.encode_window as ew
import os, datetime, time, pickle
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

tf.flags.DEFINE_float("split", 0.7, "dmm")
tf.flags.DEFINE_float("dev_size", 0.15,"dmm")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("evaluate_every", 100, "")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "")
tf.flags.DEFINE_integer("num_epochs", 300, "")
FLAGS = tf.flags.FLAGS

# dataset text test
file_prefix = os.path.abspath(os.path.join(os.path.curdir, 'script', 'ACE2005ENG', 
     'orig', 'bn', 'timex2norm', 'CNNHL_ENG_20030304_142751.10'))
apf_file_path = file_prefix + '.apf.xml'
sgm_file_path = file_prefix + '.sgm'
tokens, anchors = dt.read_file(apf_file_path, sgm_file_path)

# website text test
# tokens, anchors = dt.read_outside_file("./script/outside_text")

# encode word windows
vocab = pickle.load(open("./preprocessing/vocabulary_bn.bin", "rb"))
word_vecs = pickle.load(open("./preprocessing/vector_bn.bin", "rb"))

# construct word vectors according to vocabulary 
windows, labels = ew.encode_window(tokens, anchors, vocab)
with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        cf = config()
        vocab_length = len(word_vecs)
        cnn = ed_model(cf, vocab_length, word_vecs)
        # count training steps
        global_step = tf.Variable(0, name="global_step", trainable=False)
        restore_saver = tf.train.Saver()
        restore_saver.restore(sess, os.path.join(os.path.curdir, "runs", "1556656779", "checkpoints", "final-31097"))
        y_anchors = sess.run(cnn.predictions, {
                cnn.input_x: windows,
                cnn.dropout_keep_prob: 1,
                cnn.size_batch : len(windows)
            })

        # correct annotations (only for dataset text test)
        print("correct annotations")
        print()
        for i in range(len(anchors[0])):
            print(tokens[0][i], end = "")
            if anchors[0][i] != 0:
                for key, value in dt.EVENT_MAP.items():
                    if value == anchors[0][i]:
                        print(" %s" % {key}, end="")
            print(" ", end="")
        print()
        print("----------------------")
        print("predictions")
        print()
        # prediction annotations
        for i in range(len(y_anchors)):
            print(tokens[0][i], end = "")
            if y_anchors[i] != 0:
                for key, value in dt.EVENT_MAP.items():
                    if value == y_anchors[i]:
                        print(" %s " % {key}, end="")
            print(" ", end="")

print()