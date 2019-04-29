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
tf.flags.DEFINE_integer("num_epochs", 20, "")
FLAGS = tf.flags.FLAGS

# toke = pickle.load(open("./preprocessing/tokens_wl.bin", "rb"))
# anch = pickle.load(open("./preprocessing/anchors_wl.bin", "rb"))
# for i in range(len(anch)):
#     for j in range(len(anch[i])):
#         if anch[i][j]==34:
#             print(toke[i][j])
# print(toke[0])
# print(anch[0])

file_prefix = os.path.abspath(os.path.join(os.path.curdir, 'script', 'ACE2005ENG', 
    'orig', 'bn', 'timex2norm', 'CNNHL_ENG_20030304_142751.10'))
apf_file_path = file_prefix + '.apf.xml'
sgm_file_path = file_prefix + '.sgm'
# read token and anchor annotation information fronm file
tokens, anchors = dt.read_file(apf_file_path, sgm_file_path)

# encode word windows
vocab = pickle.load(open("./preprocessing/vocabulary1.bin", "rb"))
vocab_list = list(vocab.vocabulary_._mapping.keys())
word_vecs = load_bin_vec("./preprocessing/GoogleNews-vectors-negative300.bin", vocab_list)
pickle.dump(word_vecs, open("./preprocessing/vector1.bin", "wb"))
# construct word vectors according to vocabulary 
windows, labels = ew.encode_window(tokens, anchors, vocab)
# print(windows)
# print(labels)
# print(google_wordvector_path)
# word_vecs = ew.load_bin_vec(google_wordvector_path, vocab_list)
# pickle.dump(word_vecs, open("./signle_vector.bin", "wb"))
# word_vecs = pickle.load(open("./preprocessing/vector_all.bin", "rb"))
# print(1)
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
        restore_saver.restore(sess, os.path.join(os.path.curdir, "runs", "1556316537", "checkpoints", "final-77065"))
        y_anchors = sess.run(cnn.predictions, {
                cnn.input_x: windows,
                cnn.dropout_keep_prob: 1,
                cnn.size_batch : len(windows)
            })
        print(labels)
        print(y_anchors)
        print(1)
        # print("windows length %s " % len(windows))
        # print(windows)
        # print(tokens)
        # print("tokens length %d" % len(tokens))
        # print("anchors length %d" % len(anchors[0]))
        # print("y_pred length %d" % len(y_pred))
        # print(anchors[0])
        # for key, value in dt.EVENT_MAP.items():
        #     print("key: %s, value: %d" % (key, value))
        # print(1)

        for i in range(len(y_anchors)):
            print(tokens[0][i], end = "")
            if y_anchors[i] != 0:
                for key, value in dt.EVENT_MAP.items():
                    if value == y_anchors[i]:
                        print("(%s)" % {key}, end="")
            print(" ", end="")

        # test_step(tokens, anchors, anchor_test_std)
        # print("Test case 1:")
        # test_step(sents_test1, anchor_test1, anchor_test1_std)
        # print("Test case 2:")




print(1)