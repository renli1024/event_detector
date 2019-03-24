import numpy as np
import tensorflow as tf
import pickle


class config():
    num_epochs = 2
    sequence_length = 31
    batch_size = 50
    vocab_size = 16314
    triger_size = 34
    position_embedded_size = 50
    embedding_size = 300
    filter_sizes = [2,3,4]
    feature_size = 150


class ed_model(object):
    def __init__(self, config, vocab_length, vectors, l2_reg_lambda=0.003):
        # vectors arg is word vector
        self.config = config
        self.input_x = tf.placeholder(tf.int32,
                                      [None, self.config.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32,
                                      [None, self.config.triger_size],
                                      name="input_y")
        self.size_batch = tf.placeholder(tf.int32, name="size_batch") # equals 50
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.feature = self.add_embedding(vectors)
        self.add_model(l2_reg_lambda)


    def add_embedding(self, vectors):
        # vectors shape: (14037, 300) 
        # construct the look-up table
        initial = tf.constant(vectors, dtype=tf.float32)
        with tf.variable_scope('embedded_layer'):
            WV = tf.get_variable('word_vectors', initializer=initial)
            wv = tf.nn.embedding_lookup(WV, self.input_x) # input_x shape: (50, 31), so wv shape: (50, 31, 300)
        return tf.expand_dims(wv, [-1])
        #     position_embedding = tf.get_variable("Pos_emb",
        #                                         shape=[self.config.sequence_length,
        #                                                self.config.position_embedded_size],
        #                                         dtype=tf.float32) # shape: (31, 50)
        #     # split wv (50, 31, 300) to 31 (50, 1, 300) tensors,
        #     # squeeze each (50, 1, 300) tensor to (50, 300), then form a list
        #     wv = [tf.squeeze(x, [1]) for x in tf.split(axis=1, 
        #     num_or_size_splits=self.config.sequence_length, value=wv)]

        #     # concatenate the word vector features with position features
        #     inputs = []
        #     for v in range(len(wv)):
        #         position_features = tf.tile(position_embedding[v], [self.size_batch]) # shape(2500, )
        #         # transform shape to (50, 50)
        #         reshaped_pos_features = tf.reshape(position_features, 
        #                                 [self.size_batch, self.config.position_embedded_size])
        #         # concatenate (50, 300) and (50, 50) into (50, 350)
        #         concat_features = tf.concat(axis=1, values=[wv[v], reshaped_pos_features])
        #         inputs.append(concat_features)

        #     # stack tensor (50, 350) in list into a higher rank tensor of shape (31, 50, 350)
        #     # transpose tensor shape (31, 50, 350) to (50, 31, 350)
        #     inputs = tf.transpose(tf.stack(inputs), perm=[1,0,2])
        #     # temp = tf.expand_dims(inputs, [-1])
        #     # print(temp.shape)
        # # final shape (50, 31, 350, 1)
        # return tf.expand_dims(inputs, [-1])


    def add_model(self, l2_reg_lambda):
        """
        l2_reg_lambda: used to avoid overfitting
        self.input_x: list of tensor len = sentence_length, each tensor has
        shape = [batch_size, embed_size]
        return:
        """
        # Create a convolution + maxpool layer for each filter size
        num_filters_total = 0
        pooled_outputs = []
        W = []
        b = []
        # filter_size: 2/3/4
        for  filter_size in self.config.filter_sizes:
            with tf.name_scope("Conv-maxpool-%s" % filter_size):
                # Convolution Parameter
                # [2/3/4, 300, 1, 150]
                filter_shape = [filter_size, self.config.embedding_size, 1, self.config.feature_size]
                  
                # W.shape = [2/3/4, 300, 1, 150]
                W.append(tf.get_variable("W_%d" % filter_size, 
                        shape=filter_shape, dtype=tf.float32)) 
                # b_shape = [150]
                b.append(tf.get_variable("b_%d" % filter_size, 
                        shape=[self.config.feature_size], dtype=tf.float32))
 
               # Convolution Layer                              
                conv = tf.nn.conv2d(
                    self.feature,
                    W[-1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b[-1]), name="relu")

                # ??
                tf.add_to_collection("loss", tf.nn.l2_loss(W[-1]) + tf.nn.l2_loss(b[-1]))

                # Max-pooling over the outputs
                # ksize??
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)


        # Combine all the pooled features
        num_filters_total = self.config.feature_size * len(self.config.filter_sizes)
        triger_size = self.config.triger_size
        self.h_pool = tf.concat(axis=2, values=pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        with tf.name_scope("output"):
            W2 = tf.get_variable(
                "W2",
                shape=[ num_filters_total, self.config.triger_size])
            b2 = tf.get_variable(name="b2", shape=[self.config.triger_size], 
                                 dtype=tf.float32)
            tf.add_to_collection("loss", tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2))
            self.scores = tf.nn.xw_plus_b(self.h_drop, W2, b2, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * tf.add_n(tf.get_collection("loss")) / 2

                # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            '''self.scores = []
            losses = []
            poolers =  tf.split(1, self.config.sequence_length, self.h_drop)
            for x in range(len(poolers)):
                tmp = poolers[x]
                for i in range(self.config.triger_size - 1):
                    if x + i >= self.config.sequence_length:
                        tmp = tf.concat(1, (tmp, tf.zeros_like(poolers[0])))
                    else:
                        tmp = t

            self.predictions = tf.transpose(tf.argmax(self.scores, 2, name="predictions"))
            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                for score, label in zip(tf.unpack(self.scores),
                                        [tf.squeeze(x, [1]) for x in  
                                         tf.split(1, self.config.sequence_length, self.input_y)]):
                    loss = tf.nn.softmax_cross_entropy_with_logits(score, label)
                    losses.append(loss)
                self.loss = tf.reduce_mean(tf.reshape(tf.pack(losses), [-1])) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions , tf.argmax(self.input_y,2))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")'''
if __name__ == "__main__":
    cof = config()
    vectors = pickle.load(open("data_script/vector.bin", "rb"))
    ed = ed_model(cof, 31, vectors)
