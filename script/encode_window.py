import pickle
import numpy as np
import gensim, os
import tensorflow as tf

def create_document_iter(tokens):
    """
    transform the token list into token string, one article's tokens forms one string.
    """
    for doc in tokens:
        raw_doc = ""
        for word in doc:
            raw_doc += " " + word
        yield raw_doc.strip()

def encode_dictionary(input_iter, min_frequence=0, max_document_length=10000): 
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_document_length,
        min_frequence) # make vocabulary using tensorflow
    vocab_processor.fit(input_iter)
    return vocab_processor

def encode_window(tokens, anchors, vocab_processor):
    # enconde windows; labels: the anchor property of center word of window;
    # windows type: list, windows rank: 2, axis 0: central tokens' windows, axis 1: the tokens in the window
    # labels type: list, labels rank: 1, axis 0: each token's anchor type
    windows, window, labels = [], [], []
    unk = vocab_processor.vocabulary_._mapping["<UNK>"]
    j = 0
    for doc in tokens:
        for tok in np.arange(len(doc)):
            for i in np.arange(-15, 16): # window length is 31, index 0 is central word
                if i + tok < 0 or i + tok >= len(doc):
                    window.append(unk)
                else:
                    # add the token index into the window
                    window.append(vocab_processor.vocabulary_._mapping.get(doc[i + tok], unk))
            windows.append(window)
            labels.append(anchors[j][tok]) 
            #if anchors[j][tok] == 0: labels.append(0)
            #else: labels.append(1)
            window = []
        j += 1
    return windows, labels

def load_bin_vec(fname, vocab):
    """
    Load 300x1 word vecs from Google (Mikolov) word2vec. Return the list of word vector of corresponding tokens in the same indices.
    """
    word_vecs = np.zeros((len(vocab), 300))
    count = 0
    # vocab_bin = gensim.models.KeyedVectors.load_word2vec_format(
    #     os.path.join(os.path.dirname(__file__), fname), binary=True)
    vocab_bin = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    for word in vocab:
        if word in vocab_bin:
            count += 1
            word_vecs[vocab.index(word)]=(vocab_bin[word])
        else:
            word_vecs[vocab.index(word)] = (np.random.uniform(-0.25, 0.25, 300))
        print("found %d" %count)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

if __name__ == "__main__":
    tokens1 = pickle.load(open("./preprocessing/tokens_nw.bin", "rb"))
    tokens2 = pickle.load(open("./preprocessing/tokens_bn.bin", "rb"))
    anchors1 = pickle.load(open("./preprocessing/anchors_nw.bin", "rb"))
    anchors2 = pickle.load(open("./preprocessing/anchors_bn.bin", "rb"))
    # print(tokens1[0])
    # print(anchors1[0])
    input_iter = create_document_iter(tokens1 + tokens2) # add along axis 0
    vocab = encode_dictionary(input_iter) # vocab is the dictionary which maps the numbers to tokens
    # vocat_list: mapping list between indices and tokens
    vocab_list = list(vocab.vocabulary_._mapping.keys())
    # word_vecs = load_bin_vec("./preprocessing/GoogleNews-vectors-negative300.bin", vocab_list)
    # pickle.dump(word_vecs, open("./preprocessing/vector.bin", "wb"))

    # word_vecs = pickle.load(open("./preprocessing/vector.bin", "rb"))
    # winows中封装的是词的序号，vector.bin中的词向量是按照相同的序号排列的。
    windows1, labels1 = encode_window(tokens1, anchors1, vocab)
    # w1 = np.array(windows1)
    # l1 = np.array(labels1)
    # print(np.shape(w1))
    # print(np.shape(l1))
    windows2, labels2 = encode_window(tokens2, anchors2, vocab)    
    pickle.dump(windows1, open("./preprocessing/windows1.bin", "wb"))
    pickle.dump(labels1, open("./preprocessing/labels1.bin", "wb"))
    pickle.dump(windows2, open("./preprocessing/windows2.bin", "wb"))
    pickle.dump(labels2, open("./preprocessing/labels2.bin", "wb"))

    '''
    windows, labels = encode_window(tokens, anchors, dictionary)
    pickle.dump(windows, open("windows2.bin", "wb"))
    pickle.dump(labels, open("labels2.bin", "wb"))
    '''
    # tokens = pickle.load(open("tokens3.bin", "rb"))
    # anchors = pickle.load(open("anchors3.bin", "rb"))
    # # windows, labels = encode_window(tokens, anchors, dictionary)
    # windows, labels = encode_window(tokens, anchors, vocab)
    # pickle.dump(windows, open("windows3.bin", "wb"))
    # pickle.dump(labels, open("labels3.bin", "wb"))
    # tokens = pickle.load(open("tokens4.bin", "rb"))
    # anchors = pickle.load(open("anchors4.bin", "rb"))
    # # windows, labels = encode_window(tokens, anchors, dictionary)
    # windows, labels = encode_window(tokens, anchors, vocab)
    # pickle.dump(windows, open("windows4.bin", "wb"))
    # pickle.dump(labels, open("labels4.bin", "wb"))
    # pickle.dump(word_vecs, open("vector.bin", "wb"))
    
