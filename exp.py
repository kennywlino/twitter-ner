# import necessary packages
from collections import defaultdict
import time
import random
import dynet as dy
import numpy as np

# defines dicts to convert words and tags into indices
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

# takes a word index and returns word
def i2w(index):
    return list(w2i.keys())[list(w2i.values()).index(index)]

# takes a tag index and returns tag
def i2t(index):
     return list(t2i.keys())[list(t2i.values()).index(index)]
     
def read_dataset(filename):
    with open(filename, "r") as f:
        data_list = []
        sent_list = []
        for line in f:
            if len(line.strip()) != 0:
                word, tag = line.strip().split("\t")
                sent_list.append((w2i[word], t2i[tag]))
            else:
                if len(sent_list) != 0:
                    data_list.append(sent_list)
                sent_list = []
        return data_list

# [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 1), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0)]

train = read_dataset("wnut17/data/train")
dev = read_dataset("wnut17/data/dev")
train = train + dev

# freezes the w2i dict
w2i = defaultdict(lambda: UNK, w2i)

nwords = max(w2i.values()) + 1 # used to exclude extra UNK
ntags = len(t2i)

test = read_dataset("wnut17/data/test")

# Start DyNet and define trainer
model = dy.Model()
trainer = dy.AdamTrainer(model)

# Define the model
EMB_SIZE = 64
HID_SIZE = 64
W_emb = model.add_lookup_parameters((nwords, EMB_SIZE))  # Word embeddings
lstm_builders = [dy.LSTMBuilder(1, EMB_SIZE, HID_SIZE, model), 
                 dy.LSTMBuilder(1, EMB_SIZE, HID_SIZE, model)] # fwd and bwd LSTM
W_sm = model.add_parameters((ntags, 2 * HID_SIZE))  # Softmax weights
b_sm = model.add_parameters((ntags))  # Softmax bias

def build_tagging_graph(sent):
    '''
    Builds the comp graph for the model with:
    * Embeddings
    * BiLSTM
    @return list of error for each tag
    '''
    dy.renew_cg()
    fwd_init, bwd_init = [b.initial_state() for b in lstm_builders]
    word_embs = [dy.lookup(W_emb, word) for word, tag in sent]
    
    fwd_embs = [x.output() for x in fwd_init.add_inputs(word_embs)]
    bwd_embs = [x.output() for x in bwd_init.add_inputs(reversed(word_embs))]
    
    W_sm_exp = dy.parameter(W_sm)
    b_sm_exp = dy.parameter(b_sm)
    
    errs = []
    for (word, tag), f_rep, b_rep in zip(sent, fwd_embs, reversed(bwd_embs)):
        complete_rep = dy.concatenate([f_rep, b_rep]) # complete rep of word from LSTM
        predicted = W_sm_exp * complete_rep + b_sm_exp
        err = dy.pickneglogsoftmax(predicted, tag)
        errs.append(err)
        dy.print_graphviz()
        break
    return dy.esum(errs)
    
def tag_sent(sent):
    '''
    Builds the comp graph for the model with:
    * Embeddings
    * BiLSTM
    @ return list of (word, predicted labels)
    '''
    dy.renew_cg()
    fwd_init, bwd_init = [b.initial_state() for b in lstm_builders]
    word_embs = [dy.lookup(W_emb, word) for word, tag in sent]
    
    fwd_embs = [x.output() for x in fwd_init.add_inputs(word_embs)]
    bwd_embs = [x.output() for x in bwd_init.add_inputs(reversed(word_embs))]
    
    W_sm_exp = dy.parameter(W_sm)
    b_sm_exp = dy.parameter(b_sm)
    
    predicted_labels = []
    for (word, tag), f_rep, b_rep in zip(sent, fwd_embs, reversed(bwd_embs)):
        complete_rep = dy.concatenate([f_rep, b_rep]) # complete rep of word from LSTM
        scores = (W_sm_exp * complete_rep + b_sm_exp).npvalue()
        predict = np.argmax(scores)
        predicted_labels.append((word, predict))
    return predicted_labels

for ITER in range(50):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    if ITER == 49:
        out = open('predicted.txt', 'w')
    for sent in train:
        sent_error = build_tagging_graph(sent)
        # train_loss += sent_error.value()
        sent_error.backward()
        trainer.update()
        break
    print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start))
    break
    total_acc = 0.0
    for sent in test:
        p_labels = tag_sent(sent)
        g_labels = [tags for word, tags in sent]
        test_correct = 0
        for i, p_g in enumerate(zip(p_labels, g_labels)):
            word = p_g[0][0]
            predicted = p_g[0][1]
            gold = p_g[1]
            if predicted == gold:
                test_correct += 1
            if ITER == 49:
                out.write(i2w(word) + '\t' + i2t(predicted) + '\n')
                if i == (len(p_g) - 1):
                    out.write('\n')
        total_acc += test_correct / len(g_labels)
    print("iter %r: test acc=%.4f" % (ITER, total_acc / len(test)))