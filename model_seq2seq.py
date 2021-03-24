"""

@author: ziming yan
"""

import tensorflow as tf
import numpy as np
from dataprocess import dataprocess
import sys
import math
import pickle
import os

input_num = 4096
hidden_num = 256
frame_num = 80
batch_size = 64
Epoch = 200

class S2VT:
    def __init__(self, input_num, hidden_num, frame_num = 0, max_caption_len = 50, lr = 1e-3, sampling = 0.8):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.frame_num = frame_num
        self.max_caption_len = max_caption_len
        self.learning_rate = lr
        self.sampling_prob = sampling
        self.saver = None
        self.vocab_num = None
        self.token = None
        
    def load_token(self):
        with open('data_tokenizer.pickle', 'rb') as f:
            self.token = pickle.load(f)
        self.vocab_num = len(self.token.word_index)
#         self.vocab_num=6136
    
    def build_model(self, feat, cap=None, cap_len=None, isTrain=True):
        W_top = tf.Variable(tf.random_uniform([self.input_num, self.hidden_num],-0.1,0.1), name='W_top')
        b_top = tf.Variable(tf.zeros([self.hidden_num]), name='b_top')
        W_btm = tf.Variable(tf.random_uniform([self.hidden_num,self.vocab_num],-0.1,0.1), name='W_btm')
        b_btm = tf.Variable(tf.zeros([self.vocab_num]),name='b_btm')
        embedding = tf.Variable(tf.random_uniform([self.vocab_num,self.hidden_num],-0.1,0.1), name='Embedding')
        batch_size = tf.shape(feat)[0]
        
        with tf.variable_scope('LSTMTop'):
            lstm_top = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_num, forget_bias=1.0, state_is_tuple=True)
            if isTrain:
                lstm_top = tf.contrib.rnn.DropoutWrapper(lstm_top, output_keep_prob=0.5)    
        with tf.variable_scope('LSTMBottom'):
            lstm_btm = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_num, forget_bias=1.0, state_is_tuple=True)
            if isTrain:
                lstm_btm = tf.contrib.rnn.DropoutWrapper(lstm_btm, output_keep_prob=0.5)
                
        if isTrain:
            feat = tf.nn.dropout(feat,0.5)
            cap_mask = tf.sequence_mask(cap_len,self.max_caption_len, dtype=tf.float32)
        feat = tf.reshape(feat,[-1,self.input_num])
        img_emb = tf.add(tf.matmul(feat,W_top),b_top)
        img_emb = tf.transpose(tf.reshape(img_emb,[-1, self.frame_num, self.hidden_num]),perm=[1,0,2])
                
        h_top = lstm_top.zero_state(batch_size, dtype=tf.float32)
        h_btm = lstm_top.zero_state(batch_size, dtype=tf.float32)
        
        pad = tf.ones([batch_size, self.hidden_num])*self.token.texts_to_sequences(['<PAD>'])[0][0]
        
        for i in range(frame_num):
            with tf.variable_scope('LSTMTop'):
                output_top, h_top = lstm_top(img_emb[i,:,:],h_top)
            with tf.variable_scope('LSTMBottom'):
                output_btm, h_btm = lstm_btm(tf.concat([pad,output_top],axis=1),h_top)
                
        logit = None
        logit_list = []
        cross_entropy_list = []
        
        
        for i in range(0, self.max_caption_len):
            with tf.variable_scope('LSTMTop'):
                output_top, h_top = lstm_top(pad, h_top)

            if i == 0:
                with tf.variable_scope('LSTMBottom'):
                    bos = tf.ones([batch_size, self.hidden_num])*self.token.texts_to_sequences(['<BOS>'])[0][0]
                    bos_btm_input = tf.concat([bos, output_top], axis=1)
                    output_btm, h_btm = lstm_btm(bos_btm_input, h_btm)
            else:
                if isTrain:
                    if np.random.uniform(0,1,1) < self.sampling_prob:
                        input_btm = cap[:,i-1]
                    else:
                        input_btm = tf.argmax(logit, 1)
                else:
                    input_btm = tf.argmax(logit, 1)
                btm_emb = tf.nn.embedding_lookup(embedding, input_btm)
                with tf.variable_scope('LSTMBottom'):
                    input_btm_emb = tf.concat([btm_emb, output_top], axis=1)
                    output_btm, h_btm = lstm_btm(input_btm_emb, h_btm)
                    
            logit = tf.add(tf.matmul(output_btm, W_btm), b_btm)
            logit_list.append(logit)
            
            if isTrain:
                labels = cap[:, i]
                one_hot_labels = tf.one_hot(labels, self.vocab_num, on_value = 1, off_value = None, axis = 1) 
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=one_hot_labels)
                cross_entropy = cross_entropy * cap_mask[:, i]
                cross_entropy_list.append(cross_entropy)
        
        if isTrain:
            cross_entropy_list = tf.stack(cross_entropy_list, 1)
            loss = tf.reduce_sum(cross_entropy_list, axis=1)
            loss = tf.divide(loss, tf.cast(cap_len, tf.float32))
            loss = tf.reduce_mean(loss, axis=0)

        logit_list = tf.stack(logit_list, axis = 0)
        logit_list = tf.reshape(logit_list, (self.max_caption_len, batch_size, self.vocab_num))
        logit_list = tf.transpose(logit_list, [1, 0, 2])
        
        if isTrain:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            train_op = optimizer.minimize(loss)
        else:
            train_op = None
            loss = None
            
        pred_op = tf.argmax(logit_list, axis=2)
        return train_op, loss, pred_op, logit_list

def train(train_dict,train_label_dict):

    print("Starting loading dataset")
    train_set = dataprocess(train_dict,batch_size,label_dir=train_label_dict)
    train_set.data_token()
    train_set.save_token()
    train_set.process_data()
    print("Finishing data preprocess")
    
    graph_train = tf.Graph()
    with graph_train.as_default():
        model = S2VT(input_num, hidden_num, frame_num)
        model.load_token()
        feat = tf.placeholder(tf.float32, [None, frame_num, input_num], name='features')
        cap = tf.placeholder(tf.int32, [None, 50], name='caption')
        cap_len = tf.placeholder(tf.int32, [None], name='captionLength')
        train_op, loss_op, pred_op, logit_list_op = model.build_model(feat, cap, cap_len, True)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=3)
    sess= tf.Session(graph=graph_train)
    train_loss = []
    batch_total = int(math.ceil(train_set.data_size / batch_size))
    sess.run(init)
   
    print('strating training')
    for epoch in range(Epoch):
        train_set.shuffle()
        for i in range(batch_total):
            id_batch, feat_batch, caption_batch, caption_len_batch, = train_set.next_batch()
            sess.run(train_op, feed_dict={feat: feat_batch, cap: caption_batch, cap_len: caption_len_batch})
        loss = sess.run(loss_op, feed_dict={feat: feat_batch, cap: caption_batch, cap_len: caption_len_batch})
        train_loss.append(loss)
        print("Epoch {:d}:     ".format(epoch+1),  " Loss: ", loss)
    model_path = saver.save(sess, './model', global_step=Epoch * batch_total)
    print("Model saved: ", model_path)
    print("Finish training")
    
def test(test_dict):
    
    print('Loading testing data'+"\n")
    test_set = dataprocess(test_dict,batch_size)
    test_set.load_token()
    test_set.process_feat_data() 
    print('Test data loaded successfully.')
    
    graph_test = tf.Graph()
    with graph_test.as_default():
        model = S2VT(input_num,hidden_num,frame_num)
        model.load_token()
        feat = tf.placeholder(tf.float32, [None, frame_num, input_num], name='features')
        _, _, pred_op, logit_list_op = model.build_model(feat, isTrain=False)
        saver = tf.train.Saver(max_to_keep=3)
    sess= tf.Session(graph=graph_test)
    curr_dict = os.getcwd()
    
    print('Load model from:')
    print(curr_dict,'/model/\n')
    latest_checkpoint = tf.train.latest_checkpoint(curr_dict+'/model/')
    saver.restore(sess, latest_checkpoint)
    txt = open(curr_dict+'/test_result.txt', 'w+')
    batch_num = int(math.ceil(test_set.data_size/batch_size))
    eos = model.token.texts_to_sequences(['<EOS>'])[0][0]
    eos_idx = model.max_caption_len
    for i in range(batch_num):
        id_batch, feat_batch = test_set.next_feat_data()
        prediction = sess.run(pred_op,feed_dict={feat:feat_batch})
        for j in range(len(feat_batch)):
            for k in range(model.max_caption_len):
                if prediction[j][k]== eos:
                    eos_idx = k
                    break
            cap_output = model.token.sequences_to_texts([prediction[j][0:eos_idx]])[0]
            txt.write(id_batch[j] + "," + str(cap_output) + "\n")
    txt.close()
    print('Testing Output:',curr_dict+'/test_result.txt'+"\n")
    
    
if __name__ == '__main__':
    
#     # model training:
    
#     train_dict="./MLDS_hw2_1_data/training_data/feat/"
#     train_label_dir="./MLDS_hw2_1_data/training_label.json"
#     train(train_dict,train_label_dir)

    # model testing: 
    test_dict = sys.argv[1]
    test(test_dict)
