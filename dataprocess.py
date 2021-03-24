from tensorflow.keras.preprocessing.text import Tokenizer,text_to_word_sequence
import numpy as np
import json
import pickle
import os

filters = '`","?!/.()'

class dataprocess():
    def __init__(self,feat_dir,batch_size,label_dir=None,max_caption_len=50):
        self.feat_dir=feat_dir
        self.label_dir=label_dir
        self.batch_size=batch_size
        self.max_caption_len=max_caption_len
        self.data_size=None
        self.data_idx=None
        self.vocab_size = 0
        self.batch_idx=0
        self.id_list = None
        self.cap_list = None
        self.cap_length_list = None
        self.filters = '`","?!/.()'
        
    def data_token(self):
        self.tk=Tokenizer(filters=self.filters,split=" ")
        cap_list=[]
        with open(self.label_dir,"r") as f:
            raw_data=json.load(f)
        for lab in raw_data:
            for caption in lab["caption"]:
                cap_list.append(caption)
        self.tk.fit_on_texts(cap_list)
        self.vocab_size=len(self.tk.word_index)
        self.tk.fit_on_texts(['<PAD>','<BOS>','<EOS>','<UNK>'])
        
    def process_data(self):
        pad = self.tk.texts_to_sequences(['<PAD>'])[0]
        id_list = []
        cap_list = []
        cap_length_list = []
        self.feat_data = {}
        with open(self.label_dir) as f:
            raw_data = json.load(f)
        for vid in raw_data:
            vid_id = vid['id']
            self.feat_data[vid_id] = np.load(self.feat_dir + vid_id + '.npy')
            for caption in vid['caption']:
                words = text_to_word_sequence(caption)
                for i in range(len(words)):
                    if words[i] not in self.tk.word_index:
                        words[i] = '<UNK>'
                words.append('<EOS>')
                one_hot = self.tk.texts_to_sequences([words])[0]
                cap_length = len(one_hot)
                one_hot += pad * (self.max_caption_len - cap_length)
                id_list.append(vid_id)
                cap_list.append(one_hot)
                cap_length_list.append(cap_length)
                
        self.id_list = np.array(id_list)
        self.cap_list = np.array(cap_list)
        self.cap_length_list = np.array(cap_length_list)
        self.data_size = len(self.cap_list)
        self.data_idx = np.arange(self.data_size,dtype=np.int)
        
    def next_batch(self):
        if self.batch_idx+self.batch_size>self.data_size:
            idx=self.data_idx[self.batch_idx:]
            self.batch_idx=0
        else:
            idx=self.data_idx[self.batch_idx:self.batch_idx+self.batch_size]
            self.batch_idx=self.batch_idx+self.batch_size
        
        video_name_batch=self.id_list[idx]
        feat_data_batch=[]
        caption_batch=self.cap_list[idx]
        caption_len=self.cap_length_list[idx]
        for video in video_name_batch:
            feat_data_batch.append(self.feat_data[video])
        feat_data_batch=np.array(feat_data_batch)
        
        return video_name_batch,feat_data_batch,caption_batch,caption_len
    
    def process_feat_data(self):
        idex_list=[]
        self.feat_data={}
        for file in os.listdir(self.feat_dir):
            video=os.path.splitext(file)
            if video[-1]=='.npy':
                video_name=video[0]
                self.feat_data[video_name]=np.load(self.feat_dir+file)
                idex_list.append(video_name)
        self.id_list=np.array(idex_list)
        self.data_size=len(self.id_list)
        self.data_idx=np.arange(self.data_size,dtype=np.int)
        
    def next_feat_data(self):
        if self.batch_idx+self.batch_size>self.data_size:
            idx=self.data_idx[self.batch_idx:]
            self.batch_idx=0
        else:
            idx=self.data_idx[self.batch_idx:self.batch_idx+self.batch_size]
            self.batch_idx=self.batch_idx+self.batch_size
        
        video_name_batch=self.id_list[idx]
        feat_data_batch=[]
        for video in video_name_batch:
            feat_data_batch.append(self.feat_data[video])
        feat_data_batch=np.array(feat_data_batch)
        return video_name_batch,feat_data_batch
    
    def load_token(self):
        with open("data_tokenizer.pickle","rb") as f:
            self.tk=pickle.load(f)
        self.vocab_size=len(self.tk.word_index)
        
    def save_token(self):
        with open("data_tokenizer.pickle","wb") as f:
            pickle.dump(self.tk,f,protocol=pickle.HIGHEST_PROTOCOL)
            
    def shuffle(self):
        np.random.shuffle(self.data_idx)
        
    def get_word_index(self):
        return self.tk.word_index
    
    def text_sequence(self,txt):
        return self.tk.texts_to_sequences(txt)
    
    def get_token(self):
        return self.tk