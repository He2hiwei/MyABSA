#!/usr/bin/env python3
"""
Created on Tue Sep 18 21:27:38 2018

@author: He2hiwei
"""

from gensim import corpora
from utils import glove 
import numpy as np
import torch
import re
import torchtext.data as data

EMBEDING_FILE_PATH = "/Users/apple/AVEC2017/utils/datasets/glove.840B.300d.txt"

# def absa_text2num(train_df, test_df, 
#     Glove_PATH='/Users/aple/github/my-GCAE-master/acsa-restaurant-2014/acsa_train.json', EMBED_DIM=300):
#     """
#     Convert text data into numeric type for Aspect based sentiment 
#     analysis (ABSA) task
    
#     ----
#     Input:
#     train_df — columns=[sentence, aspect, sentiment]
#     test_df — columns=[sentence, aspect, sentiment]
    
#     Output: wrapped in a dict
#     embed_lookup_tab — word embedding for corpus(sentences and aspects of train and test)
#     # token2id_dict — dict, word token to word id. The word order is the same as embed_lookup_tab.
#     # id2token_dict — dict, word id to word token. The word order is the same as embed_lookup_tab.
#     senti_id2token_dict --- dict, word id to word token for sentiment id
#     train_sentence — list, (N_train_sample, sentence_word_num), sentence_word_num is not fixed. 
#                     The word id for sentence of each train sample.
#     train_aspect — list, (N_train_sample, aspect_word_num), aspect_word_num is not fixed. 
#                     The word id for aspect of each train sample.
#     train_senti — torch.doubleTensor, (N_train_sample, 1). sentiment label(int) for each train sample. 
#     test_sentence — similar to train_sentence
#     test_aspect — similar to train_aspect
#     test_senti — similar to train_senti
#     """
#     # Build corpus for sentence and aspcet of train and test dataset
#     train_sentence_token = []
#     train_aspect_token = []

#     test_sentence_token = []
#     test_aspect_token = []
    
#     sentence_aspect_corpus = []

#     ## Convert sentence and phrase to token. 
#     ## Aggregate tokens of differernt dataset and content
#     for dataset in ['train','test']:
#         for content in ['sentence','aspect']:    
#             for sentence in eval(dataset + '_df.' + content).values:               
#                 if dataset == 'sentence':
#                     sentence = re.sub('\.',' .',sentence)
#                     sentence = re.sub('\,',' ,',sentence)
                    
#                 eval(dataset + '_' + content + '_token').append([word for word in sentence.lower().split()])
#             sentence_aspect_corpus += eval(dataset + '_' + content + '_token')
            
#     # build word dictionry
#     ### for both sentence and aspect
#     embed_dictionary = corpora.Dictionary(sentence_aspect_corpus)
#     embed_token2id_dict = embed_dictionary.token2id
#     ### for sentiment label
#     senti_dictionary = corpora.Dictionary([np.unique(train_df.sentiment.values).tolist()])
#     senti_token2id_dict = senti_dictionary.token2id
# #    senti_id2token_dict = senti_dictionary.id2token
    
#     # token2id for sentence and aspect, word2id for sentiment label
#     train_sentence = [[embed_token2id_dict[word] for word in sentence] 
#                             for sentence in train_sentence_token]
    
#     train_aspect = [[embed_token2id_dict[word] for word in sentence] 
#                             for sentence in train_aspect_token]
    
#     train_senti = torch.from_numpy(train_df.sentiment.map(lambda x: senti_token2id_dict[x]).values)

#     test_sentence = [[embed_token2id_dict[word] for word in sentence] 
#                             for sentence in test_sentence_token]
    
#     test_aspect = [[embed_token2id_dict[word] for word in sentence] 
#                             for sentence in test_aspect_token]
    
#     test_senti = torch.from_numpy(test_df.sentiment.map(lambda x: senti_token2id_dict[x]).values)
    
#     # glove embedding for corpus of sentence and aspect
#     embed_lookup_tab = glove.loadWordVectors(embed_token2id_dict,Glove_PATH,EMBED_DIM)
    
#     return embed_lookup_tab, train_sentence, train_aspect, train_senti, \
#             test_sentence, test_aspect, test_senti,senti_token2id_dict

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def load_glove_embedding(stoi, uniform_scale=0.10,filePath=EMBEDING_FILE_PATH, dim=300):
    # faster then ever version before
    # stoi –  token strings to numerical identifiers.
    wordVectors = np.random.uniform(uniform_scale,-uniform_scale,(len(stoi), dim))
    wordVectors[stoi['<pad>']] = np.zeros(dim, dtype=np.float32)
    with open(filePath) as ifs:
        for line in ifs:
            if not line:
                continue

            token = line[:line.find(' ')].strip()
            if token not in stoi:
                continue
            
            wordVectors[stoi[token]] = np.fromstring(line[line.find(' '):].strip(),sep=' ', dtype=np.float32)
    return wordVectors

class SemEval(data.Dataset):
    # sort_key (callable) – A key to use for sorting dataset examples
    # for batching together examples with similar lengths to minimize padding.
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_as_field, sm_field, input_data, **kwargs):
        """Defines a dataset composed of Examples along with its Fields.

        Arguments:
            text_as_field: The field that will be used for text data and aspect data.
            sm_field: The field that will be used for sentiment data.
            input_data: json format. List of dicts. The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of data.Dataset.
        """

        text_as_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_as_field), ('aspect', text_as_field), ('sentiment', sm_field)]

        # list -> example
        examples = []
        for e in input_data:
            examples.append(data.Example.fromlist([e['sentence'], e['aspect'], e['sentiment']], fields))
        # data.Dataset __init__  input: examples (list(Example)) – The examples in this dataset.
        super(SemEval, self).__init__(examples, fields, **kwargs)

    
    
def absa_text2num_torchtext(train_json, test_json, UNIFORM_SCALE,
    EMBEDING_FILE_PATH=EMBEDING_FILE_PATH, EMBED_DIM=300):
    """
    Convert text data into numeric type for Aspect based sentiment 
    analysis (ABSA) task
    
    ----
    Input:
    train_json
    test_json
    
    Output: wrapped in a dict
    embed_lookup_tab — word embedding for corpus(sentences and aspects of train and test)
    # token2id_dict — dict, word token to word id. The word order is the same as embed_lookup_tab.
    # id2token_dict — dict, word id to word token. The word order is the same as embed_lookup_tab.
    senti_id2token_dict --- dict, word id to word token for sentiment id
    train_sentence — list, (N_train_sample, sentence_word_num), sentence_word_num is not fixed. 
                    The word id for sentence of each train sample.
    train_aspect — list, (N_train_sample, aspect_word_num), aspect_word_num is not fixed. 
                    The word id for aspect of each train sample.
    train_senti — torch.doubleTensor, (N_train_sample, 1). sentiment label(int) for each train sample. 
    test_sentence — similar to train_sentence
    test_aspect — similar to train_aspect
    test_senti — similar to train_senti
    """
    """
    torchtext version: json --> tokenization --> vocab --> numericalize --> embedding lookup
    """
    # define field
    text_as_field = data.Field(lower=True, tokenize='moses')
    sm_field = data.Field(sequential=False)

    # construct data.dataset object for target data
    train_data = SemEval(text_as_field,sm_field,train_json)
    test_data = SemEval(text_as_field,sm_field,test_json)

    # Construct the Vocab object for this field from one or more datasets.
    # Otherwise: AttributeError: 'Field' object has no attribute 'vocab'
    # keypoint: one field could be applied to multi datasets and
    #  one datasets could have multi fields
    text_as_field.build_vocab(train_data, test_data)
    sm_field.build_vocab(train_data, test_data)

    # list(token strings) -> Glove Embedding 2D array: tokens_num x embed_dim
    print("list(token strings) -> Glove Embedding 2D array")
    word_vecs = load_glove_embedding(text_as_field.vocab.stoi, UNIFORM_SCALE,EMBEDING_FILE_PATH, EMBED_DIM)
    # 2D array -> torch.FloatTensor
    text_as_embedding = torch.from_numpy(word_vecs.astype(np.float32))

    return train_data, test_data, text_as_embedding, sm_field.vocab.itos


# Test code
# import torchtext.data as data
# import torch
# import torch.nn as nn
# import re
# import numpy as np

# DEFAULT_FILE_PATH = "/Users/apple/AVEC2017/utils/datasets/glove.840B.300d.txt"

# def clean_str(string):
#     """
#     Tokenization/string cleaning for all datasets except for SST.
#     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
#     """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip()

# def load_glove_embedding(stoi, uniform_scale=0.10,filePath=DEFAULT_FILE_PATH, dim=300):
#     # faster then ever version before
#     # stoi –  token strings to numerical identifiers.
#     wordVectors = np.random.uniform(uniform_scale,-uniform_scale,(len(stoi), dim))
#     wordVectors[stoi['<pad>']] = np.zeros(dim, dtype=np.float32)
#     with open(filePath) as ifs:
#         for line in ifs:
#             if not line:
#                 continue

#             token = line[:line.find(' ')].strip()
#             if token not in stoi:
#                 continue
            
#             wordVectors[stoi[token]] = np.fromstring(line[line.find(' '):].strip(),sep=' ', dtype=np.float32)
#     return wordVectors

# class SemEval(data.Dataset):
#     # sort_key (callable) – A key to use for sorting dataset examples
#     # for batching together examples with similar lengths to minimize padding.
#     @staticmethod
#     def sort_key(ex):
#         return len(ex.text)

#     def __init__(self, text_as_field, sm_field, input_data, **kwargs):
#         """Defines a dataset composed of Examples along with its Fields.

#         Arguments:
#             text_as_field: The field that will be used for text data and aspect data.
#             sm_field: The field that will be used for sentiment data.
#             input_data: json format. List of dicts. The examples contain all the data.
#             Remaining keyword arguments: Passed to the constructor of data.Dataset.
#         """

#         text_as_field.preprocessing = data.Pipeline(clean_str)
#         fields = [('text', text_as_field), ('aspect', text_as_field), ('sentiment', sm_field)]

#         # list -> example
#         examples = []
#         for e in input_data:
#             examples.append(data.Example.fromlist([e['sentence'], e['aspect'], e['sentiment']], fields))
#         # data.Dataset __init__  input: examples (list(Example)) – The examples in this dataset.
#         super(SemEval, self).__init__(examples, fields, **kwargs)

# # json data
# predict_test = [{"aspect": "food",
#                     "sentiment": "positive",
#                     "sentence": "good food in cute - though a bit dan"},
#                 {"aspect": "service",
#                     "sentiment": "negative",
#                     "sentence": "good food in cute - though a bit dan"},
#                 {"aspect": "service",
#                     "sentiment": "negative",
#                     "sentence": "good food in cute - though a bit dank - little hangout, but service terrible"},
#                 {"aspect": "service",
#                     "sentiment": "positive",
#                     "sentence": "good food in cute - though a bit dank - little hangout, but service terrible"}
#                 ]
# # define field
# text_as_field = data.Field(lower=True, tokenize='moses')
# sm_field = data.Field(sequential=False)

# # construct data.dataset object for target data
# train_data = SemEval(text_as_field,sm_field,predict_test)

# # Construct the Vocab object for this field from one or more datasets.
# # Otherwise: AttributeError: 'Field' object has no attribute 'vocab'
# # keypoint: one field could be applied to multi datasets and
# #  one datasets could have multi fields
# text_as_field.build_vocab(train_data)
# sm_field.build_vocab(train_data)

# # stoi – A collections.defaultdict instance mapping token strings to numerical identifiers.
# stoi = text_as_field.vocab.stoi
# # itos – A list of token strings indexed by their numerical identifiers. 
# itos = text_as_field.vocab.itos
# # list(token strings) -> Glove Embedding 2D array: tokens_num x embed_dim
# word_vecs = load_glove_embedding(stoi, 0.1,DEFAULT_FILE_PATH, 300)
# # 2D array -> torch.FloatTensor
# text_as_embedding = torch.from_numpy(word_vecs.astype(np.float32))
# # Build embedding look-up table for torch model  
# model_embed = nn.Embedding(text_as_embedding.size()[0],text_as_embedding.size()[1])
# model_embed.weight = nn.Parameter(text_as_embedding, requires_grad=True)

# """
# torchtext.data.Iterator.splits is a classmethod

# Variable:
# repeat – Whether to repeat the iterator for multiple epochs.
# sort_key – A key to use for sorting examples in order to batch
#         together examples with similar lengths and minimize padding.
#         The sort_key provided to the Iterator constructor overrides the
#         sort_key attribute of the Dataset, or defers to it if None
# shuffle – Whether to shuffle examples between epochs.

# output: Iterator. Different batches may have different token_num. 
# """
# train_iter, = data.Iterator.splits((train_data,),batch_sizes=(2,), 
#                                     repeat=False, sort_key=lambda x: len(x.text),shuffle=True)

# EPOCH = 3
# for i in range(1, EPOCH+1):
#     for batch in train_iter:
#         # outout: torch.LongTensor
#         feature, aspect, target = batch.text, batch.aspect, batch.sentiment

#         # transpose(batch firest) and change the variable: 
#         # batch_size x token_num(plus <pad>)
#         feature.data.t_()
#         aspect.data.t_()

#         # numerical identifiers -> embedding, use look-up table
#         # batch_size x token_num(pluas <pad>) x embed_dim
#         feature_embeded = model_embed(feature)
#         aspect_embeded = model_embed(aspect)

#         # index align
#         target.data.sub_(1)

#         print(feature.size())