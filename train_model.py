import torch
import torch.nn as nn 
import pandas as pd
import numpy as np
import json
import re 
from gensim import corpora
from cnn_gate_aspect_model_acsa import CNN_Gate_Aspect_Text
from torch.autograd import Variable
from utils import glove 


PROJECT_PATH = '/Users/apple/github/my-GCAE-master'
Glove_PATH = '/Users/apple/AVEC2017/utils/datasets/glove.840B.300d.txt'
EMBED_DIM = 300
#SENTENCE_INPUT_LENGTH = 50

class Parameter:
    def __init__(self):
        self.embed_num = 100
        self.embed_dim = 10
        self.class_num = 10
        self.aspect_num = 10
#        self.aspect_embed_dim = 10
        
        self.embedding = torch.randn(100,10)
        self.aspect_embedding = torch.randn(10,10)

        self.kernel_num = 10
        self.kernel_sizes = [2,3,4]

# ## data processing
# restaurant_train  = pd.read_json(PROJECT_PATH + '/acsa-restaurant-2014/acsa_train.json')
# restaurant_test  = pd.read_json(PROJECT_PATH + '/acsa-restaurant-2014/acsa_test.json')

# restaurant_train  = pd.read_json(PROJECT_PATH + '/acsa-restaurant-2014/acsa_hard_train.json')
# restaurant_test  = pd.read_json(PROJECT_PATH + '/acsa-restaurant-2014/acsa_hard_test.json')


# restaurant_train  = pd.read_json(PROJECT_PATH + '/acsa-restaurant-large/acsa_train.json')
# restaurant_test  = pd.read_json(PROJECT_PATH + '/acsa-restaurant-large/acsa_test.json')
        
restaurant_train  = pd.read_json(PROJECT_PATH + '/acsa-restaurant-large/acsa_hard_train.json')
restaurant_test  = pd.read_json(PROJECT_PATH + '/acsa-restaurant-large/acsa_hard_test.json')


cnf_titile = 'acsa-restaurant-hard'

# ---------------
# new code
from utils import absa_text2num
import torch.utils.data as Data

embed_lookup_tab, train_sentence, \
train_aspect, train_senti, \
test_sentence, test_aspect, \
test_senti, senti_token2id_dict  = absa_text2num.absa_text2num(restaurant_train,restaurant_test)



embed_lookup_tab_zero_pad = np.concatenate((embed_lookup_tab, np.zeros((1,EMBED_DIM))),axis=0)

# Convert list of lists with different lengths to a numpy array

train_sentence_array = len(embed_lookup_tab) * (np.ones([len(train_sentence),
                                       len(max(train_sentence + test_sentence,
                                               key = lambda x: len(x)))]).astype(int))
for i,j in enumerate(train_sentence):
    train_sentence_array[i][0:len(j)] = j
    
test_sentence_array = len(embed_lookup_tab) * (np.ones([len(test_sentence),
                                       len(max(train_sentence + test_sentence,
                                               key = lambda x: len(x)))]).astype(int))
for i,j in enumerate(test_sentence):
    test_sentence_array[i][0:len(j)] = j

# parameter setting
args = Parameter()
args.embed_num = len(embed_lookup_tab_zero_pad) + 1 # the last row is zero padding(TODO: require_grad=False)
args.embed_dim = EMBED_DIM
args.class_num = train_senti.max() + 1

args.aspect_num = len(np.unique(train_aspect))
args.aspect_embed_dim = EMBED_DIM

args.embedding = torch.from_numpy(embed_lookup_tab_zero_pad) # embedding for sentence and aspect input
#args.aspect_embedding = torch.from_numpy(aspect_glove_embeds)

args.kernel_num = 100
args.kernel_sizes = [3,4,5] # n-gram


torch.manual_seed(1)    # reproducible

BATCH_SIZE = 32

x = torch.linspace(0, len(train_sentence)-1, len(train_sentence)).int()    # this is index of x data (torch tensor)
y = torch.linspace(0, len(train_sentence)-1, len(train_sentence)).int()      # this is index of y data (torch tensor)

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)

ACSA_model = CNN_Gate_Aspect_Text(args)

optimizer = torch.optim.Adagrad(ACSA_model.parameters(), lr=1e-2, weight_decay=0, lr_decay=0)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

train_sentence_tensor = Variable(torch.from_numpy(train_sentence_array))
train_aspect_tensor = Variable(torch.from_numpy(np.array(train_aspect)))

test_sentence_tensor = Variable(torch.from_numpy(test_sentence_array))
test_aspect_tensor = Variable(torch.from_numpy(np.array(test_aspect)))

# ========================== print parameters ===========
params = list(ACSA_model.parameters())
k = 0
for i in params:
    l = 1
    print("该层的结构：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("该层参数和：" + str(l))
    k = k + l
print("总参数数量和：" + str(k))

train_acc_list = []
test_acc_list = []
for epoch in range(10):   # train entire dataset 3 times
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
        pred = ACSA_model(train_sentence_tensor[batch_x],train_aspect_tensor[batch_x])
        loss = loss_func(pred, Variable(train_senti[batch_y]))
        
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        print('Epoch: ', epoch, '| Step: ', step, '| loss: ', loss.data[0])
    
    print(15 * '=' + ' Accuracy for one epoch '+ 15 * '=')
    train_pred = ACSA_model(train_sentence_tensor,train_aspect_tensor)
    train_pred = torch.max(train_pred, 1)[1]
    train_acc = sum(train_senti.numpy() == train_pred.data.numpy()) / len(train_pred)
    train_acc_list.append(train_acc)
    print('Train acc: %.2f' % train_acc)
     
    test_pred = ACSA_model(test_sentence_tensor,test_aspect_tensor)
    test_pred = torch.max(test_pred, 1)[1]
    test_acc = sum(test_senti.numpy() == test_pred.data.numpy()) / len(test_pred)
    test_acc_list.append(test_acc)
    print('Test acc: %.2f' % test_acc)
    
# save variable
import pickle
# Saving the objects:
with open(cnf_titile + '_acc.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([test_acc_list,train_acc_list], f)
   
# Getting back the objects:
with open(cnf_titile + '_acc.pkl','rb') as f:  # Python 3: open(..., 'rb')
    test_acc_list,train_acc_list = pickle.load(f)

# Plot normalized confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

cnf_matrix = confusion_matrix(test_senti.numpy(), test_pred.data.numpy())

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
plt.figure(dpi=300)
classes = [list(senti_token2id_dict.keys())[list(senti_token2id_dict.values()).index(i)] for i  in range(args.class_num)]
#classes = ['Conflict','Negative','Neutral','Positive']
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=False,
                      title=cnf_titile)
plt.savefig(cnf_titile + '_cnf.png')
plt.show(block=False)

# plot table
fig,ax = plt.subplots(1,1,dpi=100)
test_baseline = 0.7075
# test_val = 0.71
ax.plot(train_acc_list,label='train Acc.')
ax.plot(test_acc_list,label='test Acc.',color='purple')
ax.axhline(y=test_baseline,linewidth=1, color='r',label='baseline')
# ax.axhline(y=test_val,linestyle='dashed',linewidth=1,color='black',)
# ax.set_ylim(0.7,1)
plt.title(cnf_titile)
plt.xlabel('Epoch')
plt.ylabel('Acc.')
ax.legend()
ax.text(9,test_baseline,str(test_baseline))
# ax.text(9,test_val,str(test_val))
# Get current tick locations and append 271 to this array
# y_ticks = np.append(ax.get_yticks(),test_baseline)
# Set xtick locations to the values of the array `x_ticks`
# ax.set_yticks(y_ticks)
plt.savefig(cnf_titile + '_acc.png')
plt.show(block=False)
