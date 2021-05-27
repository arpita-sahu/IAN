import torch
import numpy
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from absl import flags
from absl import app
from utils import get_data_info, read_data, load_word_embeddings
# from model import IAN
from model_pytorch import IAN
from evals import *
import os
import time
import math

#setting os.environ

#configuration settings

FLAGS = flags.FLAGS
flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
flags.DEFINE_integer('n_class', 3, 'number of distinct class')
flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')

flags.DEFINE_integer('max_aspect_len', 0, 'max length of aspects')
flags.DEFINE_integer('max_context_len', 0, 'max length of contexts')
flags.DEFINE_string('embedding_matrix', '', 'word ids to word vectors')

batch_size = 128
learning_rate = 0.01
n_epoch = 20
pre_processed = 1
embedding_file_name = "/content/drive/Shareddrives/Social Media False Info Propagation_V2/Arpita Sahu/IAN/data/glove.6B.300d.txt"
dataset = "/content/drive/Shareddrives/Social Media False Info Propagation_V2/Arpita Sahu/IAN/data/laptop/"
#setting dataset and log directory

class IAN_Data(Dataset):
    def __init__(self, dataset):
        self.aspects = torch.from_numpy(np.array(dataset[0]))
        self.contexts = torch.from_numpy(np.array(dataset[1]))
        self.labels = torch.from_numpy(np.array(dataset[2]))
        self.aspect_lens = torch.from_numpy(np.array(dataset[3]))
        self.context_lens = torch.from_numpy(np.array(dataset[4]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.aspects[idx], self.contexts[idx], self.labels[idx], self.aspect_lens[idx], self.context_lens[idx])




def run(model, train_data, test_data):
  print('Training ...')
  max_acc, max_f1, step = 0., 0., -1

  train_data_size = len(train_data[0])
  train_dataset = IAN_Data(train_data)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last = True)

  test_data_size = len(test_data[0])
  test_dataset = IAN_Data(test_data)
  test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, drop_last = True)
    
  #optimizer = torch.optim.Adam(model.config(), lr=learning_rate)
    
  for i in range(n_epoch):
    cost, predict_list, labels_list = 0., [], []
    for _, data in enumerate(train_loader):
      aspects, contexts, labels, aspect_lens, context_lens = data[0], data[1], data[2], data[3], data[4]
      #print("DATA IN MAIN = ", data, len(data))
      predict, labels = model.call(data, dropout = 0.5)
      loss_t = nn.CrossEntropyLoss(predict, labels)
      loss = F.mean(loss_t)
        
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      #torch.autograd()
 
      predict_list.extend(F.argmax(F.softmax(predict)).numpy())
      labels_list.extend(F.argmax(labels).numpy())
        
        
    train_acc, train_f1, _, _ = evaluate(pred=predict_list, gold=labels_list)
    #write to summary
      
    cost, predict_list, labels_list = 0., [], []
    for _, data in enumerate(test_loader):
      aspects, contexts, labels, aspect_lens, context_lens = data[0], data[1], data[2], data[3], data[4]
      
      predict, labels = model(data, dropout=1.0)
      loss_t = torch.nn.CrossEntropyLoss(predict, labels)

      predict_list.extend(F.argmax(F.softmax(predict)).numpy())
      labels_list.extend(F.argmax(labels).numpy())
    
    test_acc, test_f1, _, _ = evaluate(pred=predict_list, gold=labels_list)
    test_loss = cost/test_data_size
    #write to summary
      
    if test_acc + test_f1 > max_acc + max_f1:
      max_acc = test_acc
      max_f1 = test_f1
      step = i
      #write to saver
            
    print('epoch %s: train-loss=%.6f; train-acc=%.6f; train-f1=%.6f; test-loss=%.6f; test-acc=%.6f; test-f1=%.6f.' % (str(i), train_loss, train_acc, train_f1, test_loss, test_acc, test_f1))  
      #print to console
  print('The max accuracy of testing results: acc %.6f and macro-f1 %.6f of step %s' % (max_acc, max_f1, step))
  #print to console
  
def main(_):
  #measure time
  #loading data information
  word2id, FLAGS.max_aspect_len, FLAGS.max_context_len = get_data_info(dataset, pre_processed)
  
  #loading training and test data
  train_data = read_data(word2id, FLAGS.max_aspect_len, FLAGS.max_context_len, dataset + 'train', pre_processed)
  test_data = read_data(word2id, FLAGS.max_aspect_len, FLAGS.max_context_len, dataset + 'test', pre_processed)
  
  #loading pre-trained word vectors
  FLAGS.embedding_matrix = load_word_embeddings(embedding_file_name, FLAGS.embedding_dim, word2id)
  
  model = IAN(FLAGS)
  run(model, train_data, test_data)
  
  #measure time
  #print time cost
  
if __name__ == '__main__':
  app.run(main)
