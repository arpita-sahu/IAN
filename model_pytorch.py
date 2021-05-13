import torch
import numpy as np
from torch.autograd import Variable
from LSTMclass import lstm_model

class IAN(torch.nn.Module):

    def __init__(self, config): #config = FLAGS
        super(IAN, self).__init__()

        self.embedding_dim = config.embedding_dim
        self.n_hidden = config.n_hidden #number of hidden layers
        self.n_class = config.n_class #number of classes
        self.l2_reg = config.l2_reg #l2 regularization 

        self.max_aspect_len = config.max_aspect_len 
        self.max_context_len = config.max_context_len
        self.embedding_matrix = config.embedding_matrix

        self.aspect_lstm = lstm_model(self.embedding_dim, self.n_hidden, False)   
        self.context_lstm = lstm_model(self.embedding_dim, self.n_hidden, True)

        self.aspect_w = Variable(torch.randn([self.n_hidden, self.n_hidden]), name='aspect_w')
        self.aspect_b = Variable(torch.zeros([self.n_hidden]), name='aspect_b')
        self.context_w = Variable(torch.randn([self.n_hidden, self.n_hidden]), name='context_w')
        self.context_b = Variable(torch.zeros([self.n_hidden]), name='context_b')

        self.output_fc = torch.nn.Linear(self.n_class, self.n_class) # in_features = out_features = n_class. Throwing an error when including just 1 parameter
        self.optimizer = torch.optim.Adam(self.output_fc.parameters(), lr=self.l2_reg) #L2 regularization

        #self.output_fc = tf.keras.layers.Dense(self.n_class, kernel_regularizer=tf.keras.regularizers.l2(l=self.l2_reg)) #kernel_regularizer: regularizer to apply a penality on the layer's kernel

    def call(self, data, dropout=0.5): #done
    
        aspects, contexts, labels, aspect_lens, context_lens = data

        aspect_inputs = torch.index_select(input = torch.tensor(self.embedding_matrix), dim = 0, index = aspects.long().flatten()) #return values of elements in embedding_matrix at indices given by aspects
        aspect_inputs = aspect_inputs.type(torch.FloatTensor) #converting tensor to float32 type 
        rate = 1 - dropout #dropout = keepprob
        func = torch.nn.Dropout(p=rate)
        aspect_inputs = func(aspect_inputs) #change some elements to 0 to reduce overfitting 


        context_inputs = torch.index_select(input = torch.tensor(self.embedding_matrix), dim = 0, index = contexts.long().flatten())
        context_inputs = context_inputs.type(torch.FloatTensor)
        context_inputs = func(context_inputs)
 
        aspect_outputs = self.aspect_lstm.forward(aspect_inputs)
        aspect_avg = torch.mean(aspect_outputs, 1)
        
        context_outputs = self.context_lstm.forward(context_inputs)
        context_avg = torch.mean(context_outputs, 1)
        
        print("\n\n", aspect_outputs.shape, self.aspect_w.shape, torch.unsqueeze(context_avg, -1))
        aspect_att = torch.nn.Softmax(torch.tanh(torch.einsum('ijk,kl,ilm->ijm', aspect_outputs, self.aspect_w,  torch.unsqueeze(context_avg, -1)) + self.aspect_b), axis=1)

        aspect_rep = torch.sum(aspect_att * aspect_outputs, 1)
        context_att = torch.nn.Softmax(torch.tanh(torch.einsum('ijk,kl,ilm->ijm', context_outputs, self.context_w, torch.unsqueeze(aspect_avg, -1)) + self.context_b), axis=1)
        
        context_rep = torch.sum(context_att * context_outputs, 1) #find sum along dim 1 

        rep = torch.cat([aspect_rep, context_rep], 1) #concat along dimension 1
        predict = self.output_fc(rep)

        return predict, labels
