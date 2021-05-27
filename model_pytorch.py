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

        self.output_fc = torch.nn.Linear(600, self.n_class) # in_features = out_features = n_class. Throwing an error when including just 1 parameter
        self.optimizer = torch.optim.Adam(self.output_fc.parameters(), lr=self.l2_reg) #L2 regularization

        #self.output_fc = tf.keras.layers.Dense(self.n_class, kernel_regularizer=tf.keras.regularizers.l2(l=self.l2_reg)) #kernel_regularizer: regularizer to apply a penality on the layer's kernel

    def call(self, data, dropout=0.5): #done
    
        aspects, contexts, labels, aspect_lens, context_lens = data
        #print(aspects, contexts, labels, aspect_lens, context_lens)
        #print("Shapes = ", aspects.shape, contexts.shape, labels.shape, aspect_lens.shape, context_lens.shape)

        aspect_inputs = torch.index_select(input = torch.tensor(self.embedding_matrix), dim = 0, index = aspects.long().flatten()) #return values of elements in embedding_matrix at indices given by aspects
        aspect_inputs = torch.reshape(aspect_inputs, (aspects.shape[0], aspects.shape[1], aspect_inputs.shape[1]))
        aspect_inputs = aspect_inputs.type(torch.FloatTensor) #converting tensor to float32 type 
        rate = 1 - dropout #dropout = keepprob
        func = torch.nn.Dropout(p=rate)
        aspect_inputs = func(aspect_inputs) #change some elements to 0 to reduce overfitting 
        #print("INPUTS = \n", aspect_inputs.shape)

        context_inputs = torch.index_select(input = torch.tensor(self.embedding_matrix), dim = 0, index = contexts.long().flatten())
        context_inputs = torch.reshape(context_inputs, (contexts.shape[0], contexts.shape[1], context_inputs.shape[1]))
        context_inputs = context_inputs.type(torch.FloatTensor)
        context_inputs = func(context_inputs)
        #print(context_inputs, context_inputs.shape)
 
        aspect_outputs = self.aspect_lstm.forward(aspect_inputs)
        #print("SHAPES = ", self.embedding_dim, self.n_hidden, self.n_class)
        #print("ASPECT OUTPUT = ", aspect_outputs, aspect_outputs.shape)
        #aspect_outputs = torch.reshape(aspect_outputs, (1, aspect_outputs.shape[0], aspect_outputs.shape[1]))
        aspect_avg = torch.mean(aspect_outputs, 1)
        
        context_outputs = self.context_lstm.forward(context_inputs)
        context_avg = torch.mean(context_outputs, 1)

        #aspect_outputs = torch.reshape(aspect_outputs, (aspect_outputs.shape[0],aspect_outputs.shape[1], 1))
        # aspect_att = tf.nn.softmax(tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', aspect_outputs, self.aspect_w, tf.expand_dims(context_avg, -1)) + self.aspect_b),
        print(aspect_outputs.shape, self.aspect_w.shape, torch.unsqueeze(context_avg, -1).shape, self.aspect_b.shape)
        aspect_att = torch.nn.functional.softmax(torch.tanh(torch.einsum('ijk,kl,ilm->ijm', aspect_outputs, self.aspect_w,  torch.unsqueeze(context_avg, -1)) + self.aspect_b), dim=1)
        #print("ASPECT ATT = ", aspect_att, aspect_att.shape)
        
        #print(type(aspect_att), type(aspect_outputs))
        aspect_rep = torch.sum(aspect_att * aspect_outputs, 1)
        print("Aspect rep = ", aspect_rep.shape)
        #print("context = ", context_outputs.shape, self.context_w.shape, torch.unsqueeze(aspect_avg, -1).shape, self.context_b.shape)
        context_att = torch.nn.functional.softmax(torch.tanh(torch.einsum('ijk,kl,ilm->ijm', context_outputs, self.context_w, torch.unsqueeze(aspect_avg, -1)) + self.context_b), dim=1)
        #print("CONTEXT ATT = ", context_att, context_att.shape)
        
        context_rep = torch.sum(torch.tensor(context_att) * context_outputs, 1) #find sum along dim 1 
        print("context rep = ", context_rep.shape)

        rep = torch.cat([aspect_rep, context_rep], 1) #concat along dimension 1
        print("rep = ", rep.shape)
        #print("test = ", self.max_aspect_len, self.embedding_dim, self.aspect_w.shape, self.aspect_b.shape, self.context_w.shape, self.context_b.shape)
        #print("\n\n\nREP = ", rep, rep.shape)
        #print(self.n_class)
        self.output_fc = torch.nn.Linear(rep.shape[1], self.n_class)
        predict = self.output_fc(rep)
        
        print("prediction = ", predict, labels, predict.shape)
        return predict, labels
