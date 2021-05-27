import torch 
from torch.autograd import Variable

class lstm_model(torch.nn.Module):

  def __init__(self, n_hidden, embedding_dim, sig): #hidden_size = number of hidden layers  
    
    super().__init__()
    self.num_layers = 1
    self.sig = sig
    self.hidden = n_hidden
    self.lstm = torch.nn.LSTM(input_size = embedding_dim, hidden_size = n_hidden, batch_first = True)
    self.sigmoid = torch.nn.Sigmoid()
    self.lstm_weight = torch.nn.init.xavier_uniform_(self.lstm.all_weights[0][0]) #weight of the lstm needs to be passed as a parameter to the xavier uniform function

  def forward(self, x):
    
    
#     print(x, x.shape, x.dim())
#     print(x.shape[0], x.shape[1])
    #x = x.view(1, x.shape[0], x.shape[1]) 
    #number of layers? number of layers = number of stacked lstm layers 
    #Input must be 3 dimensional (Sequence len, batch, input dimensions)
    #hidden size = number of hidden units
    #input size = number of features

    #x = the input tensor 

    #LSTMCell = same as LSTM except the number of layers is always 1 
    print("X = ", x, x.shape)
    h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden)) #hidden state
    print("h_0 = ", h_0, self.num_layers, self.hidden)
    c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden))
    print("c_0 = ", c_0) 
    output, (h,c) = self.lstm(x, (h_0, c_0))
    print("output, h, c = ", output,output.shape, h, c)
    #out = h.view(-1, self.hidden) #reshaping data 
    print("out = ", out, out.shape)

    #aspect lstm doesn't use the sigmoid function
    if self.sig == True:
      out = self.sigmoid(h)

    out = torch.nn.init.xavier_uniform_(out)#self.lstm_weight(out)
    
    return out 
