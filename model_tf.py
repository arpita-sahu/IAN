import tensorflow as tf


class IAN(tf.keras.Model):

    def __init__(self, config):
        super(IAN, self).__init__()
        print("INSIDE INIT")
        self.embedding_dim = config.embedding_dim
        self.n_hidden = config.n_hidden
        self.n_class = config.n_class
        self.l2_reg = config.l2_reg

        self.max_aspect_len = config.max_aspect_len
        self.max_context_len = config.max_context_len
        self.embedding_matrix = config.embedding_matrix

        self.aspect_lstm = tf.keras.layers.LSTM(self.n_hidden,
                                                return_sequences=True,
                                                recurrent_initializer='glorot_uniform',
                                                stateful=True)
        self.context_lstm = tf.keras.layers.LSTM(self.n_hidden,
                                                 return_sequences=True,
                                                 recurrent_activation='sigmoid',
                                                 recurrent_initializer='glorot_uniform',
                                                 stateful=True)

        self.aspect_w = tf.contrib.eager.Variable(tf.random_normal([self.n_hidden, self.n_hidden]), name='aspect_w')
        self.aspect_b = tf.contrib.eager.Variable(tf.zeros([self.n_hidden]), name='aspect_b')
        self.context_w = tf.contrib.eager.Variable(tf.random_normal([self.n_hidden, self.n_hidden]), name='context_w')
        self.context_b = tf.contrib.eager.Variable(tf.zeros([self.n_hidden]), name='context_b')
        self.output_fc = tf.keras.layers.Dense(self.n_class, kernel_regularizer=tf.keras.regularizers.l2(l=self.l2_reg))

    def call(self, data, dropout=0.5):
        print("INSIDE CALL")
        aspects, contexts, labels, aspect_lens, context_lens = data
        #print(aspects, contexts, aspect_lens, context_lens)
        aspect_inputs = tf.nn.embedding_lookup(self.embedding_matrix, aspects)
        aspect_inputs = tf.cast(aspect_inputs, tf.float32)
        aspect_inputs = tf.nn.dropout(aspect_inputs, keep_prob=dropout)
   
        context_inputs = tf.nn.embedding_lookup(self.embedding_matrix, contexts)
        context_inputs = tf.cast(context_inputs, tf.float32)
        context_inputs = tf.nn.dropout(context_inputs, keep_prob=dropout)
        #print(context_inputs, context_inputs.shape)
        
        print("SHAPE = ", self.embedding_dim, self.n_hidden, self.n_class)
        aspect_outputs = self.aspect_lstm(aspect_inputs)
        #print("ASPECT OUTPUT = ", aspect_outputs, aspect_outputs.shape)
        aspect_avg = tf.reduce_mean(aspect_outputs, 1)

        context_outputs = self.context_lstm(context_inputs)
        context_avg = tf.reduce_mean(context_outputs, 1)

        print(aspect_outputs.shape, self.aspect_w.shape, tf.expand_dims(context_avg,-1).shape, self.aspect_b.shape)
        aspect_att = tf.nn.softmax(tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', aspect_outputs, self.aspect_w,
                                                        tf.expand_dims(context_avg, -1)) + self.aspect_b),
                                   axis=1)
        aspect_rep = tf.reduce_sum(aspect_att * aspect_outputs, 1)
        #print("Aspect = ", aspect_rep, aspect_rep.shape)
        context_att = tf.nn.softmax(tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', context_outputs, self.context_w,
                                                         tf.expand_dims(aspect_avg, -1)) + self.context_b),
                                    axis=1)
        context_rep = tf.reduce_sum(context_att * context_outputs, 1)
        #print("Aspect = ", context_rep, context_rep.shape)
        
        rep = tf.concat([aspect_rep, context_rep], 1)
        #print("test = ", self.max_aspect_len, self.embedding_dim, self.aspect_w.shape, self.aspect_b.shape, self.context_w.shape, self.context_b.shape)
        #print("\n\nREP = ", rep, "\n\nREP SHAPE = ", rep.shape)
        predict = self.output_fc(rep)
        #print("\n\nPREDICT = ",predict, "\n\nPREDICT SHAPE = ",predict.shape) 

        return predict, labels
