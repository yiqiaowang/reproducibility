# -*- coding: utf-8 -*-
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import numpy as np

# Download the required data



import requests

def downloadFile(url):
    return requests.get(url).content
      
def loadTurkishValid():
    sample_data_id = "https://storage.googleapis.com/vincep/turkish_json/sample2.json"
    valid_sample_data = [json.loads(line) for line in downloadFile(sample_data_id).decode('utf-8').strip().splitlines()]
    return valid_sample_data

def loadTurkishVocab():
    id = "https://storage.googleapis.com/vincep/turkish_json/document.vocab"
   
    vocab_data = downloadFile(id).decode('utf-8').strip().splitlines()
    
    vocab = []
    for elem in vocab_data[2:]:
        row = elem.split('\t')
        vocab.append({'index': int(row[0]), 'word': row[1].strip(), 'count': int(row[2])})
    return vocab

# Prepare the data for the word model

def prepareData(data_list, seperator=1000000, max_document_length=60):
    return [data["document_sequence"][:max_document_length] + [seperator] + data["question_sequence"] for data in data_list], [data["answer_sequence"] for data in data_list]

# Prepare the data for the byte model

def processEntry(data, byte_limit=float("inf"), word_seperator=ord(' ')):
    entry = []
    for word in data:
        entry += list(word.encode('utf-8'))
        entry.append(word_seperator)

        if len(entry) >= byte_limit:
            break
      
    return entry



def processByByte(data_list, word_seperator=256, document_query_seperator=0, max_document_byte_length=400, max_question_byte_length=50):
    processed_data = []
    processed_answers = []
    maxCount = 3
        for data in data_list:
            document_entry = processEntry(data["string_sequence"], max_document_byte_length)
            question_entry = processEntry(data["question_string_sequence"], max_question_byte_length)

            processed_data.append(document_entry + question_entry)

            answer_entry = processEntry(data["raw_answers"][:1])
            processed_answers.append(answer_entry)
    
    return processed_data, processed_answers

# Load the vocabulary and the data

vocab = loadTurkishVocab()
valid_data = loadTurkishValid()

# The Encoder

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, cell_type,
                 bidirectional=False, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_size,embed_size)
        
        if cell_type == 'lstm':
            self.rnn = nn.LSTM(
                embed_size, hidden_size, num_layers, bidirectional=self.bidirectional
            )
        elif cell_type == 'gru':
            self.rnn = nn.GRU(
                embed_size, hidden_size, num_layers, bidirectional=self.bidirectional
            )
        else:
            raise ValueError('RNN cell type not valid')
            
            
    def forward(self, inputs, lengths, hidden=None):
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, lengths)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(outputs)  
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        return outputs, hidden


# The Attention Layer used in the decoder

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.softmax = nn.Softmax(dim=1)
        
      

    def forward(self, hidden, encoder_outputs):

        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1)
        attn_energies = self.score(H,encoder_outputs) 
        return self.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(2,1)
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1)
        energy = torch.bmm(v,energy)
        return energy.squeeze(1)

# The Decoder class with the Attention layer

class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, cell_type, n_layers=1):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, embed_size)
        self.attn = Attn(hidden_size)

        if cell_type == 'lstm':
            self.rnn = nn.LSTM(
                embed_size + hidden_size, hidden_size, num_layers=n_layers
            )
        elif cell_type == 'gru':
            self.rnn = nn.GRU(
                embed_size + hidden_size, hidden_size, num_layers=n_layers
            )
        else:
            raise ValueError('RNN cell type not valid')
        
        self.unembedding = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, word_input, last_hidden, encoder_outputs):
        batch_size = word_input.size(0)
        
        word_embedded = self.embedding(word_input).view(1, batch_size, -1)
        
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)
        
        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(0)
        output = self.unembedding(output)
        output = self.log_softmax(output)
        return output, hidden

# Used to create the models. Different parameters passed to this class, and to its init_encoder and init_decoder methods,
#     can be used to create all of the models used in this experiment

class end2endRNN():
    def __init__(self, batch_size, lr, max_pred_len=50):
        super(end2endRNN, self).__init__()
       
        self.batch_size = batch_size
        self.loss = nn.CrossEntropyLoss()
        self.sos, self.eos, self.pad = 0, 1, 3
        self.MAX_PRED_LEN = max_pred_len
        self.lr = lr


    def init_encoder(self, input_size, embed_size, hidden_size, **kwargs):
        self.encoder = EncoderRNN(input_size, embed_size, hidden_size, **kwargs)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)

    def init_decoder(self, output_size, embed_size, hidden_size, **kwargs):
        self.decoder = AttnDecoderRNN(input_size, embed_size, hidden_size, **kwargs)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)
        
    def pad_seq(self, seq, max_length):
        # 3 is the number corresponding to the token <NONE>
        seq += [self.pad for i in range(max_length - len(seq))]
        return seq

    def prepareBatch(self, data, wrap_token=False):
        
        _data = [[self.sos] + t + [self.eos] for t in data] if wrap_token else data
        
        data_seqs = sorted(_data, key=lambda p: len(p), reverse=True)
        
        data_lengths = [len(s) for s in data_seqs]
        data_padded = [self.pad_seq(s, max(data_lengths)) for s in data_seqs]
        
        data_var = Variable(torch.LongTensor(data_padded)).transpose(0, 1)

        return data_var, data_lengths
       
    def step(self, inputs, targets, hidden_layers):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
       
      
        # Prepare data for batching
        # SOS and EOS tokens are added in `prepareBatch` to the targets
        input_batch, input_batch_len = self.prepareBatch(inputs)
        target_batch, target_batch_len = self.prepareBatch(targets, wrap_token=True)
        
        max_target_length = max(target_batch_len)
        
        # Prepare decoder's hidden state
        encoder_out, hidden_state = self.encoder.forward(input_batch, input_batch_len, hidden_layers)
      
        # Actual training occurs here (Decoder)
        total_loss = 0
        
        for i in range(max_target_length - 1):
            out, hidden_state = self.decoder.forward(target_batch[i], hidden_state, encoder_out)
            current_loss = self.loss(out, target_batch[i+1])
            total_loss += current_loss

        total_loss.backward()

        self.decoder_optimizer.step()
        self.encoder_optimizer.step()

        return total_loss.data[0], hidden_layers
      
      
    def train(self, inputs, targets, batch_size, num_steps):
        losses = []
        hidden_layers = None
        for step in range(num_steps):
            loss_per_step = []
            for data_index in range(0, len(inputs)-batch_size+1, batch_size):
                batch_inputs = inputs[data_index: data_index+batch_size]
                batch_targets = targets[data_index: data_index+batch_size]
                loss, hidden_layers = self.step(batch_inputs, batch_targets, hidden_layers)
                print("loss of {} at step {} and minibatch {}".format(loss, step+1, data_index//batch_size + 1))
                loss_per_step.append(loss)
          
            losses.append(loss_per_step)
          
          
        return losses
      
     
    def crossentropy_loss_batch(self, inputs, targets, hidden_layers):
        # Prepare data for batching
        # SOS and EOS tokens are added in prepareBatch to the targets
        input_batch, input_batch_len = self.prepareBatch(inputs)
        target_batch, target_batch_len = self.prepareBatch(targets, wrap_token=True)

        max_target_length = max(target_batch_len)

        # Prepare decoder's hidden state
        encoder_out, hidden_state = self.encoder.forward(input_batch, input_batch_len, hidden_layers)

        # Actual training occurs here (Decoder)
        total_loss = 0
       
        for i in range(max_target_length - 1):
            out, hidden_state = self.decoder.forward(target_batch[i], hidden_state, encoder_out)
            current_loss = self.loss(out, target_batch[i+1])
            total_loss += current_loss

        return total_loss.data[0]
   
   
    def crossentropy_loss(self, inputs, targets, batch_size, hidden_layers=None):
        losses = []

        for data_index in range(0, len(inputs)-batch_size+1, batch_size):
            batch_inputs = inputs[data_index: data_index+batch_size]
            batch_targets = targets[data_index: data_index+batch_size]
            loss = end2endRNN.crossentropy_loss_batch(self,batch_inputs, batch_targets, hidden_layers)
            losses.append(loss)
           
        return losses
            
        
        
        

    def eval(self, inputs):
        # Prepare data for batching
        input_batch, input_batch_len = self.prepareBatch(inputs)
        
        # Prepare vectors for decoder
        encoder_out, hidden_state = self.encoder(input_batch, input_batch_len)
        
        sentence = []
        
        x_t = Variable(torch.LongTensor([self.sos] * self.batch_size))
        eos = Variable(torch.LongTensor([self.eos] * self.batch_size))
        
        
        # Stop the infinite loop...
        counter = 0
        
        # Predict
        while counter < self.MAX_PRED_LEN and not x_t.equal(eos):
            counter = counter + 1
            output, hidden_state = self.decoder.forward(x_t, hidden_state, encoder_out)
            val, ind = torch.max(output.data, 1)
            x_t = Variable(ind)
            sentence.append(ind)

        return sentence

# A function that takes all of the model hyper parameters, creates the model and trains it

def train(lr, embedding_size, hidden_size, num_layers, cell_type, train_data, train_answer, batch_size, num_epochs, input_size, output_size,bidirectional):
    rnn = end2endRNN(batch_size, lr)
    rnn.init_encoder(input_size, embedding_size, hidden_size, cell_type=cell_type,bidirectional=bidirectional,num_layers=num_layers)
    rnn.init_decoder(output_size, embedding_size, hidden_size, cell_type=cell_type)

    start = time()
    losses = rnn.train(train_data, train_answer, batch_size, num_epochs)
    end = time()
    return losses, (end-start), rnn
