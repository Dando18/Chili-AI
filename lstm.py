'''
author: Daniel Nichols
date: September 2020
'''
# std imports
import time

# 3rd party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# local imports
from utilities import vprint


class LSTMNetwork(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, dropout, padding_idx):
        super(LSTMNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        if num_layers == 1:
            dropout = 0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

        self.output = nn.Linear(hidden_dim, vocab_size)


    def forward(self, x):
        prev_state = self.zero_state(x.shape[0])

        embeddings = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embeddings, prev_state)
        logits = self.output(hn[-1])
        return logits

    def zero_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))


def train(dataset, model, optimizer, loss_func, epochs, verbose=False):
    ''' Trains 'model'.
    '''
    model.train()
    train_begin = time.time()
    for epoch in range(epochs):
        cum_loss = 0.0
        total_correct = 0
        batch_iter = 0
        epoch_begin = time.time()

        for batch, (x,y) in enumerate(dataset):
            model.zero_grad()

            logits = model(x)

            loss = loss_func(logits, y-1)
            cum_loss += loss.item()
            batch_iter += 1

            preds = torch.argmax(logits, dim=1)
            num_correct = torch.sum(preds == (y-1))
            total_correct += num_correct

            loss.backward()
            optimizer.step()
        
        epoch_end = time.time()

        vprint(verbose, 'Epoch {}:  Loss {:.3f}  Accuracy {:.3f}%  Duration {:.3f} sec'.format(
            epoch, 
            cum_loss/batch_iter,
            total_correct / (dataset.batch_size*batch_iter) * 100.0,
            epoch_end - epoch_begin
        ))

    train_end = time.time()
    vprint(verbose, 'Training finished. Took {:.5f} minute(s).'.format(
        (train_end - train_begin) / 60.0
    ))


def generate(model, vocab, num_sentences=20, num_recipes=1, verbose=False):
    ''' Generates sentences using the lstm model.
    '''
    SENTENCE_LENGTH = 20

    model.eval()
    with torch.no_grad():
        for _ in range(num_recipes):
            for _ in range(num_sentences):
                # initialize sequence
                seq = torch.randint(low=5, high=len(vocab), size=(1,))-1
                
                for _ in range(SENTENCE_LENGTH):
                    logits = model(seq.view(1, seq.shape[0]))
                    predictions = F.softmax(logits, dim=1).squeeze().detach()
                    predicted_idx = torch.argmax(predictions)
                    seq = torch.cat((seq, predicted_idx.view(1)))

                seq += 1
                words = vocab.lookup_tokens(list(seq.numpy()))
                print(' '.join(words))
            
            print()
        


def do_lstm_training(dataset, vocab, embedding_dim=32, hidden_dim=32,
    num_layers=2, dropout=0.2, loss='cross_entropy', optim='adam', lr=0.1, 
    epochs=5, load_model=None, save_model=None, num_sentences=20, num_recipes=1,
    verbose=False):
    '''
    '''
    assert loss in ['nll', 'cross_entropy']
    assert optim in ['sgd', 'adam']

    # create or load model
    padding_idx = vocab['<pad>']    # TODO -- refactor
    model = None
    if load_model is None:
        model = LSTMNetwork(embedding_dim, hidden_dim, len(vocab), num_layers, dropout, padding_idx)
    else:
        with open(load_model, 'rb') as fp:
            model = torch.load(fp)
            vprint(verbose, 'Loaded saved model {}.'.format(load_model))
    vprint(verbose, model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # prepare loss function
    loss_func = None
    if loss == 'nll':
        loss_func = nn.NLLLoss()
    else:
        loss_func = nn.CrossEntropyLoss()

    # prepare optimizer
    optimizer = None
    if optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # use training helper to train
    train(dataset, model, optimizer, loss_func, epochs, verbose=verbose)

    # generate
    generate(model, vocab, num_sentences=num_sentences, num_recipes=num_recipes,
        verbose=verbose)

    # export
    if save_model is not None:
        with open(save_model, 'wb') as fp:
            torch.save(model, fp)
            vprint(verbose, 'Exported model to {}.'.format(save_model))



    