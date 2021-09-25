'''
author: Daniel Nichols
date: September 2021
'''
# std imports
from os.path import join as path_join
import glob
from json import load as json_load
from collections import Counter
from functools import partial

# 3rd party imports
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator

# local imports
from utilities import vprint


BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'


def get_raw_data(root, verbose=False):
    ''' Reads the raw data set from root. Lazily prefers a collection of json
        files in `root`.
        Args:
            root: where to look for raw data
            verbose: verbosity
    '''
    all_data = []
    
    # read in all json files in the file path
    for json_file_path in glob.glob(path_join(root, '*.json')):
        with open(json_file_path, 'r') as fp: 
            json_data = json_load(fp)
        all_data.append(json_data)

    return all_data


def get_joined_data(raw_data, method='concat', use_stops=True, output_column='data', verbose=False):
    ''' Aggregate data in each point, so that each data object has a 
        single data value represented by a sequence of text or just text.
        Args:
            raw_data: a list of dicts like {..., 'ingredients': ..., 'steps': ...}
                      where ingredients and steps are lists of strings
            method: 'concat' or 'steps' how to join data. Concat will join into  
                    a single string of text. Steps will preserve the steps and
                    output a list of strings for each data point
            use_stops: when joining include stops. If true, returns the stops.
            output_column: what column to store the data in.
    '''
    assert method in ['concat', 'steps'], 'Invalid join method.'

    joined_data = raw_data
    stop = ''
    if method == 'concat':
        stop = '<END_INGR>\n' if use_stops else ''
        for data_point in raw_data:
            ingredients_str = '\n'.join(data_point['ingredients'])
            steps_str = '\n'.join(data_point['steps'])

            data_point[output_column] = '{}\n{}{}'.format(ingredients_str, stop, steps_str)
    else:
        raise 'Join method not implemented.'

    if use_stops:
        return joined_data, stop
    else:
        return joined_data


def get_vocab(dataset, tokenizer, stops=[], data_column='data', verbose=False):
    ''' Returns a torchtext Vocab object for the dataset.
    '''

    def generate_tokens(dataset_, column_):
        for data_point_ in dataset_:
            if type(data_point_[column_]) == list:
                raise TypeError('Building vocab for list not yet supported.')
            elif type(data_point_[column_]) == str:
                yield tokenizer(data_point_[column_])
            else:
                raise TypeError('Invalid type in data column. Expected str or list of str.') 

    
    stops.extend(['<unk>', '<pad>', '<bos>', '<eos>'])
    return build_vocab_from_iterator(generate_tokens(dataset, data_column), specials=stops)


def get_tokenized_data(dataset, stops=[], data_column='data', tokenizer='word', verbose=False):
    ''' Tokenizes the dataset.
    '''
    assert tokenizer in ['word', 'letter', 'step', 'none'], 'Invalid tokenization method.'

    if tokenizer == 'word':
        tokenizer = 'basic_english'
    elif tokenizer == 'letter':
        tokenizer = lambda s: [c for c in s]
    else:
        raise ValueError('{} tokenizer not implemented.'.format(tokenizer))

    tok = get_tokenizer(tokenizer, language='en')
    vocab = get_vocab(dataset, tok, stops, data_column=data_column, verbose=verbose)

    data = []
    for data_point in dataset:
        sample = data_point[data_column]

        if type(sample) == list:
            tokenized_steps = []
            for step in sample:
                sample_tensor = torch.tensor([vocab[token] for token in tok(sample)])
                tokenized_steps.append(sample_tensor)
            data.append(tokenized_steps)

        elif type(sample) == str:
            sample_tensor = torch.tensor([vocab[token] for token in tok(sample)])
            data.append(sample_tensor)

        else:
            raise TypeError('Expected data column to be str.')

    return data, vocab


def get_sequential_data(tokenized_data, sequence_length, verbose=False):
    ''' Turns the data into sequences via a moving window.
    '''
    x = list()
    y = list()
    for sequence in tokenized_data:
        # sequence is a tensor sequence
        for index in range(len(sequence)-sequence_length):
            x.append( sequence[index:index+sequence_length] )
            y.append( sequence[index+sequence_length] )    

    x = torch.vstack(x)
    y = torch.tensor(y)

    return x, y

def format_data_batch(batch, bos_idx, eos_idx, pad_idx):
    ''' Collates individual batches as they are loaded.
    '''
    output = []
    for b in batch:
        output.append(
            torch.cat([torch.tensor([bos_idx]), b, torch.tensor([eos_idx])],
                dim=0)
        )

    output = pad_sequence(output, padding_value=pad_idx)
    return output


def get_dataset(root, tokenize='word', batch_size=32, sequence_length=None, verbose=False):
    ''' Returns the dataset located at root.
        Args:
            root: root directory of raw data.
            tokenize: [word, letter, step, none] -- how to tokenize data
            verbose: verbosity
    '''
    assert tokenize in ['word', 'letter', 'step', 'none'], 'Invalid tokenization method.'

    # read raw data
    vprint(verbose, 'Reading raw input data.')
    raw_data = get_raw_data(root, verbose=verbose)

    # join data
    vprint(verbose, 'Joining data.')
    joined_data, stop = get_joined_data(raw_data, method='concat', verbose=verbose)

    # early exit with text data if not tokenizing
    if tokenize == 'none':
        return joined_data

    # tokenize data
    vprint(verbose, 'Tokenizing data.')
    tokenized_data, vocab = get_tokenized_data(joined_data, tokenizer=tokenize, 
                                stops=[stop, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN], 
                                verbose=verbose)

    # make sequential
    if sequence_length is not None:
        vprint(verbose, 'Making sequential data.')
        tokenized_data = get_sequential_data(tokenized_data, sequence_length,
            verbose=verbose)

        tokenized_data = TensorDataset(*tokenized_data)
    
    # TODO -- train/test/validate split
    # return DataLoaders
    bos_idx = vocab[BOS_TOKEN]
    eos_idx = vocab[EOS_TOKEN]
    pad_idx = vocab[PAD_TOKEN]
    batch_formatter = partial(format_data_batch, bos_idx=bos_idx, eos_idx=eos_idx, pad_idx=pad_idx)
    train_dl = DataLoader(tokenized_data, batch_size=batch_size, shuffle=True)
    
    return train_dl, vocab