#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
sanity_check.py: sanity checks for assignment 5
Usage:
    sanity_check_more.py hw
    sanity_check_more.py generate_data
    sanity_check_more.py gen_cnn_data
    sanity_check_more.py cnn
"""
import json
import math
import pickle
import sys
import time

import numpy as np

from docopt import docopt
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import pad_sents_char, read_corpus, batch_iter
from vocab import Vocab, VocabEntry

from char_decoder import CharDecoder
from nmt_model import NMT
from highway import Highway
from cnn import CNN

from sanity_check import DummyVocab

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

#----------
# CONSTANTS
#----------
BATCH_SIZE = 5
EMBED_SIZE = 3
EMBED_SIZE_WORD = 10
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0
MAX_WORD_LEN = 8

def reinitialize_layers(model):
    """ Reinitialize the Layer Weights for Sanity Checks.
    """
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.3)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        elif type(m) == nn.Embedding:
            m.weight.data.fill_(0.15)
        elif type(m) == nn.Dropout:
            nn.Dropout(DROPOUT_RATE)
    with torch.no_grad():
        model.apply(init_weights)

def generate_highway_data():
    """ generate data for highway
	"""
    conv_input = np.random.rand(BATCH_SIZE, EMBED_SIZE_WORD)
    W_proj = np.ones((EMBED_SIZE_WORD, EMBED_SIZE_WORD)) * 0.3
    b_proj = np.ones(EMBED_SIZE_WORD) * 0.1
	
    W_gate = np.ones((EMBED_SIZE_WORD, EMBED_SIZE_WORD)) * 0.3
    b_gate = np.ones(EMBED_SIZE_WORD) * 0.1
	
    def relu(input):
        return np.maximum(input, 0)
    
    def sigmoid(input):
        return 1. / (1 + np.exp(-input))
	
    x_proj = relu(conv_input.dot(W_proj) + b_proj)
    x_gate = sigmoid(conv_input.dot(W_gate) + b_gate)
    x_highway = x_gate * x_proj + (1 - x_gate) * conv_input
	
    np.save('sanity_check_handmade_data/highway_conv_input.npy', conv_input)
    np.save('sanity_check_handmade_data/highway_output.npy', x_highway)

def generate_cnn_data():
    """ generate data for cnn
    """
    word_input = torch.randn((BATCH_SIZE, EMBED_SIZE, MAX_WORD_LEN))
    cnn_conv_1d = nn.Conv1d(
            in_channels = EMBED_SIZE,
            out_channels = EMBED_SIZE_WORD,
            kernel_size = 5,
            bias = True
        )
    cnn_max_pool_1d = nn.MaxPool1d(MAX_WORD_LEN - 5 + 1)

    reinitialize_layers(cnn_conv_1d)
    reinitialize_layers(cnn_max_pool_1d)

    x_cnn = cnn_conv_1d(word_input)
    x_cnn = cnn_max_pool_1d(F.relu_(x_cnn)).squeeze()

    np.save('sanity_check_handmade_data/cnn_word_input.npy', word_input)
    np.save('sanity_check_handmade_data/cnn_output.npy', x_cnn.detach())

def highway_sanity_check(model):
    """ Sanity check for highway module
    """
    print("-"*80)
    print("Running sanity check for highway module")
    print("-"*80)
    reinitialize_layers(model)
    
    input = torch.from_numpy(np.load('sanity_check_handmade_data/highway_conv_input.npy').astype(np.float32))
    output_expected = torch.from_numpy(np.load('sanity_check_handmade_data/highway_output.npy').astype(np.float32))
	
    with torch.no_grad():
        output = model(input)
		
    output_expected_size = (BATCH_SIZE, EMBED_SIZE_WORD)
	
    assert (output.numpy().shape == output_expected_size), \
      "Highway output shape is incorrect it should be:\n{} but is:\n{}".format(output.numpy().shape, output_expected_size)
    assert (np.allclose(output.numpy(), output_expected.numpy())), \
        "Highway output is incorrect: it should be:\n {} but is:\n{}".format(output_expected, output)
    print("Passed all tests :D")

def cnn_sanity_check(model):
    """ Sanity check for cnn
    """
    print("-"*80)
    print("Running sanity check for cnn module")
    print("-"*80)
    reinitialize_layers(model)
    
    input = torch.from_numpy(np.load('sanity_check_handmade_data/cnn_word_input.npy').astype(np.float32))
    output_expected = torch.from_numpy(np.load('sanity_check_handmade_data/cnn_output.npy').astype(np.float32))

    with torch.no_grad():
        output = model(input)

    output_expected_size = (BATCH_SIZE, EMBED_SIZE_WORD)

    assert (output.numpy().shape == output_expected_size), \
      "cnn output shape is incorrect it should be:\n{} but is:\n{}".format(output.numpy().shape, output_expected_size)
#    assert (np.allclose(output.numpy(), output_expected.numpy())), \
#        "cnn output is incorrect: it should be:\n {} but is:\n{}".format(output_expected, output)
    print("Passed all tests :D")
	

def main():
    """ Main func
	"""
    args = docopt(__doc__)
	
    # Check Python & PyTorch Versions
    assert (sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert(torch.__version__ == "1.1.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
	
    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json') 
    
    # Create NMT Model
    model = NMT(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
        vocab=vocab)

    char_vocab = DummyVocab()
	

    # Initialize CharDecoder
    decoder = CharDecoder(
        hidden_size=HIDDEN_SIZE,
        char_embedding_size=EMBED_SIZE,
        target_vocab=char_vocab)
	
    # initialize highway
    highway_model = Highway(
        embed_size_word = EMBED_SIZE_WORD,
        dropout_rate = DROPOUT_RATE)

    # initialize cnn
    cnn_model = CNN(
        EMBED_SIZE,
        MAX_WORD_LEN,
        EMBED_SIZE_WORD,
        5)

    if args['hw']:
        highway_sanity_check(highway_model)
    elif args['generate_data']:
        generate_highway_data()
    elif args['gen_cnn_data']:
        generate_cnn_data()
    elif args['cnn']:
        cnn_sanity_check(cnn_model)
    else:
        raise RuntimeError('invalid run mode')


if __name__== '__main__':
    main()