#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, embed_size_char, max_word_len, num_filters, kernel_size=5):
        """ init convolution neural network module.
        
        @param embed_size_char (int): char embed size
        @param max_word_len (int): maximum word length 
        @param num_filters (int): number of output features
        @param kernel_size (int): window size
        """
        super(CNN, self).__init__()
		
        self.embed_size_char = embed_size_char
        self.max_word_len = max_word_len
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.conv_1d = nn.Conv1d(
            in_channels = embed_size_char,
            out_channels = num_filters,
            kernel_size = kernel_size,
            bias = True
        )

        self.max_pool_1d = nn.MaxPool1d(max_word_len - kernel_size + 1)

    def forward(self, input):
        """ take a minibatch of character embeddings, compute word embeddings

        @param input (tensor): a tensor of shape (batch_size, embed_size_char, max_word_len)
        @return x_conv_out: a tensor of shape (batch_size, num_filters)
        """
        # x of size (batch_size, embed_size_word, max_word_len - k + 1)
        x = self.conv_1d(input)
		# x of size (batch_size, embed_size_word)
        x = self.max_pool_1d(F.relu_(x)).squeeze()

        return x

### END YOUR CODE

