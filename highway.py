#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):

    def __init__(self, embed_size_word):
        """ init highway module.

        @param embed_size_word (int): Word embedding size
        """
        super(Highway, self).__init__()

        self.embed_size_word = embed_size_word
#        self.dropout_rate = dropout_rate

        self.project = nn.Linear(self.embed_size_word, self.embed_size_word, bias = True)
        self.gate = nn.Linear(self.embed_size_word, self.embed_size_word, bias = True)
#        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x_conv_out):
        """ Take a minibatch of x_conv_out, compute projection and gate
        obtain the output x_highway by using the gate to combine the projection with the skip-connection

        @param x_conv_out (tensor): a tensor of shape (batch_size, embed_size_word)
        @return a tensor of shape (batch_size, embed_size_word)
        """
        x_proj = torch.relu_(self.project(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
		
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
		
        return x_highway
### END YOUR CODE 

