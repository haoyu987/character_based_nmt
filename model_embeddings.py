#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab.char2id['<pad>']
        self.embed_size = embed_size
        self.embed_size_char = 50    # predefined character embedding size
        self.max_word_len = 21       # predefined maximum word length
        self.dropout_rate = 0.3 
        self.char_embedding = nn.Embedding(
            num_embeddings = len(vocab.char2id),
            embedding_dim  = self.embed_size_char,
            padding_idx    = pad_token_idx
        )

        self.CNN = CNN(
            embed_size_char = self.embed_size_char,
            max_word_len    = self.max_word_len,
            num_filters     = self.embed_size,
            )

        self.highway = Highway(
            embed_size_word = self.embed_size
            )

        self.dropout = nn.Dropout(p=self.dropout_rate)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        sent_len, batch_size, max_word_length = input.shape
        # input_reshape of shape (sent_len * batch_size, max_word_length)
        input_reshape = input.reshape(sent_len * batch_size, max_word_length)
        # embedding_output of shape (sent_len * batch_size, max_word_length, embed_size_char)
        embedding_output = self.char_embedding(input_reshape)
        CNN_output = self.CNN(embedding_output.permute(0, 2, 1))
        highway_output = self.highway(CNN_output)
        output = self.dropout(highway_output)
        output_reshape = output.reshape(sent_len, batch_size, -1)

        return output_reshape
        ### END YOUR CODE

