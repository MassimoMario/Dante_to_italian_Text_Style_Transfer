import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from gensim.models import Word2Vec




def BoW(tensor, vocab_size, sequence_length):
    ''' Function computing the Bag of Words from a tensor containing word indices
    
    Input
    ----------
    tensor : torch tensor with shape [Batch_size, Sequence_length]
    vocab_size : int, number of unique tokens
    sequence_length : int
    
    
    Returns
    ----------
    bow : torch tensor with shape [Batch_size, vocab_size], Bag of Words of the input sentence'''

    bow = torch.zeros(size = (tensor.shape[0],vocab_size))

    for i in range(tensor.shape[0]):
        for num in tensor[i]:
            bow[i][num] = (tensor[i] == num).sum().item()/sequence_length

    return torch.FloatTensor(bow)


# ---------------------------------------------------------------------------------------------------------------------- #



def divide_text(text, sequence_length):
    ''' Function dividing text in order to feed the Word2vec model
    
    Inputs
    ----------
    text : text corpus from a file
    sequence_length : int
    
    
    Returns
    ----------
    output_text : 2D list of words with shape [text_length/sequence_length, sequence_length]'''

    words = text.split()
    grouped_words = [' '.join(words[i:i+sequence_length]) for i in range(0,len(words),int(sequence_length))]  
    output_text = [grouped_words[i].split() for i in range(len(grouped_words)) if len(grouped_words[i].split()) == sequence_length]

    return output_text


# ---------------------------------------------------------------------------------------------------------------------- #



def custom_dataset(file1 : str,file2 : str, sequence_length, embedding_dim, batch_size, training_fraction):
    ''' Function creating dataset
    
    Inputs
    ----------
    file1 : str, name of the file containing the first corpus
    file2 : str, name of the file containing the second corpus
    sequence_length : int
    embedding_dim : int, number of dimension for the embedded words using Word2vec model
    batch_size : int
    training_fraction : float, fraction of training data
    
    
    Returns
    ----------
    dataloader_train : istance of torch.utils.data.Dataloader, training data
    dataloader_val : istance of torch.utils.data.Dataloader, validation data
    embedding_dim : int
    embedding_matrix : 2d torch tensor matrix from word2vec embedding
    word2vec : trained Word2vec model
    idx2word : dictionary from indices to words
    word2idx : dictionart from words to indices
    vocab_size : int, number of unique tokens
    style0_val : torch tensor containing every validation data belonging to first style
    style1_val : torch tensor containing every validation data belonging to second style'''

    # reading the two corpus
    with open(file1, 'r', encoding='utf-8') as f:
        text1 = f.read()


    with open(file2, 'r', encoding='utf-8') as f:
        text2 = f.read()

    # adding a special token for the start of the sequence
    text1 = '<sos> ' + text1 
    text = text1 + ' ' + text2

    # divide the whole text to feed the Word2vec model
    divided_text = divide_text(text, sequence_length)

    # training the Word2vec model with the whole corpus
    word2vec = Word2Vec(divided_text, vector_size = embedding_dim, window = sequence_length, min_count=1, workers=4, epochs = 50)
    word2vec.train(divided_text, total_examples=word2vec.corpus_count, epochs=word2vec.epochs)

    # Get the embedding dimension
    embedding_dim = word2vec.wv.vector_size

    # Prepare the embedding matrix
    vocab_size = len(word2vec.wv)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    word2idx = {word: idx for idx, word in enumerate(word2vec.wv.index_to_key)}
    idx2word = {idx: word for idx, word in enumerate(word2vec.wv.index_to_key)}

    # creating the embedding matrix from the trained Word2vec model
    for word, idx in word2idx.items():
        embedding_matrix[idx] = word2vec.wv[word]

    
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)


    # dividing only the first text
    text1_divided = divide_text(text1, sequence_length)
    data1 = torch.LongTensor([[word2idx[char] for char in text1_divided[i]] for i in range(len(text1_divided))])

    
    # dividing only the second text
    text2_divided = divide_text(text2, sequence_length)
    data2 = torch.LongTensor([[word2idx[char] for char in text2_divided[i]] for i in range(len(text2_divided))])


    # defining training and validation data for the first style
    data1_train = data1[:int(training_fraction * data1.shape[0])]
    data1_val = data1[int(training_fraction * data1.shape[0]):]

    # defining training and validation data for the second style
    data2_train = data2[:int(training_fraction * data2.shape[0])]
    data2_val = data2[int(training_fraction * data2.shape[0]):]


    # creating training and validation labels for the first style
    label0_train = torch.tensor([[1,0]]).repeat(data1_train.shape[0],1)
    label0_val = torch.tensor([[1,0]]).repeat(data1_val.shape[0],1)

    # creating training and validation labels for the second style
    label1_train = torch.tensor([[0,1]]).repeat(data2_train.shape[0],1)
    label1_val = torch.tensor([[0,1]]).repeat(data2_val.shape[0],1)


    # creating training and validation labels
    labels_train = torch.cat((label0_train, label1_train), dim = 0)
    labels_val = torch.cat((label0_val, label1_val), dim = 0)

    # creating training and validation data
    data_train = torch.cat((data1_train, data2_train), dim = 0)
    data_val = torch.cat((data1_val, data2_val), dim = 0)

    data_train = torch.LongTensor(data_train)
    labels_train = labels_train.type(torch.LongTensor)
    bow_train = BoW(data_train, vocab_size, sequence_length)

    dataset_train = TensorDataset(data_train, bow_train, labels_train)

    # Create a training DataLoader with shuffling enabled
    dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle=True)
   


    data_val = torch.LongTensor(data_val)
    labels_val = labels_val.type(torch.LongTensor)
    bow_val = BoW(data_val, vocab_size, sequence_length)

    dataset_val = TensorDataset(data_val, bow_val, labels_val)

    # Create a validation DataLoader with shuffling enabled
    dataloader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle = True)
    

    # validation data for both styles that will be used during inference
    style0_val = torch.LongTensor(data1_val)
    style1_val = torch.LongTensor(data2_val)
    
    return dataloader_train, dataloader_val, embedding_dim, embedding_matrix, word2vec, idx2word, word2idx, vocab_size, style0_val, style1_val