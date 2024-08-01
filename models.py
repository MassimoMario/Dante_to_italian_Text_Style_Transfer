import torch
import torch.nn as nn
from torch.nn import functional as F


class GRUVAE(nn.Module):
    ''' Class of a VAE where both Encoder and Decoder are RNNs with GRU units

    Attributes
    ----------
    embedding_matrix : 2d torch tensor matrix from word2vec embedding
    hidden_dim : int, dimension of RNNs hidden state
    latent_dim : int, dimension of the VAE latent space
    style_dim : int, dimension of the style space within the latent space
    content_dim : int, dimension of the content space within the latent 
    vocab_size : int, number of unique tokens in the dataset
    sos_token : torch tensor of the 'start of the sequence' token
    num_layers : int, number of RNNs layers

    
    Methods
    -------
    forward(x) : perform the forward pass of the VAE
    reparametrization(mu, log_var) : perform the reparametrization trick
    reconstruction(x) : inference for reconstruction
    TST(x, new_style) : inference for Text Style Transfer
    '''

    def __init__(self, embedding_matrix, hidden_dim, latent_dim, style_dim, content_dim, vocab_size, sos_token, num_layers):
        super(GRUVAE, self).__init__()

        self.embedding_dim = embedding_matrix.shape[1]
        self.sos_token = sos_token

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.encoder_style = nn.GRU(self.embedding_dim, hidden_dim, num_layers, batch_first=True) # (N,B,H) N batches, B sequence length, H input dim
    
        self.fcmu_s = nn.Linear(hidden_dim, style_dim)
        self.fcvar_s = nn.Linear(hidden_dim, style_dim)
        self.fcmu_c = nn.Linear(hidden_dim, content_dim)
        self.fcvar_c = nn.Linear(hidden_dim, content_dim)


        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(self.embedding_dim, hidden_dim, num_layers,batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        ''' Performs the VAE forward pass 
        Input
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
        

        Returns
        -------
        reconstructed_sequence : torch tensor with shape [Batch_size, Sequence_length, Embedding_dimension] 
        style : torch tensor with shape [num_layers, batch_size, style_dim]
        content : torch tensor with shape [num_layers, batch_size, content_dim]
        mu_s : torch tensor with shape [num_layers, batch_size, style_dim]
        logvar_s : [num_layers, batch_size, style_dim]
        mu_c : [num_layers, batch_size, content_dim]
        logvar_c : [num_layers, batch_size, content_dim]'''

        # embedding input and GRU encoder pass
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)
        _, hn = self.encoder_style(embedded_input)
        

        # computing mu and log_var for style and content space
        mu_s = self.fcmu_s(hn)
        logvar_s = self.fcvar_s(hn)
        mu_c = self.fcmu_c(hn)
        logvar_c = self.fcvar_c(hn)

        # reparametrization for style and content
        style = self.reparametrization(mu_s, logvar_s)
        content = self.reparametrization(mu_c, logvar_c)

        # concatenating style and content space
        z = torch.cat((style,content), dim = 2)
        z = self.fc(z)

        # prepare sos_token for the decoder
        sos_token = self.sos_token.repeat(x.size(0),1)
        sos_token = self.embedding(sos_token)
        sos_token = self.layer_norm(sos_token)

        decoder_input = torch.cat((sos_token, embedded_input), dim = 1)
        decoder_input = decoder_input[:,:-1,:]

        # reconstructing sequence through the decoder giving z as hidden state for each time step
        reconstructed_sequence = []
        for t in range(x.shape[1]):
            outputs, _ = self.decoder(decoder_input[:,:t+1,:], z)
            reconstructed_sequence.append(outputs[:,-1,:].unsqueeze(1))

        # concatenating reconstructed words and push them into vocab_size dimensions
        reconstructed_sequence = torch.cat(reconstructed_sequence, dim=1)
        reconstructed_sequence = self.fc_out(reconstructed_sequence)

        return reconstructed_sequence, style, content, mu_s, logvar_s, mu_c, logvar_c


    def reparametrization(self, mu, log_var):
        ''' Reparametrization trick
        
        Inputs
        -------
        mu : torch tensor
        log_var : torch tensor
            
        
        Returns
        -------
        mu + eps*std : torch tensor with the same shape as mu and log_var'''
        
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)

        return mu + eps*std
    
    def reconstruction(self, x):
        ''' Reconstruction function for inference
        Input
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
            
        
        Returns
        -------
        outputs : torch tensor with shape [Batch_size, Sequence_length, Embedding_dim], the reconstructed sentence'''
        
        # embedding input and GRU encoder pass
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)
        _, hn = self.encoder_style(embedded_input)
        
        # computing mu and log_var for style and content space
        mu_s = self.fcmu_s(hn)
        logvar_s = self.fcvar_s(hn)
        mu_c = self.fcmu_c(hn)
        logvar_c = self.fcvar_c(hn)

        # reparametrization for style and content
        style = self.reparametrization(mu_s, logvar_s)
        content = self.reparametrization(mu_c, logvar_c)

        # concatenating style and content space
        z = torch.cat((style,content), dim = 2)
        z = self.fc(z)

        # prepare sos_token for the decoder
        sos_token = self.sos_token.repeat(x.size(0),1)
        sos_token = self.embedding(sos_token)
        sos_token = self.layer_norm(sos_token)


        # decoder pass where the input is the previous output
        output = sos_token
        for _ in range(x.shape[1]):
            outputs, _ = self.decoder(output, z)
            outputs = self.fc_out(outputs)
            next_token = torch.argmax(F.softmax(outputs[:,-1,:], dim = -1), dim=-1)
            next_token = self.embedding(next_token)
            output = torch.cat((output, next_token.unsqueeze(1)), dim=1)
        
        
        return outputs
    
    def TST(self, x, new_style):
        ''' Performs Text Style Transfer
        
        Inputs
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
        new_style : torch tensor with shape [num_layers, 1, style_dim]
            
        
        Returns
        -------
        outputs : torch tensor with shape [Batch_size, Sequence_length, Embedding_dim], transferred sentence'''
        
        # embedding input and GRU encoder pass
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)
        _, hn = self.encoder_style(embedded_input)

        # computing mu and logvar only for content space
        mu_c = self.fcmu_c(hn)
        log_var_c = self.fcvar_c(hn)

        content = self.reparametrization(mu_c,log_var_c)

        # concatenating content with the new style space
        z = torch.cat((new_style,content), dim = 2)
        z = self.fc(z)

        # prepare sos_token for the decoder
        sos_token = self.sos_token.repeat(x.size(0),1)
        sos_token = self.embedding(sos_token)
        sos_token = self.layer_norm(sos_token)


        # decoder pass where the input is the previous output
        output = sos_token
        for _ in range(x.shape[1]):
            outputs, _ = self.decoder(output, z)
            outputs = self.fc_out(outputs)
            next_token = self.embedding(torch.argmax(F.softmax(outputs[:,-1,:], dim = -1), dim=-1))
            output = torch.cat((output, next_token.unsqueeze(1)), dim=1)
        
        
        return outputs
    


# ---------------------------------------------------------------------------------------------------------------------- #



class LSTMVAE(nn.Module):

    ''' Class of a VAE where the Encoder is a RNN with LSTM unit and the Decoder a RNN with GRU unit

    Attributes
    ----------
    embedding_matrix : 2d torch tensor matrix from word2vec embedding
    hidden_dim : int, dimension of RNNs hidden state
    latent_dim : int, dimension of the VAE latent space
    style_dim : int, dimension of the style space within the latent space
    content_dim : int, dimension of the content space within the latent 
    vocab_size : int, number of unique tokens in the dataset
    sos_token : torch tensor of the 'start of the sequence' token
    num_layers : int, number of RNNs layers

    
    Methods
    -------
    forward(x) : perform the forward pass of the VAE
    reparametrization(mu, log_var) : perform the reparametrization trick
    reconstruction(x) : inference for reconstruction
    TST(x, new_style) : inference for Text Style Transfer
    '''

    def __init__(self, embedding_matrix, hidden_dim, latent_dim, style_dim, content_dim, vocab_size, sos_token, num_layers):
        super(LSTMVAE, self).__init__()

        self.embedding_dim = embedding_matrix.shape[1]
        self.sos_token = sos_token

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.encoder = nn.LSTM(self.embedding_dim, hidden_dim, num_layers, batch_first=True) # (N,B,H) N batches, B sequence length, H input dim
       
        self.fcmu_s = nn.Linear(hidden_dim, style_dim)
        self.fcvar_s = nn.Linear(hidden_dim, style_dim)
        self.fcmu_c = nn.Linear(hidden_dim, content_dim)
        self.fcvar_c = nn.Linear(hidden_dim, content_dim)


        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(self.embedding_dim, hidden_dim, num_layers,batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        ''' Performs the VAE forward pass 
        Input
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
        

        Returns
        -------
        reconstructed_sequence : torch tensor with shape [Batch_size, Sequence_length, Embedding_dimension] 
        style : torch tensor with shape [num_layers, batch_size, style_dim]
        content : torch tensor with shape [num_layers, batch_size, content_dim]
        mu_s : torch tensor with shape [num_layers, batch_size, style_dim]
        logvar_s : [num_layers, batch_size, style_dim]
        mu_c : [num_layers, batch_size, content_dim]
        logvar_c : [num_layers, batch_size, content_dim]'''

        # embedding input and GRU encoder pass
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)
        _, (hn, cn) = self.encoder(embedded_input)
        
        # computing mu and log_var for style and content space
        mu_s = self.fcmu_s(hn)
        logvar_s = self.fcvar_s(hn)
        mu_c = self.fcmu_c(hn)
        logvar_c = self.fcvar_c(hn)

        # reparametrization for style and content
        style = self.reparametrization(mu_s, logvar_s)
        content = self.reparametrization(mu_c, logvar_c)

        # concatenating style and content space
        z = torch.cat((style,content), dim = 2)
        z = self.fc(z)

        # prepare sos_token for the decoder
        sos_token = self.sos_token.repeat(x.size(0),1)
        sos_token = self.embedding(sos_token)
        decoder_input = torch.cat((sos_token, embedded_input), dim = 1)
        decoder_input = decoder_input[:,:-1,:]

        
        # decoder pass with the input as the previous output
        reconstructed_sequence = []
        for t in range(x.shape[1]):
            outputs, _ = self.decoder(decoder_input[:,:t+1,:], z)
            reconstructed_sequence.append(outputs[:,-1,:].unsqueeze(1))

        reconstructed_sequence = torch.cat(reconstructed_sequence, dim=1)
        reconstructed_sequence = self.fc_out(reconstructed_sequence)

        return reconstructed_sequence, style, content, mu_s, logvar_s, mu_c, logvar_c


    def reparametrization(self, mu, log_var):
        ''' Reparametrization trick
        
        Inputs
        -------
        mu : torch tensor
        log_var : torch tensor
            
        
        Returns
        -------
        mu + eps*std : torch tensor with the same shape as mu and log_var'''

        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def reconstruction(self, x):
        ''' Reconstruction function for inference
        Input
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
            
        
        Returns
        -------
        outputs : torch tensor with shape [Batch_size, Sequence_length, Embedding_dim], the reconstructed sentence'''
        
        # embedding input and GRU encoder pass
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)
        _, (hn, cn) = self.encoder(embedded_input)
        
        # computing mu and log_var for style and content space
        mu_s = self.fcmu_s(hn)
        logvar_s = self.fcvar_s(hn)
        mu_c = self.fcmu_c(hn)
        logvar_c = self.fcvar_c(hn)

        # reparametrization for style and content
        style = self.reparametrization(mu_s, logvar_s)
        content = self.reparametrization(mu_c, logvar_c)

        # concatenating style and content space
        z = torch.cat((style,content), dim = 2)
        z = self.fc(z)

        # prepare sos_token for the decoder
        sos_token = self.sos_token.repeat(x.size(0),1)
        sos_token = self.embedding(sos_token)


        # decoder pass where the input is the previous output
        output = sos_token
        for _ in range(x.shape[1]):
            outputs, _ = self.decoder(output, z)
            outputs = self.fc_out(outputs)
            next_token = torch.argmax(F.softmax(outputs[:,-1,:], dim = -1), dim=-1)
            next_token = self.embedding(next_token)
            output = torch.cat((output, next_token.unsqueeze(1)), dim=1)
        
        return outputs
    
    def TST(self, x, new_style):

        ''' Performs Text Style Transfer
        
        Inputs
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
        new_style : torch tensor with shape [num_layers, 1, style_dim]
            
        
        Returns
        -------
        outputs : torch tensor with shape [Batch_size, Sequence_length, Embedding_dim], transferred sentence'''

        # embedding input and GRU encoder pass
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)
        _, (hn, cn) = self.encoder(embedded_input)

        # computing mu and log_var only for content space
        mu_c = self.fcmu_c(hn)
        log_var_c = self.fcvar_c(hn)

        content = self.reparametrization(mu_c,log_var_c)

        # concatenating content with the new style space
        z = torch.cat((new_style,content), dim = 2)
        z = self.fc(z)

        # prepare sos_token for the decoder
        sos_token = self.sos_token.repeat(x.size(0),1)
        sos_token = self.embedding(sos_token)

        # decoder pass where the input is the previous output
        output = sos_token
        for _ in range(x.shape[1]):
            outputs, _ = self.decoder(output, z)
            outputs = self.fc_out(outputs)
            next_token = self.embedding(torch.argmax(F.softmax(outputs[:,-1,:], dim = -1), dim=-1))
            output = torch.cat((output, next_token.unsqueeze(1)), dim=1)

        return outputs
    


# ---------------------------------------------------------------------------------------------------------------------- #



class StyleClassifier(nn.Module):
    ''' Class for Style classification from the last hidden state of the Encoder
    
    Attributes
    ----------
    input_dim : int
    
    Methods:
    ----------
    forward(x) : forward pass through a MLP
    '''

    def __init__(self, input_dim):
        super(StyleClassifier, self).__init__()

        self.input_dim = input_dim
        self.mlp = nn.Sequential(nn.Linear(input_dim,int(input_dim*0.5)),
                                  nn.ReLU(),
                                  nn.Linear(int(input_dim*0.5),2))
        
    def forward(self,x):
        ''' Forward pass
        
        Input
        ----------
        x : torch tensor with shape [1, Batch_size, Style_dim]
            
        
        Returns
        ----------
        out : torch tensor with shape [Batch_size, 2]'''
        
        out = self.mlp(x)
        out = F.softmax(out, dim=-1)
        return out.view(out.shape[1], out.shape[2])
    

# ---------------------------------------------------------------------------------------------------------------------- #



class AdvStyleClassifier(nn.Module):
    ''' Class for Adversarial Style classification from the last hidden state of the Encoder
    
    Attributes
    ----------
    input_dim : int
    
    Methods:
    ----------
    forward(x) : forward pass through a MLP
    '''

    def __init__(self, input_dim):
        super(AdvStyleClassifier, self).__init__()
        self.input_dim = input_dim
        self.mlp = nn.Sequential(nn.Linear(input_dim,int(input_dim*0.5)),
                                  nn.ReLU(),
                                  nn.Linear(int(input_dim*0.5),2))
        
    def forward(self,x):
        ''' Forward pass
        
        Input
        ----------
        x : torch tensor with shape [1, Batch_size, Content_dim]
            
        
        Returns
        ----------
        out : torch tensor with shape [Batch_size, 2]'''
        
        out = self.mlp(x)
        out = F.softmax(out, dim=-1)
        return out.view(out.shape[1], out.shape[2])
    

# ---------------------------------------------------------------------------------------------------------------------- #



class ContentClassifier(nn.Module):
    ''' Class for Content classification from the last hidden state of the Encoder
    
    Attributes
    ----------
    input_dim : int
    vocab_size : number of unique tokens in the dataset
    
    Methods:
    ----------
    forward(x) : forward pass through a MLP
    '''

    def __init__(self, input_dim, vocab_size):
        super(ContentClassifier, self).__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.fc = nn.Linear(input_dim,vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=2)
        
    def forward(self,x):
        ''' Forward pass
        
        Input
        ----------
        x : torch tensor with shape [1, Batch_size, Content_dim]
            
        
        Returns
        ----------
        out : torch tensor with shape [Batch_size, vocab_size]'''
        
        out = self.fc(x)
        out = F.softmax(out,dim=2)
        return out.squeeze(0)
    

# ---------------------------------------------------------------------------------------------------------------------- #



class AdvContentClassifier(nn.Module):

    ''' Class for Adversarial Style classification from the last hidden state of the Encoder
    
    Attributes
    ----------
    input_dim : int
    vocab_size : number of unique tokens in the dataset
    
    Methods:
    ----------
    forward(x) : forward pass through a MLP
    '''

    def __init__(self, input_dim, vocab_size):
        super(AdvContentClassifier, self).__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.fc = nn.Linear(input_dim,vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=2)
        
    def forward(self,x):
        ''' Forward pass
        
        Input
        ----------
        x : torch tensor with shape [1, Batch_size, Content_dim]
            
        
        Returns
        ----------
        out : torch tensor with shape [Batch_size, vocab_size]'''
        
        out = self.fc(x)
        out = F.softmax(out,dim=2)
        return out.squeeze(0)
    

# ---------------------------------------------------------------------------------------------------------------------- #



class EarlyStopping:
    ''' Class used in training for checking if an early stopping is required
    
    Attributes
    ----------
    patience : int, how many epochs wait until stop
    min_delta : float, minimum difference between previous and current loss


    Methods:
    ---------
    __call__(current_score) : checking if early stop is required

    '''
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        ''' Checks if early stopping is required
        
        Input
        ----------
        current_score : float
        
        
        Returns
        ----------
        Bool : True if early stopping'''

        if self.best_score is None:
            self.best_score = current_score
            return False

        if current_score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = current_score
            self.counter = 0

        return False