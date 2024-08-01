import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from models import EarlyStopping


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def entropy(pred_tensor, target_tensor):
    ''' Computing cross entropy between two torch tensor
    
    Input
    ----------
    pred_tensor : torch tensor
    target_tensor : torch tensor


    Returns:
    ----------
    cross entropy
    '''
    return - (target_tensor * torch.log(pred_tensor + 1e-9)).sum(dim=-1).mean()


# ---------------------------------------------------------------------------------------------------------------------- #



def vae_loss(recon_x, x, mu_s, logvar_s, mu_c, logvar_c, l_s = 0.03, l_c = 0.03, CE = nn.CrossEntropyLoss()):
    ''' Computing the loss for VAE, as reconstruction error + KL divergence

    Inputs
    ----------
    recon_x : torch tensor with shape [Batch_size, Sequence_length, Embedding_dim]
    x : torch tensor with shape [Batch_size, Sequence_length]
    mu_s : torch tensor with shape [num_layers, batch_size, style_dim]
    logvar_s : torch tensor with shape [num_layers, batch_size, style_dim]
    mu_c : torch tensor with shape [num_layers, batch_size, content_dim]
    logvar_c : torch tensor with shape [num_layers, batch_size, content_dim]
    l_s : float, coefficient for style KL divergence
    l_c : float, coefficient for content KL divergence
    CE : loss function, initialized with nn.CrossEntropyLoss()


    Returns
    ----------
    reconstruction error + Dkl for style + Dkl for content
    '''

    BCE = CE(recon_x.reshape((recon_x.size(0)*recon_x.size(1),recon_x.size(2))),x.view(-1))
    KLD_s = -0.5 * torch.sum(1 + logvar_s - mu_s.pow(2) - logvar_s.exp())
    KLD_c = -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())
    return BCE + l_s*KLD_s + l_c*KLD_c


# ---------------------------------------------------------------------------------------------------------------------- #



def mul_s_loss(y_s, labels, loss_fn=nn.BCELoss()):
    ''' Computing loss for style classification
    
    Inputs
    ----------
    y_s : torch tensor with shape [Batch_size, 2], predicted style
    labels : torch tensor with shape [Batch_size, 2]
    loss_fn : loss function initialized with nn.BCELoss()
    

    Returns
    ----------
    L_mul_s : float, loss value'''

    L_mul_s = loss_fn(y_s, labels)

    return L_mul_s


# ---------------------------------------------------------------------------------------------------------------------- #



def mul_c_loss(y_c, bow):
    ''' Computing loss for content classification
    
    Input
    ----------
    y_c : torch tensor with shape [Batch_size, vocab_size], predicted Bag of Words
    bow : torch tensor with shape [Batch_size, vocab_size], ground truth Bag of Words
    

    Returns
    ----------
    L_mul_c : float, loss value'''

    L_mul_c = entropy(y_c, bow)

    return L_mul_c


# ---------------------------------------------------------------------------------------------------------------------- #



def dis_s_loss(y_s, labels, loss_fn=nn.BCELoss()):
    ''' Computing adversarial loss for style classification between predicted style given content and ground truth labels
    
    Inputs
    ----------
    y_s : torch tensor with shape [Batch_size, 2], predicted style from content space
    labels : torch tensor with shape [Batch_size, 2]
    loss_fn : loss function initialized with nn.BCELoss()
    

    Returns
    ----------
    L_dis_s : float, loss value'''

    L_dis_s = loss_fn(y_s, labels)

    return L_dis_s


# ---------------------------------------------------------------------------------------------------------------------- #



def dis_c_loss(y_c, bow):
    ''' Computing adversarial loss for content classification between predicted BoW given style and ground truth BoW
    
    Inputs
    ----------
    y_c : torch tensor with shape [Batch_size, vocab_size], predicted Bag of Words
    bow : torch tensor with shape [Batch_size, vocab_size], ground truth Bag of Words
    

    Returns
    ----------
    L_dis_c : float, loss value'''

    L_dis_c = entropy(y_c, bow)

    return L_dis_c


# ---------------------------------------------------------------------------------------------------------------------- #



def adv_s_loss(y_s):
    '''Computing adversarial loss for style classification as entropy of predicted style given content
    
    Inputs
    ----------
    y_s : torch tensor with shape [Batch_size, 2], predicted style from content space
    

    Returns
    ----------
    L_adv_s : float, loss value'''
    
    L_adv_s = entropy(y_s, y_s)

    return L_adv_s


# ---------------------------------------------------------------------------------------------------------------------- #



def adv_c_loss(y_c):
    ''' Computing adversarial loss for content classification as entropy of predicted BoW given style
    
    Inputs
    ----------
    y_c : torch tensor with shape [Batch_size, vocab_size], predicted Bag of Words
    
    

    Returns
    ----------
    L_adv_c : float, loss value'''

    L_adv_c = entropy(y_c, y_c)

    return L_adv_c


# ---------------------------------------------------------------------------------------------------------------------- #



def total_loss(recon_x, x, mu_s, logvar_s, mu_c, logvar_c, y_s, y_c, y_s_given_c, y_c_given_s, labels, bow, l_dk, l_muls=10, l_mulc=3, l_advs=1, l_advc=0.03):
    ''' Computing the loss for VAE, as reconstruction error + KL divergence

    Inputs
    ----------
    recon_x : torch tensor with shape [Batch_size, Sequence_length, Embedding_dim]
    x : torch tensor with shape [Batch_size, Sequence_length]
    mu_s : torch tensor with shape [num_layers, batch_size, style_dim]
    logvar_s : torch tensor with shape [num_layers, batch_size, style_dim]
    mu_c : torch tensor with shape [num_layers, batch_size, content_dim]
    logvar_c : torch tensor with shape [num_layers, batch_size, content_dim]
    y_s : torch tensor with shape [Batch_size, 2], predicted style
    y_c : torch tensor with shape [Batch_size, vocab_size], predicted BoW
    y_s_given_c : torch tensor with shape [Batch_size, 2], predicted style from content space
    y_c_given_s : torch tensor with shape [Batch_size, vocab_size], predicted BoW from style space
    labels : torch tensor with shape [Batch_size, 2], ground truth labels for style
    bow : torch tensor with shape [Batch_size, vocab_size], ground truth Bag of Words
    l_dk : float, coefficient for KL divergences 
    l_muls : float, coefficient for style classification loss
    l_mulc : float, coefficient for content classification loss
    l_advs : float, coefficient for adversarial style loss
    l_advc : float, coefficient for adversarial content loss
    l_s : float, coefficient for style KL divergence
    l_c : float, coefficient for content KL divergence


    Returns
    ----------
    VAE loss + style classification loss + content classification loss - adversarial style loss - adversarial content loss
    '''

    L_VAE = vae_loss(recon_x, x, mu_s, logvar_s, mu_c, logvar_c, l_dk)
    L_muls = mul_s_loss(y_s, labels)
    L_mulc = mul_c_loss(y_c, bow)
    L_advs = adv_s_loss(y_s_given_c)
    L_advc = adv_c_loss(y_c_given_s)

    return L_VAE + l_muls*L_muls + l_mulc*L_mulc - l_advs*L_advs - l_advc*L_advc


# ---------------------------------------------------------------------------------------------------------------------- #



def train_VAE(vae, style_classif, adv_style_classif, content_classif, adv_content_classif, train_loader, val_loader, num_epochs, lr = 4e-4):
    ''' Training function
    
    Inputs
    ----------
    vae : istance of GRUVAE or LSTMVAE class
    style_classif : istance of StyleClassifier class
    adv_style_classif : istance of AdvStyleClassifier class
    content_classif : istance of ContentClassifier class
    adv_content_classif : istance of AdvContentClassifier class
    train_loader : istance of torch.utils.data.Dataloader, training data
    val_loader : istance of torch.utils.data.Dataloader, validation data
    num_epochs : int, number of epochs
    lr : float, learning rate for Adam optimizer


    Returns
    ----------
    average_losses : list of training losses'''

    # parameters for optimizer
    params_tot = list(vae.parameters()) + list(style_classif.parameters()) + list(content_classif.parameters())
    params_dis_s = list(adv_style_classif.parameters())
    params_dis_c = list(adv_content_classif.parameters())

    # three different optimizer
    optimizer_tot = torch.optim.Adam(params_tot, lr = lr)
    optimizer_dis_s = torch.optim.Adam(params_dis_s, lr = lr)
    optimizer_dis_c = torch.optim.Adam(params_dis_c, lr = lr)

    average_losses = []
    val_losses = []
    early_stopping = EarlyStopping()
    
    # for loop through epochs
    for epoch in tqdm(range(num_epochs)):
        train_loss = 0.0
        average_loss = 0.0
        val_loss = 0.0
        average_val_loss = 0.0
        l_dk = 0.03
        

        for  i, (data, bow, labels) in enumerate(train_loader):
            data = data.to(device)
            bow = bow.to(device)
            labels = labels.to(device)

            labels = labels.type(torch.FloatTensor)

            optimizer_tot.zero_grad()
            optimizer_dis_s.zero_grad()
            optimizer_dis_c.zero_grad()

            # forward pass through VAE
            reconstructed_sequence, style, content, mu_s, logvar_s, mu_c, logvar_c = vae(data)
            
            # adversarial predicted style given content
            predicted_adv_style = adv_style_classif(content)
            predicted_adv_style = predicted_adv_style.type(torch.FloatTensor)
            

            L_dis_s = dis_s_loss(predicted_adv_style, labels)

            L_dis_s.backward()
            optimizer_dis_s.step()

            # forward pass through VAE
            reconstructed_sequence, style, content, mu_s, logvar_s, mu_c, logvar_c = vae(data)

            # adversarial predicted BoW given style
            predicted_adv_content = adv_content_classif(style)

            L_dis_c = dis_c_loss(predicted_adv_content, bow)

            L_dis_c.backward()
            optimizer_dis_c.step()

            # forward pass through VAE
            reconstructed_sequence, style, content, mu_s, logvar_s, mu_c, logvar_c = vae(data)
            
            # predicted style, content, style given content and content given style
            y_s = style_classif(style)
            y_c = content_classif(content)
            y_s_given_c = adv_style_classif(content)
            y_c_given_s = adv_content_classif(style)


            reconstructed_sequence = reconstructed_sequence.to(device)
            reconstructed_sequence = torch.FloatTensor(reconstructed_sequence)
            
            # comuting total training loss
            loss_tot = total_loss(reconstructed_sequence.to(device), 
                                  data, 
                                  mu_s, 
                                  logvar_s, 
                                  mu_c, 
                                  logvar_c, 
                                  y_s.to(device), 
                                  y_c.to(device), 
                                  y_s_given_c.to(device), 
                                  y_c_given_s.to(device), 
                                  labels.to(device), 
                                  bow.to(device), 
                                  l_dk)
            
            loss_tot.backward()
            train_loss += loss_tot.item()


            optimizer_tot.step()
            
            if (i + 1) % 5000 == 0:
                print(f'Train Epoch: {epoch+1} [{i * len(data)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tLoss: {loss_tot.item() / len(data):.6f}')
        
        

        with torch.no_grad():
            for i, (data, bow, labels) in enumerate(val_loader):
                data = data.to(device)
                bow = bow.to(device)
                labels = labels.to(device)

                labels = labels.type(torch.FloatTensor)

                # forward pass through VAE
                reconstructed_sequence, style, content, mu_s, logvar_s, mu_c, logvar_c = vae(data)
                
                # predicted style, content, style given content and content given style
                y_s = style_classif(style)
                y_c = content_classif(content)
                y_s_given_c = adv_style_classif(content)
                y_c_given_s = adv_content_classif(style)

                reconstructed_sequence = reconstructed_sequence.to(device)
                reconstructed_sequence = torch.FloatTensor(reconstructed_sequence)
                
                # comuting total validation loss
                val_loss_tot = total_loss(reconstructed_sequence.to(device), 
                                          data, mu_s, 
                                          logvar_s, mu_c, 
                                          logvar_c, y_s.to(device), 
                                          y_c.to(device), 
                                          y_s_given_c.to(device), 
                                          y_c_given_s.to(device), 
                                          labels.to(device), 
                                          bow.to(device), 
                                          l_dk)
                
                val_loss += val_loss_tot.item()


                
                if (i + 1) % 5000 == 0:
                    print(f'Train Epoch: {epoch+1} [{i * len(data)}/{len(val_loader.dataset)} ({100. * i / len(val_loader):.0f}%)]\tLoss: {val_loss_tot.item() / len(data):.6f}')
            
            
        average_loss = train_loss / len(train_loader.dataset)
        average_losses.append(average_loss)

        average_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(average_val_loss)
        
        # printing average training and validation losses
        print(f'====> Epoch: {epoch+1} Average train loss: {average_loss:.4f}, Average val loss: {average_val_loss:.4f}')
        if average_val_loss > 1.2*average_loss or early_stopping(average_val_loss):
            print('Early stopping\n')
            break
    
    # plotting training and validation curve at the end of the for loop 
    plt.plot(np.linspace(1,num_epochs,len(average_losses)), average_losses, c = 'darkcyan',label = 'train')
    plt.plot(np.linspace(1,num_epochs,len(val_losses)), val_losses, c = 'orange',label = 'val')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title('Training')
    plt.show()

    return average_losses