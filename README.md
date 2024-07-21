# Dante <--> Italian text style transfer with non parallel data
Pytorch implementation of text style transfer as described in [J. Vineet et al. 2018](https://arxiv.org/abs/1808.04339). The goal of this project is achieving neural style transfer between Dante Alighieri sentences and modern italian (and viceversa) training an AI model on non parallel data to learn the two styles and to perform the style transfer in inference phase.

## Theory background
The AI model consist of a Variational Autoencoder in which both Encoder and Decoder are Recurrent Neural Networks.

The aim of the training phase is to learn a disentangled representation of the latent space in order to interpret it as composed of two parts: Style and Content space. During inference a sentence is embedded in this Style-Content latent space, and before feeding the decoder the Style space is replaced with a different Style tensor and concatenated with the original Content.

<img src="Images/TST_model.png" width=40% height=40%>

_From J. Vineet et al. 2018_

### VAEs
A simple Autoencoder is an unsupervised Machine Learning tool that encodes each instance of the dataset in an latent space with a dimension much smaller than the dimension of the input data. A decoder then attempts to reconstruct the input data from the information encoded in the bottleneck.
A Variational Autoencoder adds a structure to the latent space, which is taken as a Gaussian a multivariate distribution as a prior knowledge. The training objective of a VAE is to maximize the ELBO (Evidence LOwer Bound), or, in other words, minimizing this loss function:

$\mathcal{L}(\theta_E, \theta_D )  = - \mathbb{E}_ {p^E_{\theta}(z|x)} [log p^D_{\theta}(x|z)] + \lambda_{KL}\cdot \mathcal{D}_ {KL} \left( p^E_{\theta}(z|x) || p_{latent}^{prior}(z) \right)$

In order to compute meaningful gradients for the backpropagation, after the Encoder pass, the so called *re-parametrization trick* allows to write $z$ as a deterministic and differentiable function of $\theta_E$ and $x$ introducing a Standard Gaussian random varaiable $\epsilon$:

$\epsilon \sim \mathcal{N}_{0,1}$

$\mu, log\sigma^2 = Encoder(x)$

$z = \mu + \sigma \bigodot \epsilon$

<img src="Images/VAE_structure.png" width=60% height=60%>

*Pictorial rapresentation of a VAE*

### RNNs: GRU & LSTM
Recurrent Neural Networks (RNNs) are a class of artificial neural networks designed to handle sequential data. Unlike traditional feedforward neural networks, RNNs have connections that form directed cycles, allowing information to persist and be carried from one step of the sequence to the next. They are well-suited for tasks where temporal dynamics or sequential patterns are important, such as time series forecasting, natural language processing, and speech recognition.

<img src="Images/many_to_many.png" width=60% height=60%>

The input of a RNN is elaborated by the hidden states $h$ in a non-linear way by a matrix $W_{hh}$ and then fed to a linear layer to get an output $y$ through a matrix $W_{hy}$. The hidden layer produces also an hidden vector $v$ which is fed again to the hidden states and determines their evolution. The output at each time step depends not only on the current input but also on the output from the previous time step. This characteristic allows RNNs to capture dependencies and patterns across sequences. The transformation relations can be written as:

$h_t = tanh(W_{hh} h_{t-1} + W_{xh} x_t)$

$y_t = W_{ht}h_t$

<img src="Images/RNN.png" width=60% height=60%>



## Structure (in progress . . .)

## Usage (in progess . . .)


## References
- J. Vineet et al. 2018 *"Disentangled Representation Learning for Non-Parallel Text Style Transfer"*, [https://arxiv.org/abs/1808.04339](https://arxiv.org/abs/1808.04339)
- [Matteo Falcioni Github repository](https://github.com/MatteoFalcioni/pattern_recognition)

