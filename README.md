# *Dante <--> Modern Italian* Text Style Transfer with non parallel data
Pytorch implementation of text style transfer as described in [J. Vineet et al. 2018](https://arxiv.org/abs/1808.04339). The goal of this project is achieving neural style transfer between Dante Alighieri sentences and modern italian (and viceversa) training an AI model on non parallel data to learn the two styles and to perform the style transfer during inference phase.

## Theory background
The AI model consist of a Variational Autoencoder in which both Encoder and Decoder are Recurrent Neural Networks.

The aim of the training phase is to learn a disentangled representation of the latent space in order to interpret it as composed of two parts: Style and Content space. During inference a sentence is embedded in this Style-Content latent space and, before feeding the decoder, the Style space is replaced with a different Style tensor and concatenated with the original Content.

<img src="Images/TST_model.png" width=50% height=50%>

_From J. Vineet et al. 2018_

### VAEs
A simple Autoencoder is an unsupervised Machine Learning tool that encodes each instance of the dataset in an latent space with a dimension much smaller than the dimension of the input data. A decoder then attempts to reconstruct the input data from the information encoded in the latent space.

A Variational Autoencoder adds a structure to the latent space, which is taken as a Gaussian a multivariate distribution as a prior knowledge. The training objective of a VAE is to maximize the ELBO (Evidence LOwer Bound), or, in other words, minimizing this loss function:

$\mathcal{L}_ {VAE} (\theta_E, \theta_D )  = - \mathbb{E}_ {p^E_{\theta}(z|x)} [log p^D_{\theta}(x|z)] + \lambda_{KL}\cdot \mathcal{D}_ {KL} \left( p^E_{\theta}(z|x) || p_{latent}^{prior}(z) \right)$

Where $\mathcal{D}_ {KL}$ is the Kullback–Leibler divergence, $\lambda_{KL} \in \mathbb{R}^+$ is an hyperparameter, and $p_{latent}^{prior}(z) = \mathcal{N}(\vec{0}, \mathbb{I})$

In order to compute meaningful gradients for the backpropagation, after the Encoder pass, the so called *re-parametrization trick* allows to write the latent space variable $z$ as a deterministic and differentiable function of $\theta_E$ and $x$ introducing a Standard Gaussian random varaiable $\epsilon$:

$\epsilon \sim \mathcal{N} (\vec{0}, \mathbb{I})$

$\mu, log \left( \sigma^2 \right) = Encoder(x)$

$z = \mu + \sigma \bigodot \epsilon$

Where $\bigodot$ represent the element-wise product.

<img src="Images/VAE_structure.png" width=70% height=70%>

*Pictorial rapresentation of a VAE*

### RNNs: GRU & LSTM
Recurrent Neural Networks (RNNs) are a class of artificial neural networks designed to handle sequential data. Unlike traditional feedforward neural networks, RNNs have connections that form directed cycles, allowing information to persist and be carried from one step of the sequence to the next. They are well-suited for tasks where temporal dynamics or sequential patterns are important, such as time series forecasting, natural language processing, and speech recognition.

<img src="Images/many_to_many.png" width=65% height=65%>

The input of a RNN is elaborated by the hidden states $h$ in a non-linear way by a matrix $W_{hh}$ and then fed to a linear layer to get an output $y$ through a matrix $W_{hy}$. The hidden layer produces also an hidden vector $v$ which is fed again to the hidden states and determines their evolution. The output at each time step depends not only on the current input but also on the output from the previous time step. This characteristic allows RNNs to capture dependencies and patterns across sequences. The transformation relations can be written as:

$h_t = tanh(W_{hh} h_{t-1} + W_{xh} x_t)$

$y_t = W_{ht}h_t$

<img src="Images/RNN.png" width=65% height=65%>

To overcome RNNs limitations in processing long term dependencies (vanishing and exploding gradient), more sophisticated of RNNs have been designed: Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU).

LSTMs incorporate three types of gates:  the input gate, the forget gate, and the output gate. These gates work together to control the flow of information into the cell state, out of the cell state, and how much of the cell state should be updated. The input gate determines how much of the new information should be added to the cell state, while he forget state decides which information from the previous cell state $C_{t-1}$ should be forgotten, and finally the output gate determines which part of the cell state should be output as the hidden state. This structure allows LSTMs to remember information over longer sequences and selectively retain or discard information as needed. The equations below describe the forward pass in a LSTM cell:

<img src="Images/LSTM_formula.png" width=35% height=35%>

Where $t$ denotes time step. $W_{\alpha}$ and $U_{\alpha}$ matrices contain, respectively, the weights of the input and recurrent connections, $b_{\alpha}$ refers layer biases and $\sigma_{\alpha}$ to gates activation function, where the subscript $\alpha$ can refer to input $i$, forget $f$, output $o$ gate or memory cell $c$. $x_t$ and $h_t$ are input and hidden state at time step $t$.

A sketch of LSTM cell structure:

<img src="Images/LSTM_structure.png" width=45% height=45%>

A simplified alternative to LSTM is the GRU unit. GRUs use just two types of gates: the update gate and the reset gate. The update gate determines how much of the past information needs to be passed to the future, while the reset gate decides how much of the of the previous hidden state should be forgotten. 

The equations for a forward pass in a GRU cell are:

<img src="Images/GRU_formula.png" width=50% height=50%>

Where $r_t, z_t$ and $n_t$ are respectively the reset, update and new gates. $\sigma$ is the activation function and $\bigodot$ the Hadamard product. $h_t$ is the hidden state at time step $t$.

A sketch of GRU cell structure is shown below:

<img src="Images/GRU_structure.png" width=45% height=45%>


## Structure (in progress . . .)

## Usage (in progess . . .)


## References
- J. Vineet et al. 2018 *"Disentangled Representation Learning for Non-Parallel Text Style Transfer"*, [https://arxiv.org/abs/1808.04339](https://arxiv.org/abs/1808.04339)
- [Matteo Falcioni Github repository](https://github.com/MatteoFalcioni/pattern_recognition)

