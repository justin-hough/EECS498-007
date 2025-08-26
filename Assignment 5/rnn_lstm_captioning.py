import math
from typing import Optional, Tuple

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import feature_extraction


def hello_rnn_lstm_captioning():
    print("Hello from rnn_lstm_captioning.py!")


class ImageEncoder(nn.Module):
    """
    Convolutional network that accepts images as input and outputs their spatial
    grid features. This module servesx as the image encoder in image captioning
    model. We will use a tiny RegNet-X 400MF model that is initialized with
    ImageNet-pretrained weights from Torchvision library.

    NOTE: We could use any convolutional network architecture, but we opt for a
    tiny RegNet model so it can train decently with a single K80 Colab GPU.
    """

    def __init__(self, pretrained: bool = True, verbose: bool = True):
        """
        Args:
            pretrained: Whether to initialize this model with pretrained weights
                from Torchvision library.
            verbose: Whether to log expected output shapes during instantiation.
        """
        super().__init__()
        self.cnn = torchvision.models.regnet_x_400mf(pretrained=pretrained)

        # Torchvision models return global average pooled features by default.
        # Our attention-based models may require spatial grid features. So we
        # wrap the ConvNet with torchvision's feature extractor. We will get
        # the spatial features right before the final classification layer.
        self.backbone = feature_extraction.create_feature_extractor(
            self.cnn, return_nodes={"trunk_output.block4": "c5"}
        )
        # We call these features "c5", a name that may sound familiar from the
        # object detection assignment. :-)

        # Pass a dummy batch of input images to infer output shape.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))["c5"]
        self._out_channels = dummy_out.shape[1]

        if verbose:
            print("For input images in NCHW format, shape (2, 3, 224, 224)")
            print(f"Shape of output c5 features: {dummy_out.shape}")

        # Input image batches are expected to be float tensors in range [0, 1].
        # However, the backbone here expects these tensors to be normalized by
        # ImageNet color mean/std (as it was trained that way).
        # We define a function to transform the input images before extraction:
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def out_channels(self):
        """
        Number of output channels in extracted image features. You may access
        this value freely to define more modules to go with this encoder.
        """
        return self._out_channels

    def forward(self, images: torch.Tensor):
        # Input images may be uint8 tensors in [0-255], change them to float
        # tensors in [0-1]. Get float type from backbone (could be float32/64).
        if images.dtype == torch.uint8:
            images = images.to(dtype=self.cnn.stem[0].weight.dtype)
            images /= 255.0

        # Normalize images by ImageNet color mean/std.
        images = self.normalize(images)

        # Extract c5 features from encoder (backbone) and return.
        # shape: (B, out_channels, H / 32, W / 32)
        features = self.backbone(images)["c5"]
        return features


##############################################################################
# Recurrent Neural Network                                                   #
##############################################################################
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Args:
        x: Input data for this timestep, of shape (N, D).
        prev_h: Hidden state from previous timestep, of shape (N, H)
        Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        b: Biases, of shape (H,)

    Returns a tuple of:
        next_h: Next hidden state, of shape (N, H)
        cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##########################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store next
    # hidden state and any values you need for the backward pass in the next_h
    # and cache variables respectively.
    ##########################################################################
    # Replace "pass" statement with your code
    
    # Using vanilla RNN formulae given in Lecture 16 slides 16-17
    next_h = nn.Tanh()(torch.matmul(prev_h, Wh) + torch.matmul(x, Wx) + b)

    cache = (next_h, Wh, prev_h, Wx, x)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Args:
        dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
        cache: Cache object from the forward pass

    Returns a tuple of:
        dx: Gradients of input data, of shape (N, D)
        dprev_h: Gradients of previous hidden state, of shape (N, H)
        dWx: Gradients of input-to-hidden weights, of shape (D, H)
        dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
        db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.
    #
    # HINT: For the tanh function, you can compute the local derivative in
    # terms of the output value from tanh.
    ##########################################################################
    # Replace "pass" statement with your code
    
    # Unload cache
    next_h, Wh, prev_h, Wx, x = cache

    ## Full working for dx:
    # dL/dx = dL/dnext_h * dnext_h/dx
    # dx = dnext_h * d/dx (tanh(u)), where u = Wh*prev_h + Wx*x + b
    # dx = dnext_h * d/du (tanh(u)) * du/dx
    # dx = dnext_h * (1 - tanh(u)^2) * Wx, using d/dx (tanh(x)) = 1 - tanh(x)^2
    # dx = dnext_h * (1 - tanh(u)^2) * Wx, using d/dx (tanh(x)) = 1 - tanh(x)^2
    # NOTE: tanh is applied elementwise, so gradient is distributed elementwise below
    dldu = dnext_h * (1 - next_h**2) # (N, H) = (N, H) * (N, H).
    dx = dldu @ Wx.T # (N, D) = (N, H) @ (D, H).T = (N, H) @ (H, D)

    # Similarly, for remaining derivatives
    dprev_h = dldu @ Wh.T # (N, H) = (N, H) @ (H, H)
    dWx = (dldu.T @ x).T # (D, H) = ((N, H).T @ (N, D)).T = ((H, N) @ (N, D)).T
    dWh = prev_h.T @ dldu # (H, H) = (N, H).T * (N, H) = (H, N) * (N, H)
    db = torch.sum(dldu, dim=0) # (H, ) = sum((N, H), dim=0)

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Args:
        x: Input data for the entire timeseries, of shape (N, T, D).
        h0: Initial hidden state, of shape (N, H)
        Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        b: Biases, of shape (H,)

    Returns a tuple of:
        h: Hidden states for the entire timeseries, of shape (N, T, H).
        cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##########################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of
    # input data. You should use the rnn_step_forward function that you defined
    # above. You can use a for loop to help compute the forward pass.
    ##########################################################################
    # Replace "pass" statement with your code
    
    # Get dimensions
    N, T, D = x.shape
    
    # Take the initial step
    h_t, cache_t = rnn_step_forward(x[:, 0, :], h0, Wx, Wh, b)
    
    # Keep track of the hidden states and caches
    hs = [h_t]
    cache = [cache_t]

    # For each reamining vector in the sequence
    for t in range(1, T):
        h_t, cache_t = rnn_step_forward(x[:, t, :], h_t, Wx, Wh, b) # Take a step
        hs.append(h_t)
        cache.append(cache_t)
    
    # Stack all of the hidden states into a (N, T, D) tensor
    h = torch.stack(hs, dim=1)

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Args:
        dh: Upstream gradients of all hidden states, of shape (N, T, H).

    NOTE: 'dh' contains the upstream gradients produced by the
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
        dx: Gradient of inputs, of shape (N, T, D)
        dh0: Gradient of initial hidden state, of shape (N, H)
        dWx: Gradient of input-to-hidden weights, of shape (D, H)
        dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
        db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire
    # sequence of data. You should use the rnn_step_backward function that you
    # defined above. You can use a for loop to help compute the backward pass.
    ##########################################################################
    # Replace "pass" statement with your code
    
    # Get dimensions
    N, T, H = dh.shape    

    # Take the initial step
    dx_t, dprev_h, dWx, dWh, db = rnn_step_backward(dh[:, -1, :], cache.pop(-1))

    # Keep track of the input gradients
    dxs = [dx_t]

    # For each reamining vector in the sequence, working backwards
    for t in reversed(range(T-1)):
        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dh[:, t, :]+dprev_h, cache.pop(-1))
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
        dxs.append(dx_t)
    
    dh0 = dprev_h

    # Reverse the order of the input gradients and stack into a (N, T, D) tensor
    dx = torch.stack(dxs[::-1], dim=1)

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return dx, dh0, dWx, dWh, db


class RNN(nn.Module):
    """
    Single-layer vanilla RNN module.

    You don't have to implement anything here but it is highly recommended to
    read through the code as you will implement subsequent modules.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize an RNN. Model parameters to initialize:
            Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
            Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
            b: Biases, of shape (H,)

        Args:
            input_dim: Input size, denoted as D before
            hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h0):
        """
        Args:
            x: Input data for the entire timeseries, of shape (N, T, D)
            h0: Initial hidden state, of shape (N, H)

        Returns:
            hn: The hidden state output
        """
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        """
        Args:
            x: Input data for one time step, of shape (N, D)
            prev_h: The previous hidden state, of shape (N, H)

        Returns:
            next_h: The next hidden state, of shape (N, H)
        """
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h


class WordEmbedding(nn.Module):
    """
    Simplified version of torch.nn.Embedding.

    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Args:
        x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.

    Returns a tuple of:
        out: Array of shape (N, T, D) giving word vectors for all input words.
    """

    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()

        # Register parameters
        self.W_embed = nn.Parameter(
            torch.randn(vocab_size, embed_size).div(math.sqrt(vocab_size))
        )

    def forward(self, x):

        out = None
        ######################################################################
        # TODO: Implement the forward pass for word embeddings.
        ######################################################################
        # Replace "pass" statement with your code

        # Get the rows of self.W_embed using the indices in x
        out = self.W_embed[x]

        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return out


def temporal_softmax_loss(x, y, ignore_index=None):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, *summing* the loss over all timesteps and *averaging* across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional ignore_index argument
    tells us which elements in the caption should not contribute to the loss.

    Args:
        x: Input scores, of shape (N, T, V)
        y: Ground-truth indices, of shape (N, T) where each element is in the
            range 0 <= y[i, t] < V

    Returns a tuple of:
        loss: Scalar giving loss
    """
    loss = None

    ##########################################################################
    # TODO: Implement the temporal softmax loss function.
    #
    # REQUIREMENT: This part MUST be done in one single line of code!
    #
    # HINT: Look up the function torch.functional.cross_entropy, set
    # ignore_index to the variable ignore_index (i.e., index of NULL) and
    # set reduction to either 'sum' or 'mean' (avoid using 'none' for now).
    #
    # We use a cross-entropy loss at each timestep, *summing* the loss over
    # all timesteps and *averaging* across the minibatch.
    ##########################################################################
    # Replace "pass" statement with your code
    
    # Reshape x to (N*T, V)
    # Reshape y to (N*T, )
    # Sum across all of the timesteps (i.e. reduction='sum')
    # Average across the minibatch (i.e. divide by N)
    loss = F.cross_entropy(x.reshape(-1, x.shape[-1]), y.reshape(-1), ignore_index=ignore_index, reduction='sum')/x.shape[0]

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return loss


class CaptioningRNN(nn.Module):
    """
    A CaptioningRNN produces captions from images using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.

    You will implement the `__init__` method for model initialization and
    the `forward` method first, then come back for the `sample` method later.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim: int = 512,
        wordvec_dim: int = 128,
        hidden_dim: int = 128,
        cell_type: str = "rnn",
        image_encoder_pretrained: bool = True,
        ignore_index: Optional[int] = None,
    ):
        """
        Construct a new CaptioningRNN instance.

        Args:
            word_to_idx: A dictionary giving the vocabulary. It contains V
                entries, and maps each string to a unique integer in the
                range [0, V).
            input_dim: Dimension D of input image feature vectors.
            wordvec_dim: Dimension W of word vectors.
            hidden_dim: Dimension H for the hidden state of the RNN.
            cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        """
        super().__init__()
        if cell_type not in {"rnn", "lstm", "attn"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)
        self.ignore_index = ignore_index

        ######################################################################
        # TODO: Initialize the image captioning module. Refer to the TODO
        # in the captioning_forward function on layers you need to create
        #
        # You may want to check the following pre-defined classes:
        # ImageEncoder WordEmbedding, RNN, LSTM, AttentionLSTM, nn.Linear
        #
        # (1) output projection (from RNN hidden state to vocab probability)
        # (2) feature projection (from CNN pooled feature to h0)
        ######################################################################
        # Replace "pass" statement with your code
        
        # Not sure why (1) and (2) are given in the above order.
        # I will define the feature projection first and then the output projection

        # Step 1. Feature projection & word embedding
        self.imageEncoder = ImageEncoder(pretrained=image_encoder_pretrained) # Gets the features from the image
        self.featureProjection = nn.Linear(input_dim, hidden_dim)
        self.wordEmbedding = WordEmbedding(vocab_size=vocab_size, embed_size=wordvec_dim)

        # Step 2. RNN, LSTM, or AttentionLSTM processing
        if self.cell_type == "rnn":
            self.rnn = RNN(input_dim=wordvec_dim, hidden_dim=hidden_dim)
        elif self.cell_type == "lstm":
            self.lstm = LSTM(input_dim=wordvec_dim, hidden_dim=hidden_dim)
        elif self.cell_type == "attn":
            self.attnLstm = AttentionLSTM(input_dim=wordvec_dim, hidden_dim=hidden_dim)

        # Step 3. Output projection
        self.outputProjection = nn.Linear(hidden_dim, vocab_size)

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def forward(self, images, captions):
        """
        Compute training-time loss for the RNN. We input images and the GT
        captions for those images, and use an RNN (or LSTM) to compute loss. The
        backward part will be done by torch.autograd.

        Args:
            images: Input images, of shape (N, 3, 112, 112)
            captions: Ground-truth captions; an integer array of shape (N, T + 1)
                where each element is in the range 0 <= y[i, t] < V

        Returns:
            loss: A scalar loss
        """
        # Cut captions into two pieces: captions_in has everything but the last
        # word and will be input to the RNN; captions_out has everything but the
        # first word and this is what we will expect the RNN to generate. These
        # are offset by one relative to each other because the RNN should produce
        # word (t+1) after receiving word t. The first element of captions_in
        # will be the START token, and the first element of captions_out will
        # be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        loss = 0.0
        ######################################################################
        # TODO: Implement the forward pass for the CaptioningRNN.
        # In the forward pass you will need to do the following:
        # (1) Use an affine transformation to project the image feature to
        #     the initial hidden state $h0$ (for RNN/LSTM, of shape (N, H)) or
        #     the projected CNN activation input $A$ (for Attention LSTM,
        #     of shape (N, H, 4, 4).
        # (2) Use a word embedding layer to transform the words in captions_in
        #     from indices to vectors, giving an array of shape (N, T, W).
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to
        #     process the sequence of input word vectors and produce hidden state
        #     vectors for all timesteps, producing an array of shape (N, T, H).
        # (4) Use a (temporal) affine transformation to compute scores over the
        #     vocabulary at every timestep using the hidden states, giving an
        #     array of shape (N, T, V).
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring
        #     the points where the output word is <NULL>.
        #
        # Do not worry about regularizing the weights or their gradients!
        ######################################################################
        # Replace "pass" statement with your code
        
        # Get image features
        features = self.imageEncoder(images) # (N, input_dim, feature height, feature width)

        # Get word embeddings
        embeddings = self.wordEmbedding(captions_in) # (N, T-1, wordvec_dim)

        # Step 1. Affine transformation on image feature to hidden state h0
        # For RNN and LSTM, we use the average pooled features (shape(B, C))
        if self.cell_type in ['rnn', 'lstm']:
            N, C, H, W = features.shape
            # Average pool over spatial dimensions with kernel size equal to feature map size
            features = F.avg_pool2d(features, (H, W)).squeeze() # (N, input_dim)
            # Transform image features to hidden state h0
            h0 = self.featureProjection(features) # (N, hidden_dim)
        elif self.cell_type == 'attn':
            # We pay attention over the whole image, so we don't want to average pool it
            # Instead we just pass the feature map through a linear layer to get A

            # Features has shape (N, D, 4, 4), self.featureProjection wants input dimension D and outputs H
            # Need to reshape features to (N, 4, 4, D) to give result (N, 4, 4, H), then reshape to (N, H, 4, 4)
            A = self.featureProjection(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # N, hidden_dim, 4, 4

        # RNN
        if self.cell_type == 'rnn':
            # Step 3. Run input word embeddings and initial hidden state through RNN
            hn = self.rnn(embeddings, h0) # (N, T-1, hidden_dim)
        
        # LSTM
        elif self.cell_type == 'lstm':
            # Step 3. Run input word embeddings and initial hidden state through LSTM
            hn = self.lstm(embeddings, h0) # (N, T-1, hidden_dim)

        elif self.cell_type == 'attn':
            hn = self.attnLstm(embeddings, A)
        
        # Step 4. Transform the hidden state dimension to the vocab_size
        scores = self.outputProjection(hn) # (N, T-1, vocab_size)

        # Step 5. Compute softmax loss
        loss = temporal_softmax_loss(scores, captions_out, ignore_index=self._null)

        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return loss

    def sample(self, images, max_length=15):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Args:
            images: Input images, of shape (N, 3, 112, 112)
            max_length: Maximum length T of generated captions

        Returns:
            captions: Array of shape (N, max_length) giving sampled captions,
                where each element is an integer in the range [0, V). The first
                element of captions should be the first sampled word, not the
                <START> token.
        """
        N = images.shape[0]
        captions = self._null * images.new(N, max_length).fill_(1).long()

        if self.cell_type == "attn":
            attn_weights_all = images.new(N, max_length, 4, 4).fill_(0).float()

        ######################################################################
        # TODO: Implement test-time sampling for the model. You will need to
        # initialize the hidden state of the RNN by applying the learned affine
        # transform to the image features. The first word that you feed to
        # the RNN should be the <START> token; its value is stored in the
        # variable self._start. At each timestep you will need to do to:
        # (1) Embed the previous word using the learned word embeddings
        # (2) Make an RNN step using the previous hidden state and the embedded
        #     current word to get the next hidden state.
        # (3) Apply the learned affine transformation to the next hidden state to
        #     get scores for all words in the vocabulary
        # (4) Select the word with the highest score as the next word, writing it
        #     (the word index) to the appropriate slot in the captions variable
        #
        # For simplicity, you do not need to stop generating after an <END> token
        # is sampled, but you can if you want to.
        #
        # NOTE: we are still working over minibatches in this function. Also if
        # you are using an LSTM, initialize the first cell state to zeros.
        # For AttentionLSTM, first project the 1280x4x4 CNN feature activation
        # to $A$ of shape Hx4x4. The LSTM initial hidden state and cell state
        # would both be A.mean(dim=(2, 3)).
        #######################################################################
        # Replace "pass" statement with your code
        
        # Get image features
        features = self.imageEncoder(images) # (N, input_dim, feature height, feature width)

        # For RNN and LSTM, we use the average pooled features (shape(B, C))
        if self.cell_type in ['rnn', 'lstm']:
            N, C, H, W = features.shape
            # Average pool over spatial dimensions with kernel size equal to feature map size
            features = F.avg_pool2d(features, (H, W)).squeeze() # (N, input_dim)
            # Transform image features to hidden state h0
            next_h = self.featureProjection(features) # (N, hidden_dim)
        elif self.cell_type == 'attn':
            # We pay attention over the whole image, so we don't want to average pool it
            # Instead we just pass the feature map through a linear layer to get A
            
            # Features has shape (N, D, 4, 4), self.featureProjection wants input dimension D and outputs H
            # Need to reshape features to (N, 4, 4, D) to give result (N, 4, 4, H), then reshape to (N, H, 4, 4)
            A = self.featureProjection(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # N, hidden_dim, 4, 4
            next_h = A.mean(dim=(2,3))

        # Initialise cell state if using LSTM / AttentionLSTM
        if self.cell_type == "lstm":
            next_c = torch.zeros_like(
                next_h
            )
        elif self.cell_type == "attn":
            next_c = next_h  # Initial cell state, of shape (N, H)     

        # For each timestep
        for t in range(max_length):
            # Step 1. Embed the previous word using the learned embeddings
            if t == 0:
                # First word is the <START> token. Embed and copy for each image in the minibatch
                embedding = self.wordEmbedding(torch.tensor([self._start]).repeat(N))
            else:
                embedding = self.wordEmbedding(word)

            if self.cell_type == "rnn":
                # Step 2. Make an RNN step using the previous hidden state and the embedded current word
                next_h = self.rnn.step_forward(embedding, next_h)
            
            elif self.cell_type == "lstm":
                # Step 2. Make an LSTM step using the previous hidden state, cell state, and the embedded current word
                next_h, next_c = self.lstm.step_forward(embedding, next_h, next_c)

            elif self.cell_type == "attn":
                # Step 2. Make an AttentionLSTM step using the attention matrix, previous hidden state, cell state, the embedded current word
                attn, _ = dot_product_attention(next_h, A)
                next_h, next_c = self.attnLstm.step_forward(embedding, next_h, next_c, attn) # Take a step

            # Step 3. Apply the output projection to the hidden state to get the scores
            scores = self.outputProjection(next_h)

            # Step 4. Select the word with the highest score. Record this to the captions variable
            word = torch.argmax(scores, dim=1)
            captions[:, t] = word

        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        if self.cell_type == "attn":
            return captions, attn_weights_all.cpu()
        else:
            return captions


class LSTM(nn.Module):
    """Single-layer, uni-directional LSTM module."""

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a LSTM. Model parameters to initialize:
            Wx: Weights for input-to-hidden connections, of shape (D, 4H)
            Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
            b: Biases, of shape (4H,)

        Args:
            input_dim: Input size, denoted as D before
            hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self, x: torch.Tensor, prev_h: torch.Tensor, prev_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single timestep of an LSTM.
        The input data has dimension D, the hidden state has dimension H, and
        we use a minibatch size of N.

        Args:
            x: Input data for one time step, of shape (N, D)
            prev_h: The previous hidden state, of shape (N, H)
            prev_c: The previous cell state, of shape (N, H)
            Wx: Input-to-hidden weights, of shape (D, 4H)
            Wh: Hidden-to-hidden weights, of shape (H, 4H)
            b: Biases, of shape (4H,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
                next_h: Next hidden state, of shape (N, H)
                next_c: Next cell state, of shape (N, H)
        """
        ######################################################################
        # TODO: Implement the forward pass for a single timestep of an LSTM.
        ######################################################################
        next_h, next_c = None, None
        # Replace "pass" statement with your code
        
        ## Following the instructions given in the notebook for "LSTM Update Rule"
        # Compute the activation vector
        a = (x @ self.Wx) + (prev_h @ self.Wh) + self.b

        # Divide the activation vector into the "input", "forget", "output", and "block input" inputs,
        # then compute their outputs
        H = prev_h.shape[1]
        i = torch.sigmoid(a[:, :H])
        f = torch.sigmoid(a[:, H:2*H])
        o = torch.sigmoid(a[:, 2*H:3*H])
        g = torch.tanh(a[:, 3*H:])

        # Compute the next cell and hidden states
        next_c = f*prev_c + i*g
        next_h = o*torch.tanh(next_c)
    
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return next_h, next_c

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an
        input sequence composed of T vectors, each of dimension D. The LSTM
        uses a hidden size of H, and we work over a minibatch containing N
        sequences. After running the LSTM forward, we return the hidden states
        for all timesteps.

        Note that the initial cell state is passed as input, but the initial
        cell state is set to zero. Also note that the cell state is not returned;
        it is an internal variable to the LSTM and is not accessed from outside.

        Args:
            x: Input data for the entire timeseries, of shape (N, T, D)
            h0: Initial hidden state, of shape (N, H)

        Returns:
            hn: The hidden state output.
        """

        c0 = torch.zeros_like(
            h0
        )  # we provide the intial cell state c0 here for you!
        ######################################################################
        # TODO: Implement the forward pass for an LSTM over entire timeseries
        ######################################################################
        hn = None
        # Replace "pass" statement with your code
        
        # Get dimensions
        N, T, D = x.shape

        # Take the initial step
        h_t, c_t = self.step_forward(x[:, 0, :], h0, c0)

        # Keep track of the hidden states
        hn = [h_t]

        # For each reamining vector in the sequence
        for t in range(1, T):
            h_t, c_t = self.step_forward(x[:, t, :], h_t, c_t) # Take a step
            hn.append(h_t)

        # Stack all of the hidden states into a (N, T, D) tensor
        hn = torch.stack(hn, dim=1)

        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return hn


def dot_product_attention(prev_h, A):
    """
    A simple scaled dot-product attention layer.

    Args:
        prev_h: The LSTM hidden state from previous time step, of shape (N, H)
        A: **Projected** CNN feature activation, of shape (N, H, 4, 4),
         where H is the LSTM hidden state size

    Returns:
        attn: Attention embedding output, of shape (N, H)
        attn_weights: Attention weights, of shape (N, 4, 4)

    """
    N, H, D_a, _ = A.shape

    attn, attn_weights = None, None
    ##########################################################################
    # TODO: Implement the scaled dot-product attention we described earlier. #
    # You will use this function for `AttentionLSTM` forward and sample      #
    # functions. HINT: Make sure you reshape attn_weights back to (N, 4, 4)! #
    ##########################################################################
    # Replace "pass" statement with your code
    
    ## Following the formulae given in the notebook

    attn_weights = prev_h.view(N, 1, H) @ A.view(N, H, -1)
    attn_weights = F.softmax(attn_weights.squeeze(dim=1) / (H**0.5), dim=1)

    attn = A.view(N, H, -1) @ attn_weights.view(N, -1, 1)
    attn = attn.squeeze(dim=2)

    attn_weights = attn_weights.reshape(N, D_a, -1)

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return attn, attn_weights


class AttentionLSTM(nn.Module):
    """
    This is our single-layer, uni-directional Attention module.

    Args:
        input_dim: Input size, denoted as D before
        hidden_dim: Hidden size, denoted as H before
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a LSTM. Model parameters to initialize:
            Wx: Weights for input-to-hidden connections, of shape (D, 4H)
            Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
            Wattn: Weights for attention-to-hidden connections, of shape (H, 4H)
            b: Biases, of shape (4H,)
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.Wattn = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self,
        x: torch.Tensor,
        prev_h: torch.Tensor,
        prev_c: torch.Tensor,
        attn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input data for one time step, of shape (N, D)
            prev_h: The previous hidden state, of shape (N, H)
            prev_c: The previous cell state, of shape (N, H)
            attn: The attention embedding, of shape (N, H)

        Returns:
            next_h: The next hidden state, of shape (N, H)
            next_c: The next cell state, of shape (N, H)
        """

        #######################################################################
        # TODO: Implement forward pass for a single timestep of attention LSTM.
        # Feel free to re-use some of your code from `LSTM.step_forward()`.
        #######################################################################
        next_h, next_c = None, None
        # Replace "pass" statement with your code
        
        # Compute the activation vector
        # Similar to LSTM, this time including attention
        a = (x @ self.Wx) + (prev_h @ self.Wh) + (attn @ self.Wattn) + self.b

        # The remaining code is identical to LSTM
        # Divide the activation vector into the "input", "forget", "output", and "block input" inputs,
        # then compute their outputs
        H = prev_h.shape[1]
        i = torch.sigmoid(a[:, :H])
        f = torch.sigmoid(a[:, H:2*H])
        o = torch.sigmoid(a[:, 2*H:3*H])
        g = torch.tanh(a[:, 3*H:])

        # Compute the next cell and hidden states
        next_c = f*prev_c + i*g
        next_h = o*torch.tanh(next_c)

        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return next_h, next_c

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an
        input sequence composed of T vectors, each of dimension D. The LSTM uses
        a hidden size of H, and we work over a minibatch containing N sequences.
        After running the LSTM forward, we return hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it
        is an internal variable to the LSTM and is not accessed from outside.

        h0 and c0 are same initialized as the global image feature (meanpooled A)
        For simplicity, we implement scaled dot-product attention, which means in
        Eq. 4 of the paper (https://arxiv.org/pdf/1502.03044.pdf),
        f_{att}(a_i, h_{t-1}) equals to the scaled dot product of a_i and h_{t-1}.

        Args:
            x: Input data for the entire timeseries, of shape (N, T, D)
            A: The projected CNN feature activation, of shape (N, H, 4, 4)

        Returns:
            hn: The hidden state output
        """

        # The initial hidden state h0 and cell state c0 are initialized
        # differently in AttentionLSTM from the original LSTM and hence
        # we provided them for you.
        h0 = A.mean(dim=(2, 3))  # Initial hidden state, of shape (N, H)
        c0 = h0  # Initial cell state, of shape (N, H)

        ######################################################################
        # TODO: Implement the forward pass for an LSTM over an entire time-  #
        # series. You should use the `dot_product_attention` function that   #
        # is defined outside this module.                                    #
        ######################################################################
        hn = None
        # Replace "pass" statement with your code
        
        # Get dimensions
        N, T, D = x.shape

        # Take the initial step
        attn, _ = dot_product_attention(h0, A)
        h_t, c_t = self.step_forward(x[:, 0, :], h0, c0, attn)

        # Keep track of the hidden states
        hn = [h_t]

        # For each reamining vector in the sequence
        for t in range(1, T):
            attn, _ = dot_product_attention(h_t, A)
            h_t, c_t = self.step_forward(x[:, t, :], h_t, c_t, attn) # Take a step
            hn.append(h_t)

        # Stack all of the hidden states into a (N, T, D) tensor
        hn = torch.stack(hn, dim=1)

        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return hn
