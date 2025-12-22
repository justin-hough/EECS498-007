"""
Implements a style transfer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import torch.nn as nn
from a6_helper import *

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from style_transfer.py!')

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    ############################################################################
    # TODO: Compute the content loss for style transfer.                       #
    ############################################################################
    # Replace "pass" statement with your code
    
    # Calculate content loss using equation given in notebook
    loss = content_weight * torch.sum((content_current - content_original)**2)

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return loss

def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    gram = None
    ############################################################################
    # TODO: Compute the Gram matrix from features.                             #
    # Don't forget to implement for both normalized and non-normalized version #
    ############################################################################
    # Replace "pass" statement with your code
    
    # Flatten features spatial dimensions
    flat_features = torch.flatten(features, start_dim=2)

    # Compute the Gram matrix using the equation given in the notebook
    gram = torch.bmm(flat_features, torch.transpose(flat_features, -1, -2))

    # Normalise Gram matrix if specified
    if normalize:
       gram /= features.shape[1:].numel() # [1:].numel() gives C * H * W

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the style loss at a set of layers.                        #
    # Hint: you can do this with one for loop over the style layers, and       #
    # should not be very much code (~5 lines).                                 #
    # You will need to use your gram_matrix function.                          #
    ############################################################################
    # Replace "pass" statement with your code

    # Initialise style loss    
    loss = 0

    # Accumulate the style loss at each layer index specified in style_layers
    for i, layer_idx in enumerate(style_layers):
       current_gram = gram_matrix(feats[layer_idx])
       loss += style_weights[i] * torch.sum((current_gram - style_targets[i])**2)

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return loss

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    ############################################################################
    # TODO: Compute total variation loss.                                      #
    # Your implementation should be vectorized and not require any loops!      #
    ############################################################################
    # Replace "pass" statement with your code

    # Compute the total variation loss
    loss = torch.sum((img[:, :, :-1, :] - img[:, :, 1:, :])**2) # Vertical differences
    loss += torch.sum((img[:, :, :, :-1] - img[:, :, :, 1:])**2) # Horizontal differences
    loss *= tv_weight

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return loss

def guided_gram_matrix(features, masks, normalize=True):
  """
  Inputs:
    - features: PyTorch Tensor of shape (N, R, C, H, W) giving features for
      a batch of N images.
    - masks: PyTorch Tensor of shape (N, R, H, W)
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, R, C, C) giving the
      (optionally normalized) guided Gram matrices for the N input images.
  """
  guided_gram = None
  ##############################################################################
  # TODO: Compute the guided Gram matrix from features.                        #
  # Apply the regional guidance mask to its corresponding feature and          #
  # calculate the Gram Matrix. You are allowed to use one for-loop in          #
  # this problem.                                                              #
  ##############################################################################
  # Replace "pass" statement with your code

  # Get dimensions
  N, R, C, H, W = features.shape

  # Flatten features and masks spatial dimensions
  flat_features = torch.flatten(features, start_dim=3) # (N, R, C, H*W)
  flat_masks = torch.flatten(masks, start_dim=2)[:, :, None, :] # (N, R, 1, H*W)

  # Get the spatially guided feature maps
  spatial_guided_feats = flat_features * flat_masks

  # Combine the batch and R dimensions
  spatial_guided_feats = torch.reshape(spatial_guided_feats, (N*R, C, H*W))

  # Compute the guided Gram matrix
  guided_gram = torch.bmm(spatial_guided_feats, torch.transpose(spatial_guided_feats, -1, -2))

  # Normalise Gram matrix if specified
  if normalize:
      guided_gram /= spatial_guided_feats.shape[1:].numel() # [1:].numel() gives C * H * W

  # Reshape from (N*R, C, C) back to (N, R, C, C)
  guided_gram = guided_gram.reshape(N, R, C, C)

  ''' The above code doesn't require a for-loop, but I wrote a for-loop version 
  below in case the above code misbehaves for some reason.

  # Get dimensions
  N, R, C, H, W = features.shape

  # Flatten features and masks spatial dimensions
  flat_features = torch.flatten(features, start_dim=3) # (N, R, C, H*W)
  flat_masks = torch.flatten(masks, start_dim=2)[:, :, None, :] # (N, R, 1, H*W)

  # Get the spatially guided feature maps
  spatial_guided_feats = flat_features * flat_masks

  guided_gram = []
  for r_idx in range(R):
     guided_gram.append(
        torch.bmm(spatial_guided_feats[:, r_idx, :], torch.transpose(spatial_guided_feats[:, r_idx, :], -1, -2))
     )

  guided_gram = torch.stack(guided_gram).permute((1, 0, 2, 3))

  # Normalise Gram matrix if specified
  if normalize:
      guided_gram /= spatial_guided_feats.shape[2:].numel() # [2:].numel() gives C * (H * W)
  '''

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return guided_gram

def guided_style_loss(feats, style_layers, style_targets, style_weights, content_masks):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the guided Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
    - content_masks: List of the same length as feats, giving a binary mask to the
      features of each layer.
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the guided style loss at a set of layers.                 #
    ############################################################################
    # Replace "pass" statement with your code
    
    # Initialise guided style loss    
    loss = 0

    # Accumulate the style loss at each layer index specified in style_layers
    for i, layer_idx in enumerate(style_layers):
       current_gram = guided_gram_matrix(feats[layer_idx], content_masks[layer_idx])
       loss += style_weights[i] * torch.sum((current_gram - style_targets[i])**2)

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return loss