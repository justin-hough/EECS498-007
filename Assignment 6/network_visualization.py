"""
Implements a network visualization in PyTorch.
Make sure to write device-agnostic code. For any function, initialize new tensors
on the same device as input tensors
"""

import torch


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from network_visualization.py!")


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    # Hint: X.grad.data stores the gradients                                     #
    ##############################################################################
    # Replace "pass" statement with your code

    N, C, H, W = X.shape

    # Forward pass through the model
    y_hat = model(X)

    # Calculate CE loss
    p = torch.softmax(y_hat, dim=1)
    log_likelihood = torch.log(p[range(N), y])
    loss = -torch.sum(log_likelihood, dim=0)

    # Backwards pass w.r.t loss
    loss.backward()

    # Take absolute value of the gradient of the image, then max over the channels
    X_grad = X.grad.detach()
    saliency = torch.max(torch.abs(X_grad), dim=1).values

    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return saliency


def make_adversarial_attack(X, target_y, model, max_iter=100, verbose=True):
    """
    Generate an adversarial attack that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN
    - max_iter: Upper bound on number of iteration to perform
    - verbose: If True, it prints the pogress (you can use this flag for debugging)

    Returns:
    - X_adv: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our adversarial attack to the input image, and make it require
    # gradient
    X_adv = X.clone()
    X_adv = X_adv.requires_grad_()

    learning_rate = 1
    ##############################################################################
    # TODO: Generate an adversarial attack X_adv that the model will classify    #
    # as the class target_y. You should perform gradient ascent on the score     #
    # of the target class, stopping when the model is fooled.                    #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate an adversarial     #
    # attack in fewer than 100 iterations of gradient ascent.                    #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    # Replace "pass" statement with your code

    # Training loop
    for epoch in range(max_iter):
        # Forward pass
        y_hat = model(X_adv)

        # Calculate CE loss
        p = torch.softmax(y_hat, dim=1)
        loss = torch.log(p[0, target_y])

        # Backwards pass w.r.t loss
        loss.backward()

        # Gradient ascent
        # Get gradient of the image
        dX = X_adv.grad.detach()

        # Normalise the gradient using L2-norm
        dX = dX / (dX**2).sum().sqrt()

        # Gradient ascent on the image
        X_adv.data += learning_rate * dX

        # Zero the gradient as to not accumulate it over iterations
        X_adv.grad = torch.zeros_like(X_adv)

        print(
            "Iteration %d: target score %.3f, max score %.3f"
            % (epoch, y_hat[0, target_y], y_hat.max()),
        )

    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_adv


def class_visualization_step(img, target_y, model, **kwargs):
    """
    Performs gradient step update to generate an image that maximizes the
    score of target_y under a pretrained model.

    Inputs:
    - img: random image with jittering as a PyTorch tensor
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    """

    l2_reg = kwargs.pop("l2_reg", 1e-3)
    learning_rate = kwargs.pop("learning_rate", 25)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    # Hint: You have to perform inplace operations on img.data to update   #
    # the generated image using gradient ascent & reset img.grad to zero   #
    # after each step.                                                     #
    ########################################################################
    # Replace "pass" statement with your code

    # Forward pass
    y_hat = model(img)

    # Score on target_y
    score = y_hat[:, target_y]

    # Calculate loss as the difference between the score and the
    # regularisation term (squared L2-norm of the image)
    loss = score - l2_reg * (img**2).sum()

    # Backwards pass w.r.t loss
    loss.backward()

    # Get gradient on image
    dX = img.grad.detach()

    # Gradient ascent on image
    img.data += learning_rate * dX

    # Zero the gradient as to not accumulate it over iterations
    img.grad = torch.zeros_like(img)

    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img
