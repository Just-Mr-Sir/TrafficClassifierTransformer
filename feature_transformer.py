import torch
import torch.nn as nn

class FeatureTransformer(nn.Module):
    def __init__(self, ):
        super(FeatureTransformer, self).__init__()
        pass
        #embed tokens into vectors
        #transformer block

        #head dim = hidden dim / num heads

        # img_pixel x img_pixel
        # patch_pixel x patch_pixel
        #img_pixel / patch_pixel = patches
        # flatten matrix data of patch into a vector (or us convolutional layer)

        # what is a token?
        # patches? rows (of matrix)?
        # position should be with respect to time

        # for variable length tokens -> pad



    def forward(self, x):

        return x