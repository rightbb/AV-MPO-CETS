import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
class StateEncoder(nn.Module):
    def __init__(self, input_state_dim, img_enc_dim):
        super(StateEncoder, self).__init__()

        self.last=nn.Sequential(
            nn.Linear(input_state_dim,512),
            nn.ReLU(True),
            nn.Linear(512,img_enc_dim)
        )

    def forward(self, x):

        # print('cdscd',len(x.shape))
        if len(x.shape)==1:
            x = x.view(-1, len(x))
        else:
            x=x.view(-1,x.shape[1])

        x = self.last(x)
        return x
