from torch import nn
import torch

class Flagger(nn.Module):
    def __init__(self):
        super(Flagger,self).__init__()      
        layer1 = nn.Linear(22,15)
        layer2 = nn.Linear(15,3)
        with torch.no_grad():
            nn.init.xavier_normal_(layer1.weight)
            nn.init.xavier_normal_(layer2.weight)
            layer1.bias.fill_(0)
            layer2.bias.fill_(0)

        self.layer_stack = nn.Sequential(
            layer1,
            nn.Tanh(),
            layer2,
            nn.Tanh(),
        )
    
    def forward(self,x):
        self.softmax = nn.Softmax(dim = 1)
        val = self.layer_stack(x)
        val = self.softmax(val)
        return val



