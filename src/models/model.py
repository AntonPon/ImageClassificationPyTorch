from torchvision import models
from torch.nn import Linear, Module

class CustomResnet18(Module):
    def __init__(self, pretrained, outputs=8):
        super(CustomResnet18, self).__init__()
        self.model = models.resnet18(pretrained)
        self.model.fc = Linear(512, outputs)

    def forward(self, input):
        return self.model(input)

if __name__ == '__main__':
    print(CustomResnet18(True))
