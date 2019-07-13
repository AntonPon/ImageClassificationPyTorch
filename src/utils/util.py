from models.model import CustomResnet18

def get_customresnet18(pretrained, outputs=8):
    return CustomResnet18(pretrained, outputs)


def get_models_selector():
    models_selector = {'resnet18': get_customresnet18}
    return models_selector