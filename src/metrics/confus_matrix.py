from torch.nn import Module
from torch import sigmoid


class ConfMatrix(Module):
    def __init__(self):
        super(ConfMatrix, self).__init__()

    def forward(self, predicted, ground_truth, threshold):
        #predicted = F.sigmoid(predicted).gt(threshold)
        #print(predicre)
        predicted = sigmoid(predicted).gt(threshold)
        #print('predicted', predicted)
        true_positive = predicted[ground_truth == 1].sum().item()
        #print('true_positive', true_positive)
        false_positive = ground_truth.sum().item() - true_positive
        #print('false_positive', false_positive)
        false_negative = predicted.sum().item() - true_positive
        return true_positive, false_positive, false_negative


