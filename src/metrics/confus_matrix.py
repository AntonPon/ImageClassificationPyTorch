from torch.nn import Module
from torch.nn import functional as F


class ConfMatrix(Module):
    def __init__(self):
        super(ConfMatrix, self).__init__()

    def forward(self, predicted, ground_truth, threshold):
        predicted = F.sigmoid(predicted) .gt(threshold)
        true_positive = predicted[ground_truth == 1].sum().item()
        false_positive = ground_truth.sum().item() - true_positive
        false_negative = predicted.sum().item() - true_positive
        return true_positive, false_positive, false_negative


