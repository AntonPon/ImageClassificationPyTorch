import torch
from tqdm import tqdm


def train(model, data_loader, epoch, optimizer, criterion, metric, board_writer=None, scheduler=None, device='cpu', threshold=0.5):
    train_loss = 0.
    train_true_positive = 0.
    train_false_negative = 0.
    train_false_positive = 0.
    scalars_dict = {'train/loss': 0, 'train/precision': 0, 'train/recall': 0, 'train/f1': 0}
    data_len = len(data_loader)
    pbar = tqdm(enumerate(data_loader), total=data_len, desc='epoch: {} train'.format(epoch))
    for idx, input_batch in pbar:
        img_batch = input_batch['image'].to(device)
        masks_batch = input_batch['labels'].to(device)

        optimizer.zero_grad()
        output_masks = model(img_batch)
        loss, true_positive, false_positive, false_negative = current_loss_metric_calc(criterion, metric, output_masks,
                                                                                       masks_batch, threshold)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_true_positive += true_positive
        train_false_positive += false_positive
        train_false_negative += false_negative

    print(train_true_positive, train_false_positive, train_false_negative)
    precision, recall, f1_score = metric_calculation(train_true_positive, train_false_positive, train_false_negative)
    if board_writer is not None:
        scalars_dict['train/loss'] = train_loss / data_len
        scalars_dict['train/precision'] = precision
        scalars_dict['train/recall'] = recall
        scalars_dict['train/f1_score'] = f1_score
        log_scalars(board_writer, scalars_dict, epoch)
        print('train epoch: {}, precision: {}, recall: {}, f1_score: {}, loss: {}'.format(epoch, precision, recall,
                                                                                          f1_score,
                                                                                          train_loss / data_len))


def val(model, criterion, metric, data_loader, epoch, board_writer, device='cpu', threshold=0.5):
    with torch.no_grad():
        val_loss = 0.
        val_true_positive = 0.
        val_false_negative = 0.
        val_false_positive = 0.

        scalars_dict = {'val/loss': 0}
        data_len = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=data_len, desc='poch: {} val'.format(epoch))
        for idx, input_batch in pbar:
            img_batch = input_batch['image'].to(device)
            masks_batch = input_batch['labels'].to(device)
            output_masks = model(img_batch)
            loss, true_positive, false_positive, false_negative = current_loss_metric_calc(criterion, metric, output_masks,
                                                                                           masks_batch, threshold)
            val_loss += loss.item()
            val_true_positive += true_positive
            val_false_positive += false_positive
            val_false_negative += false_negative
        precision, recall, f1_score = metric_calculation(val_true_positive, val_false_positive, val_false_negative)
        if board_writer is not None:
            scalars_dict['val/loss'] = val_loss / data_len
            scalars_dict['val/precision'] = precision
            scalars_dict['val/recall'] = recall
            scalars_dict['val/f1_score'] = f1_score
            log_scalars(board_writer, scalars_dict, epoch)
            print('val epoch: {}, precision: {}, recall: {}, f1_score: {}, loss: {}'.format(epoch, precision, recall,
                                                                                            f1_score,
                                                                                            val_loss / data_len))
        return precision, recall, f1_score


def log_scalars(board_writer, scalars_dict, epoch):
    for key in scalars_dict.keys():
        board_writer.add_scalar(key, scalars_dict[key], epoch)


def current_loss_metric_calc(criterion, metric, output_masks, masks_batch, threshold):
    loss = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(output_masks.shape[1]):
        loss += criterion(output_masks[:, i].reshape(-1, 1), masks_batch[:, i].reshape(-1, 1))
        results = metric(output_masks[:, i].reshape(-1, 1), masks_batch[:, i].reshape(-1, 1), threshold)
        true_positive += results[0]
        false_positive += results[1]
        false_negative += results[2]
    return loss, true_positive, false_positive, false_negative


def metric_calculation(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall / (precision + recall))
    return precision, recall, f1_score