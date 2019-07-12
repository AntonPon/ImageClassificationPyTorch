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
        loss = 0
        for i in range(output_masks.shape[1]):
            loss += criterion(masks_batch[:, i].reshape(-1 , 1), output_masks[:, i].reshape(-1, 1))
            results = metric(masks_batch[:, i].reshape(-1 , 1), output_masks[:, i].reshape(-1, 1), threshold)
            train_true_positive += results[0]
            train_false_positive += results[1]
            train_false_negative += results[2]

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    precision = train_true_positive / (train_true_positive + train_false_positive)
    recall = train_true_positive / (train_true_positive + train_false_negative)
    f1_score = 2 * (precision * recall / (precision + recall))
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
            img_batch = input_batch['imgs'].to(device)
            masks_batch = input_batch['masks'].to(device)

            output_masks = model(img_batch)

            loss = 0
            for i in range(output_masks.shape[1]):
                loss += criterion(masks_batch[:, i].reshape(-1, 1), output_masks[:, i].reshape(-1, 1))
                results = metric(masks_batch[:, i].reshape(-1, 1), output_masks[:, i].reshape(-1, 1), threshold)
                val_true_positive += results[0]
                val_false_positive += results[1]
                val_false_negative += results[2]

            val_loss += loss.item()
        precision = val_true_positive / (val_true_positive + val_false_positive)
        recall = val_true_positive / (val_true_positive + val_false_negative)
        f1_score = 2 * (precision * recall / (precision + recall))
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
