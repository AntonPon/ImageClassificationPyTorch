import torch
from tqdm import tqdm


def train(model, data_loader, epoch, optimizer, criterion, metric, board_writer=None, scheduler=None, device='cpu'):
    train_loss = 0.
    train_miou = 0.
    scalars_dict = {'train/loss': 0, 'train/miou': 0}
    data_len = len(data_loader)
    pbar = tqdm(enumerate(data_loader), total=data_len, desc='epoch: {} train'.format(epoch))
    for idx, input_batch in pbar:
        print(input_batch['image'].to().type())
        img_batch = input_batch['image'].type(torch.FloatTensor).to(device)
        masks_batch = input_batch['labels'].to(device)

        optimizer.zero_grad()
        output_masks = model(img_batch)
        loss = criterion(masks_batch, output_masks)

        loss.backward()
        optimizer.step()

        #train_loss += loss.item()
        print(loss)
        '''
        train_miou += metric(masks_batch, output_masks).item()
    if board_writer is not None:
        scalars_dict['train/loss'] = train_loss / data_len
        scalars_dict['train/miou'] = train_miou / data_len
        log_scalars(board_writer, scalars_dict, epoch)
        '''

def val(model, criterion, metric, data_loader, epoch, board_writer, device='cpu'):
    with torch.no_grad():
        val_loss = 0.
        val_miou = 0.
        scalars_dict = {'val/loss': 0, 'val/miou': 0}
        data_len = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=data_len, desc='poch: {} val'.format(epoch))
        for idx, input_batch in pbar:
            img_batch = input_batch['imgs'].to(device)
            masks_batch = input_batch['masks'].to(device)

            output_masks = model(img_batch)

            loss = criterion(masks_batch, output_masks).item()
            val_loss += loss
            miou = metric(masks_batch, output_masks).item()
            val_miou += miou
        if board_writer is not None:
            scalars_dict['val/loss'] = val_loss/data_len
            scalars_dict['val/miou'] = val_miou/data_len
            log_scalars(board_writer, scalars_dict, epoch)


def log_scalars(board_writer, scalars_dict, epoch):
    for key in scalars_dict.keys():
        board_writer.add_scalar(key, scalars_dict[key], epoch)
