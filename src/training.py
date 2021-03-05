from tqdm import tqdm
import torch


def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    n_samples, train_loss = 0, 0

    for batch in tqdm(loader, total=len(loader), desc='training...'):
        input = batch[0].to(device)
        target = batch[1].to(device)

        output = model(input)
        loss = loss_fn(output, target, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = target.size(0)
        n_samples += batch_size
        train_loss += loss.item() * batch_size
    
    return train_loss / n_samples


def validate(model, loader, loss_fn, device):
    model.eval()
    n_samples, val_loss, corrects = 0, 0, 0
    
    for batch in tqdm(loader, total=len(loader), desc='validation...'):
        input = batch[0].to(device)
        target = batch[1].to(device)

        with torch.no_grad():
            output = model(input)
        
        loss = loss_fn(output, target, reduction='mean')
        
        batch_size = target.size(0)
        n_samples += batch_size
        val_loss += loss.item() * batch_size
        corrects += (torch.max(output, 1)[1] == target).sum().item()
    
    accuracy = corrects / n_samples
    
    return val_loss / n_samples, accuracy


def train(model, train_loader, val_loader, loss_fn, optimizer, epochs, device, logger, 
          model_save_path=None, scheduler=None, start_epoch=1, best_val_acc=0, verbose=True):
    model.to(device)

    logger.info('Start training with params:')
    logger.info(f'loss_fn: {loss_fn.__module__}.{loss_fn.__name__}')
    logger.info(f'optimizer: {optimizer.__class__.__name__}({optimizer.defaults})')
    logger.info(f'scheduler: {scheduler.__class__.__name__ + str(scheduler.state_dict()) if scheduler else None}')
    logger.info(f'train dataloader: batch_size={train_loader.batch_size}\ntransform={train_loader.dataset.transform}')
    logger.info(f'val dataloader: batch_size={val_loader.batch_size}\ntransform={val_loader.dataset.transform}')
    logger.info(f'device: {device.type}')
    logger.info(f'model save path: {model_save_path}')

    for epoch in range(start_epoch, start_epoch + epochs):
        logger.info('Starting epoch {}/{}. lr: {:.8}'.format(epoch, epochs, 
                                                              optimizer.param_groups[0]['lr']))

        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)

        logger.info('Epoch finished. Train loss: {:.5f}'.format(train_loss))

        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        logger.info('Val loss: {:.5f}, val accuracy: {:.5f} (best: {:.5f})'\
                    .format(val_loss, val_acc, max(val_acc, best_val_acc)))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if model_save_path is not None:
                with open(model_save_path, 'wb') as fp:
                    torch.save({'model_state_dict': model.state_dict(),
                                #'optimizer_state_dict': optimizer.state_dict(),
                                'epochs': epoch,
                                'accuracy': best_val_acc}, fp)
        #if verbose:        
        #    print('Epoch #{}. lr: {:.8}, train loss: {:.5f}, val loss: {:.5f}, val accuracy: {:.5f} (best: {:.5f})'.\
        #          format(epoch, optimizer.param_groups[0]['lr'], train_loss, val_loss, val_acc, best_val_acc))
