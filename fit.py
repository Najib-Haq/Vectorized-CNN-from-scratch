import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

from model.Optimizer import SGD
from model.Loss import CrossEntropyLoss
from model.Metrics import accuracy, macro_f1
from model.LRScheduler import ReduceLROnPlateau
from utils import one_hot_encoding, AverageMeter, mixup, update_loggings, visualize_training, update_wandb

def train_one_epoch(model, train_loader, config, optimizer):
    celoss = CrossEntropyLoss()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    f1_meter = AverageMeter('f1')

    model.train()
    for step, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, labels = data[0], data[1]

        if np.random.rand() < config['augment']['mixup']:
            images, one_hot_labels = mixup(images, labels, config['num_class'])
        else:
            one_hot_labels = one_hot_encoding(labels, config['num_class'])
        
        out = model(images)
        loss = celoss(out, one_hot_labels)
        model.backward(celoss.get_grad_wrt_softmax(out, one_hot_labels))
        optimizer.step(model) # optimizer step

        out = np.argmax(out, axis=1)
        acc = accuracy(labels, out)
        macf1 = macro_f1(labels, out)

        loss_meter.update(loss)
        acc_meter.update(acc)
        f1_meter.update(macf1)

    return model, loss_meter.avg, acc_meter.avg, f1_meter.avg

def validate_one_epoch(model, val_loader, config):
    celoss = CrossEntropyLoss()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    f1_meter = AverageMeter('f1')

    model.eval()
    for step, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        images, labels = data[0], data[1]
        one_hot_labels = one_hot_encoding(labels, config['num_class'])
        
        out = model(images)
        loss = celoss(out, one_hot_labels)

        out = np.argmax(out, axis=1)
        acc = accuracy(labels, out)
        macf1 = macro_f1(labels, out)

        loss_meter.update(loss)
        acc_meter.update(acc)
        f1_meter.update(macf1)

    return loss_meter.avg, acc_meter.avg, f1_meter.avg


def fit_model(model, train_loader, val_loader, config, wandb_run):
    # save based on macro f1
    best_macro_f1 = 0

    loggings = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    
    if config['resume']:
        with open(config['checkpoint_path'], "rb") as f:
            prev_run = pickle.load(f)
            epoch = prev_run['epoch']
            lr = prev_run['lr']

    optimizer = SGD(lr=lr if config['resume'] else config['lr'])
    scheduler = ReduceLROnPlateau(factor=config['lr_scheduler']['factor'], patience=config['lr_scheduler']['patience'], verbose=1)


    start_epoch = epoch if config['resume'] else 0
    for epoch in range(start_epoch, config['epochs']):
        model, train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, config, optimizer)
        val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, config)

        print(f"Epoch {epoch+1}/{config['epochs']} => LR {optimizer.lr}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

        scheduler.step(val_f1, optimizer) # reduce lr based on validation f1 performance

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            model.save_model(f"{config['output_dir']}/best_model_E{epoch}.npy", epoch, config['wandb_id'], optimizer.lr)

        loggings = update_loggings(loggings, epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1)
        if config['use_wandb']: update_wandb(epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, optimizer.lr)
    
    loggings = pd.DataFrame(loggings)
    loggings.to_csv(f"{config['output_dir']}/logs.csv", index=False)
    
    if config['use_wandb']:
        wandb_run.summary[f"Best VAL MacroF1"] = best_macro_f1
        wandb_run.summary[f"Best VAL Accuracy"] = loggings['val_acc'].max()
        wandb_run.summary[f"Best VAL Loss"] = loggings['val_loss'].min()
        wandb_run.finish()

    visualize_training(loggings, config['output_dir'])

       



        




