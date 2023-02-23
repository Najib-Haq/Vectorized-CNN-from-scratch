import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


def get_near_duplicate_removed_train():
    # https://www.kaggle.com/code/nexh98/cse-472-offline4-make-dataset/data?scriptVersionId=118126349
    # This is the dataset with near duplicate images removed using CNN embeddings
    df1 = pd.read_csv("resources/nd_removed_train_a.csv")
    df3 = pd.read_csv("resources/nd_removed_train_c.csv")

    df1 = df1[df1.included == True].reset_index(drop=True)
    df3 = df3[df3.included == True].reset_index(drop=True)
    return df1, df3

def split_dataset(parent_dir="NumtaDB_with_aug", validation_percentage=0.2):
    df1 = pd.read_csv(f"{parent_dir}/training-a.csv")
    df2 = pd.read_csv(f"{parent_dir}/training-b.csv")
    df3 = pd.read_csv(f"{parent_dir}/training-c.csv")

    # df1, df3 = get_near_duplicate_removed_train() 
    df = pd.concat([df1, df2, df3], ignore_index=True)
    df['img_path'] = df['database name'] + '/' + df['filename']
    # df = df2

    df['split_col'] = df['database name original'] + '_' + df['digit'].astype(str)
    df = df.sample(frac=1) # shuffle

    split_col = df['split_col'].unique().tolist()
    train_indexes = []
    for cat in split_col:
        indexes = df[df['split_col'] == cat].index.tolist()
        train_indexes.extend(indexes[:int(len(indexes) * (1 - validation_percentage))])
    train_df = df.loc[train_indexes]
    val_df = df.drop(train_indexes)
    print("Train: ", train_df.shape, "; Valid: ", val_df.shape)

    # save csv
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    return train_df, val_df


def one_hot_encoding(y, num_class):
    bs = y.shape[0]
    label = np.zeros((bs, num_class))
    label[np.arange(bs), y] = 1
    return label


def mixup(images, labels, num_classes=10):
    changed_indices = np.random.permutation(images.shape[0])
    # alpha beta values from https://github.com/ultralytics/yolov5/issues/3380
    lam = np.random.beta(8.0, 8.0) 

    changed_images = images[changed_indices]
    changed_labels = labels[changed_indices]

    labels = one_hot_encoding(labels, num_classes)
    changed_labels = one_hot_encoding(changed_labels, num_classes)

    images = lam * images + (1 - lam) * changed_images
    labels = lam * labels + (1 - lam) * changed_labels
    return images, labels


def set_seed(seed):
    np.random.seed(seed)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name):
        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def update_loggings(loggings, epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1):
    loggings['epoch'].append(epoch)
    loggings['train_loss'].append(train_loss)
    loggings['train_acc'].append(train_acc)
    loggings['train_f1'].append(train_f1)
    loggings['val_loss'].append(val_loss)
    loggings['val_acc'].append(val_acc)
    loggings['val_f1'].append(val_f1)
    return loggings


def visualize_training(loggings, save_dir):
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    ax[0].plot(loggings['epoch'], loggings['train_loss'], label='train')
    ax[0].plot(loggings['epoch'], loggings['val_loss'], label='val')
    ax[0].set_title('Loss')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(loggings['epoch'], loggings['train_acc'], label='train')
    ax[1].plot(loggings['epoch'], loggings['val_acc'], label='val')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(loggings['epoch'], loggings['train_f1'], label='train')
    ax[2].plot(loggings['epoch'], loggings['val_f1'], label='val')
    ax[2].set_title('F1 Score')
    ax[2].legend()
    ax[2].grid()

    plt.legend()

    plt.savefig(f'{save_dir}/metrics.png', bbox_inches='tight')
    # plt.show()

# wandb stuff
try:
    import wandb
except:
    print("please install wandb if you want to use wandb loggings")

def wandb_init(config):
    if config['wandb']['entity'] == 'anonymous':
        print("Anonymouse run wandb")
        wandb.login(anonymous="must", relogin=True)
        run = wandb.init(anonymous="allow")
    else:
        if config['resume']:
            with open(config['checkpoint_path'], "rb") as f:
                wandb_id = pickle.load(f)['wandb_id']

        run = wandb.init(
            project=config['wandb']['project'], 
            entity=config['wandb']['entity'],
            name=config['name'],
            config=config, 
            resume="allow",
            id=wandb_id if config['resume'] else None
        )
    return run


def update_wandb(epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, lr):
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_f1': train_f1,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1,
        'lr': lr
    })


