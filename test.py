import numpy as np
import pandas as pd
import os
from glob import glob
import sys
import yaml
from tqdm import tqdm

from dataset.dataset import Dataset, DataLoader, check_test_dataset
from model.Model import Model
from model.Metrics import macro_f1, accuracy, confusion_matrix_sk

def parse_arguments():
    # sys arg should have following format:
    # python train.py path_to_folder [Optional]path_to_gt_csv
    sys_args = sys.argv
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['data_dir'] = sys_args[1]
    # config['gt_csv'] = sys_args[2] if len(sys_args) > 2 else False
    return config


def path_to_csv(path_dir):
    # path_dir should be like "data/train"
    # return a dataframe with columns: "path", "digit"
    path_list = []
    for ext in ['.png', '.jpg', '.jpeg']:
        path_list.extend(glob(f'{path_dir}/*{ext}'))

    # path_list = glob(os.path.join(path_dir, "*.png"))
    print(f"Found a total of {len(path_list)} files in path {path_dir}")
    df = pd.DataFrame(path_list, columns=["img_full_path"])
    df['img_path'] = df['img_full_path'].apply(lambda x: os.path.basename(x))
    return df


def infer(model, test_loader):
    model.eval()
    preds = []
    all_paths = []
    for step, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        images, paths = data[0], data[1]
        out = model(images)
        out = np.argmax(out, axis=1)
        preds.append(out)
        all_paths += paths
    preds = np.concatenate(preds)

    df = pd.DataFrame({'FileName': all_paths, 'Digit': preds})
    return df


def show_results(pred_df, gt_df):
    # align pred_df and gt_df by FileName
    gt_df['FileName'] = gt_df['filename']
    gt_df['GT'] = gt_df['digit']
    pred_df['Pred'] = pred_df['Digit']
    df = pd.merge(pred_df, gt_df, on='FileName')

    # calculate metrics
    acc = accuracy(df['GT'], df['Pred'])
    macro_f1_score = macro_f1(df['GT'], df['Pred'])
    # cm_score = confusion_matrix_sk(df['GT'], df['Pred'])

    print(f"Accuracy: {acc}")
    print(f"Macro F1: {macro_f1_score}")
    # print(f"Confusion Matrix: {cm_score}")


if __name__ == "__main__":
    config = parse_arguments()
    df = path_to_csv(config['data_dir'])
    if config['debug']: df = df.sample(frac=1).reset_index(drop=True) #[:100]
    os.makedirs(config['output_dir'], exist_ok=True)

    # make and load model
    model = Model(config)
    print(model)
    if config['checkpoint_path']:
        model.load_model(config['checkpoint_path'])

    # make dataset
    test_dataset = Dataset(config['data_dir'], df, label_col=None, mode='test', config=config['augment'])
    check_test_dataset(test_dataset, save_dir=config['output_dir'])

    test_loader = DataLoader(test_dataset, batch_size=config['valid_batch'], shuffle=False)

    prediction_df = infer(model, test_loader)
    prediction_df.to_csv(os.path.join(config['output_dir'], 'prediction.csv'), index=False)

    if config['gt_csv']:
        gt_df = pd.read_csv(config['gt_csv'])
        show_results(prediction_df, gt_df)
    
    print("Done!")


    



    
