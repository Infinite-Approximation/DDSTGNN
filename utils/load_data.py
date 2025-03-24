import os
import numpy as np
from utils.cal_adj import *

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """Load train/val/test data and get a dataloader.
            Ref code: https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
        Args:
            xs (np.array): history sequence, [num_samples, history_len, num_nodes, num_feats].
            ys (np.array):  future sequence, [num_samples, future_len, num_nodes, num_feats].
            batch_size (int): batch size
            pad_with_last_sample (bool, optional): pad with the last sample to make number of samples divisible to batch_size. Defaults to True.
            shuffle (bool, optional): shuffle dataset. Defaults to False.
        """

        self.batch_size = batch_size
        self.current_ind = 0

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        # number of batches
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        if shuffle:
            self.shuffle()

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        """Fetch a batch of data."""
        
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size *
                              (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    对数据进行归一化和反归一化
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
def load_dataset(data_dir, batch_size, model_args):
    """
    返回一个
    """
    data_dict = {}
    # read data: train_x, train_y, val_x, val_y, test_x, test_y
    # the data has been processed and stored in datasets/{dataset}/{mode}.npz
    for mode in ['train', 'val', 'test']:
        _ = np.load(os.path.join(data_dir, f"{model_args['in_seq_length']}_{model_args['out_seq_length']}_{mode}.npz"), allow_pickle=True)
        data_dict['x_' + mode] = _['x']
        data_dict['y_' + mode] = _['y'] 
    scaler = StandardScaler(mean=data_dict['x_train'][..., 0].mean(),
                            std=data_dict['x_train'][..., 0].std())  # we only see the training data.
    for mode in ['train', 'val', 'test']:
        # continue  
        data_dict['x_' + mode][..., 0] = scaler.transform(data_dict['x_' + mode][..., 0])
        data_dict['y_' + mode][..., 0] = scaler.transform(data_dict['y_' + mode][..., 0])

    data_dict['train_loader'] = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
    data_dict['val_loader'] = DataLoader(data_dict['x_val'], data_dict['y_val'], batch_size)
    data_dict['test_loader'] = DataLoader(data_dict['x_test'], data_dict['y_test'], batch_size)
    data_dict['scaler'] = scaler

    return data_dict

def load_adj(file_path):
    """
    处理邻接矩阵
    """
    adj_mx = np.load(file_path)
    adj = [transition_matrix(adj_mx).T, transition_matrix(adj_mx.T).T]
    return adj, adj_mx