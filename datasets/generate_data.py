import argparse
import os
import pandas as pd
import numpy as np
import yaml


def generate_graph_seq2seq_io_data(
    df, x_offsets, y_offsets, add_day_of_week=True, add_month_of_year=True
):
    # 这里默认将day_of_week, month_of_year:这两个特征加入进去，和原先的sst构成了三个特征
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]
    # 原先的索引是 RangeIndex(start=0, stop=4018, step=1)，需要变换一下，将索引设置为第一列，也就是时间
    # df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    # df = df.set_index(df.columns[0])
    if add_day_of_week:
        # 把每天转化为是一周的第几天
        day_of_week = df.index.dayofweek
        day_of_week_tiled = np.tile(day_of_week, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(day_of_week_tiled)
    if add_month_of_year:
        # 把每个月转化为是一年的第几个月
        month_of_year = df.index.month - 1
        month_of_year_tiled = np.tile(month_of_year, [1, num_nodes, 1]).transpose(
            (2, 1, 0)
        )
        feature_list.append(month_of_year_tiled)
    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    # 这里将csv的数据读取，保存为dataframe
    df = pd.read_csv(args.sst_df_filename, index_col=0, parse_dates=True)
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, output_length, num_nodes, output_dim) -- (3995, 12, 136, 3)
    # y: (num_samples, output_length, num_nodes, output_dim) -- (3995, 12, 136, 3)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train : num_train + num_val],
        y[num_train : num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{seq_length_x}_{seq_length_y}_{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def cal_adj_entry(point1, point2):
    point1_x, point1_y = float(point1[0]), float(point1[1])
    point2_x, point2_y = float(point2[0]), float(point2[1])
    dis = np.sqrt((point1_x - point2_x) ** 2 + (point1_y - point2_y) ** 2)
    if dis > np.sqrt(2):
        return 0
    elif dis == 0:
        # 我们认为自身到自身的距离为 0.1
        return 1 / 0.1
    else:
        return 1 / dis


def cal_dis(point1, point2):
    point1_x, point1_y = float(point1[0]), float(point1[1])
    point2_x, point2_y = float(point2[0]), float(point2[1])
    dis = np.sqrt((point1_x - point2_x) ** 2 + (point1_y - point2_y) ** 2)
    return dis


def generate_adj(args):
    """
    邻接矩阵的构造：
    根据两点之间的距离的倒数来决定邻接矩阵的值
    """
    df = pd.read_csv(args.sst_df_filename, index_col=0, parse_dates=True)
    headers = df.columns
    # 提取经纬度
    lat_lon = [col.split(":") for col in headers]
    num_nodes = len(lat_lon)
    adj = np.zeros((num_nodes, num_nodes))
    # 第一种计算邻接矩阵的方法，也就是对距离取倒数
    # for i in range(num_nodes):
    #     for j in range(num_nodes):
    #         adj[i, j] = cal_adj_entry(lat_lon[i], lat_lon[j])

    # 第二种方法是：exp(-d / sigma^2)
    for i in range(num_nodes):
        for j in range(num_nodes):
            adj[i, j] = cal_dis(lat_lon[i], lat_lon[j])
    sigma = np.std(adj.flatten())
    for i in range(num_nodes):
        for j in range(num_nodes):
            res = np.exp(-adj[i][j] ** 2 / sigma**2)
            if res < 0.1:
                adj[i][j] = 0
            else:
                adj[i][j] = res

    np.save(os.path.join(args.output_dir, "adj.npy"), adj)
    print(f"邻接矩阵处理完成，保存目录为{os.path.join(args.output_dir, 'adj.npy')}")


if __name__ == "__main__":
    dataset = "Bo_Hai"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=f"{dataset}", help="Dataset name.")
    parser.add_argument(
        "--output_dir", type=str, default=f"datasets/{dataset}", help="Output directory."
    )
    parser.add_argument(
        "--sst_df_filename", type=str, default=f"datasets/raw_data/{dataset}.csv", help="Raw SST"
    )
    args = parser.parse_args()
    config_path = "configs/" + args.dataset + ".yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # 用过去的 in_seq_length 个时间窗，来预测未来的 out_seq_length 个时间窗户
    seq_length_x = config["model_args"]["in_seq_length"]
    y_start = 1
    seq_length_y = config["model_args"]["out_seq_length"]
    args.seq_length_x = seq_length_x
    args.y_start = y_start
    args.seq_length_y = seq_length_y
    print(f"用过去 {seq_length_x} 个时间窗来预测未来 {seq_length_y} 个时间窗")
    generate_train_val_test(args)
    generate_adj(args)
