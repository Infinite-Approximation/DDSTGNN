import argparse
import time
import yaml
import torch
import wandb
from utils.load_data import *
from models.model import D2STGNN
from models import trainer
from utils.train import *

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # wandb.init(project="SST_Prediction")
    # 初始化种子，保证实验结果可复现
    set_random_seed(2025)        
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Bo_Hai', help='Dataset name.')
    # parser.add_argument('--dataset', type=str, default='Nan_Hai', help='Dataset name.')
    args = parser.parse_args()
    config_path = "configs/" + args.dataset + ".yaml"
    # 读取config配置文件
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_dir = config['data_args']['data_dir']
    dataset_name = config['data_args']['data_dir'].split("/")[-1]
    device = torch.device(config['start_up']['device'])
    save_path = 'output/' + config['start_up']['model_name'] + "_" + dataset_name + ".pt"  # the best model
    # 加载数据
    ## 加载每个时间每个节点的特征值
    t1 = time.time()
    batch_size = config['model_args']['batch_size']
    dataloader = load_dataset(data_dir, batch_size, config['model_args'])
    t2 = time.time()
    print("Load dataset: {:.2f}s...".format(t2 - t1))
    scaler = dataloader['scaler']
    ## 加载邻接矩阵
    t1 = time.time()
    adj_mx, adj_ori = load_adj(config['data_args']['adj_data_path'])
    t2 = time.time()
    print("Load adjacent matrix: {:.2f}s...".format(t2 - t1))
    # 加载超参数
    model_args = config['model_args']
    model_args['device'] = device
    model_args['num_nodes'] = adj_mx[0].shape[0]
    model_args['adjs'] = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori'] = torch.tensor(adj_ori).to(device)
    model_args['dataset'] = dataset_name
    # training strategy parametes
    optim_args = config['optim_args']
    optim_args['cl_steps'] = optim_args['cl_epochs'] * len(dataloader['train_loader'])
    optim_args['warm_steps'] = optim_args['warm_epochs'] * len(dataloader['train_loader'])
    # init the model
    model = D2STGNN(**model_args).to(device)
    # get a trainer
    engine = trainer(scaler, model, model_args['out_seq_length'], **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)
    # begin training:
    train_time = []  # training time
    val_time = []  # validate time
    print("Whole trainining iteration is " + str(len(dataloader['train_loader'])))
    # training init: resume model & load parameters
    mode = config['start_up']['mode']
    assert mode in ['test', 'scratch']
    resume_epoch = 0
    if mode == 'test':
        model = load_model(model, save_path)  # resume best
    else:
        resume_epoch = 0
    batch_num = resume_epoch * len(dataloader['train_loader'])  # batch number (maybe used in schedule sampling)
    # 开始训练
    if mode == 'test':
        engine.test(model, device, dataloader, scaler, loss=engine.loss, dataset_name=dataset_name, 
                    out_seq_length=model_args['out_seq_length'])
        return
    for epoch in range(resume_epoch + 1, optim_args['epochs']):
        # ================= Train ===============================================
        time_train_start = time.time()
        current_learning_rate = engine.lr_scheduler.get_last_lr()[0]
        train_loss = []
        train_mape = []
        train_rmse = []
        dataloader['train_loader'].shuffle()  # traing data shuffle when starting a new epoch.
        for itera, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = data_reshaper(x, device)
            trainy = data_reshaper(y, device)
            mae, mape, rmse = engine.train(trainx, trainy, batch_num=batch_num)
            print("{0}: {1}".format(itera, mae), end='\r')
            train_loss.append(mae)
            train_mape.append(mape)
            train_rmse.append(rmse)
            batch_num += 1
        time_train_end = time.time()
        train_time.append(time_train_end - time_train_start)
        current_learning_rate = engine.optimizer.param_groups[0]['lr']
        if engine.if_lr_scheduler:
            engine.lr_scheduler.step()
                    # record history loss
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        # =============== Validation =============================================
        time_val_start = time.time()
        mvalid_loss, mvalid_mape, mvalid_rmse, = engine.eval(device, dataloader)
        time_val_end = time.time()
        val_time.append(time_val_end - time_val_start)
        curr_time = str(time.strftime("%d-%H-%M", time.localtime()))
        log = 'Current Time: ' + curr_time + ' | Epoch: {:03d} | Train_Loss: {:.4f} | Train_MAPE: {:.4f} | Train_RMSE: {:.4f} | Valid_Loss: {:.4f} | Valid_RMSE: {:.4f} | Valid_MAPE: {:.4f} | LR: {:.6f}'
        print(log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_rmse, mvalid_mape,
                             current_learning_rate))
        early_stopping(mvalid_loss, engine.model)
        if early_stopping.early_stop:
            print('Early stopping!')
            break
        # wandb.log({"mtrain_loss": mtrain_loss, "mvalid_loss": mvalid_loss, "Time(each epoch)": time_train_end - time_train_start}, step=epoch)
        # ================================= Test =============================== #
        # engine.test(model, device, dataloader, scaler, loss=engine.loss, dataset_name=dataset_name, out_seq_length=model_args['out_seq_length'])
        torch.cuda.empty_cache()
    return

if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end = time.time()
    print("Total time spent: {0}".format(t_end - t_start))
