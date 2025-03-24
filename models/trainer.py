from sklearn.metrics import mean_absolute_error
from .losses import masked_mae, masked_rmse, masked_mape, metric
from utils.train import data_reshaper, save_model
import numpy as np
import torch
import torch.optim as optim

class trainer():
    def __init__(self, scaler, model, out_seq_length, **optim_args):
        self.model  = model         # init model
        self.scaler = scaler        # data scaler
        self.output_seq_len = out_seq_length
        self.print_model_structure = optim_args['print_model']
        # training strategy parametes
        ## adam optimizer
        self.lrate  =  optim_args['lrate']
        self.wdecay = optim_args['wdecay']
        self.eps    = optim_args['eps']
        ## learning rate scheduler
        self.if_lr_scheduler    = optim_args['lr_schedule']
        self.lr_sche_steps      = optim_args['lr_sche_steps']
        self.lr_decay_ratio     = optim_args['lr_decay_ratio']
        ## curriculum learning
        self.if_cl          = optim_args['if_cl']
        self.cl_steps       = optim_args['cl_steps']
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        ## warmup
        self.warm_steps     = optim_args['warm_steps']

        # Adam optimizer
        self.optimizer      = optim.Adam(self.model.parameters(), lr=self.lrate, weight_decay=self.wdecay, eps=self.eps)
        # learning rate scheduler
        self.lr_scheduler   = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_sche_steps, gamma=self.lr_decay_ratio) if self.if_lr_scheduler else None
        
        # loss
        self.loss   = masked_mae
        self.clip   = 5             # gradient clip
    
    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        output  = self.model(input)
        output  = output.transpose(1,2)
        # curriculum learning
        # if kwargs['batch_num'] < self.warm_steps:   # warmupping
        #     self.cl_len = self.output_seq_len
        # elif kwargs['batch_num'] == self.warm_steps:
        #     # init curriculum learning
        #     self.cl_len = 1
        #     for param_group in self.optimizer.param_groups:
        #         param_group["lr"] = self.lrate
        #     print("======== Start curriculum learning... reset the learning rate to {0}. ========".format(self.lrate))
        # else:
        #     # begin curriculum learning
        #     if (kwargs['batch_num'] - self.warm_steps) % self.cl_steps == 0 and self.cl_len <= self.output_seq_len:
        #         self.cl_len += int(self.if_cl)
        # # scale data and calculate loss
        ## inverse transform for both predict and real value.
        predict     = self.scaler.inverse_transform(output)
        real_val    = self.scaler.inverse_transform(real_val[:,:,:,0])
        ## loss
        mae_loss    = self.loss(predict[:, :self.cl_len, :], real_val[:, :self.cl_len, :], 0)
        loss = mae_loss
        loss.backward()

        # gradient clip and optimization
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        # metrics
        mape = masked_mape(predict, real_val, 0.0)
        rmse = masked_rmse(predict, real_val, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()
    
    def eval(self, device, dataloader, **kwargs):
        # val a epoch
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        self.model.eval()
        for itera, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device)
            # for dstgnn
            output = self.model(testx)
            output = output.transpose(1, 2)

            # scale data
            predict = self.scaler.inverse_transform(output)
            real_val = self.scaler.inverse_transform(testy[:,:,:,0])
            
            # metrics
            loss = self.loss(predict, real_val, 0.0).item()
            mape = masked_mape(predict,real_val,0.0).item()
            rmse = masked_rmse(predict,real_val,0.0).item()

            print("test: {0}".format(loss), end='\r')

            valid_loss.append(loss)
            valid_mape.append(mape)
            valid_rmse.append(rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        return mvalid_loss,mvalid_mape,mvalid_rmse

    @staticmethod
    def test(model, device, dataloader, scaler, **kwargs):
        # test
        model.eval()
        outputs = []
        realy   = torch.Tensor(dataloader['y_test']).to(device)
        realy   = realy.transpose(1, 2)
        y_list  = []
        for itera, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device).transpose(1, 2)

            with torch.no_grad():
                preds   = model(testx)

            outputs.append(preds)
            y_list.append(testy)
        yhat    = torch.cat(outputs,dim=0)[:realy.size(0),...]
        y_list  = torch.cat(y_list, dim=0)[:realy.size(0),...]

        assert torch.where(y_list == realy)

        # scale data
        realy   = scaler.inverse_transform(realy)[:, :, :, 0]
        yhat    = scaler.inverse_transform(yhat)
        
        # 保存第一个节点在测试集上的预测值
        pred_data_path = "visualization/data/DDSTGNN"
        # np.save(f"{pred_data_path}/{kwargs['dataset_name']}_{kwargs['out_seq_length']}step_pred.npy", yhat.cpu().numpy())
        # np.save(f"{pred_data_path}/{kwargs['dataset_name']}_{kwargs['out_seq_length']}step_.npy", realy.cpu().numpy())
        # summarize the results.
        amae    = []
        amape   = []
        armse   = []
        # 先计算单步的误差
        for i in range(kwargs['out_seq_length']):
            # For horizon i, only calculate the metrics **at that time** slice here.
            pred    = yhat[:,:,i]
            real    = realy[:,:,i]
            metrics = metric(pred,real)
            # log     = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            # print(log.format(i+1, metrics[0], metrics[2], metrics[1]))
            amae.append(metrics[0])     # mae
            amape.append(metrics[1])    # mape
            armse.append(metrics[2])    # rmse
        # 求解每一步及之前的平均误差
        amae = np.cumsum(amae)
        amse = np.cumsum([i ** 2 for i in armse])
        num_arr = np.arange(kwargs['out_seq_length']) + 1
        amae = amae / num_arr
        amse = amse / num_arr
        for i in range(kwargs['out_seq_length']):
            print(f"Evaluate best model on test data for horizon {i + 1}, Test MSE: {amse[i]:.4f}, Test MAE: {amae[i]:.4f}")
        # print(f"(On average over {kwargs['out_seq_length']} horizons) Test MAE: {np.mean(amae):.2f} | Test RMSE: {np.mean(armse):.2f} | Test MAPE: {np.mean(amape) * 100:.2f}% |")
