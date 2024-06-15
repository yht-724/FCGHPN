import numpy as np
import torch

def MAPE(y_true, y_pred, null_val=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        y_true[y_true < 1] = 0
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.where(y_true == 0,1e-8,y_true)
    return np.mean(np.abs((y_pred-y_true) / y_true))*100

def mean_squared_error_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def MAE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae

def MSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mse = np.square(y_pred - y_true)
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)
        return mse


def RMSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse

class OD_loss(torch.nn.Module):
    def __init__(self):
        super(OD_loss, self).__init__()
        self.pro = torch.nn.ReLU()#对应论文中的负值被替换成0

    def forward(self, predict, truth):
        # predict = predict.detach().cpu().numpy()
        # truth = truth.detach().cpu().numpy()

        mask = (truth < 1)
        mask2 = (predict > 0)
        loss = torch.mean(((predict - truth) ** 2) * ~mask + ((mask2 * predict - truth) ** 2) * mask)
        return loss
        # mask表示真实值是0的情况
        #((predict - truth) ** 2) * ~mask:表示在true>=1的时候，按照(y-y^)^2求loss
        #((mask2 * predict - truth) ** 2) * mask:表示在真实值是0的时候，只关注预测值>0的时候



def In_Out_loss(prediction, target):
    loss = torch.mean(target - prediction) ** 2
    return loss

def cal_acc(predict,truth):
    if isinstance(predict,torch.Tensor):
        count_true = np.sum(predict.cpu().detach().numpy() == truth.cpu().detach().numpy())
        total = predict.cpu().detach().numpy().size
    else:
        count_true = np.sum(predict == truth)
        total = predict.size
    acc = count_true / total
    return acc

def RMSE_MAE_MAPE_MSE(y_true, y_pred):
    return (
        root_mean_squared_error(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
        mean_absolute_percentage_error(y_true, y_pred),
        mean_squared_error_error(y_true, y_pred)
        # RMSE(y_true, y_pred),
        # MAE(y_true, y_pred),
        # MAPE(y_true, y_pred),
        # MSE(y_true, y_pred)
    )
