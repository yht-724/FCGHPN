import argparse
from datetime import datetime
import numpy as np
import torch
from model.FCGHPN import FCGHPN
import torch.optim as optim
from intensity_hawkes import cal_intensityAdj, hawkes_train
from libs.loss_functions import OD_loss, cal_acc, RMSE_MAE_MAPE_MSE, In_Out_loss
from time import time
from tqdm import tqdm
import os
from libs.utils import print_log,EarlyStopMonitor
from libs.prepare_data import generate_data, load_data
import dgl
from libs.process_data import processNeighbors
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 40]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.1, help='Initial learning rate [default: 0.001]')
parser.add_argument('--optimizer', default='adamW', help='adam or momentum [default: adam]')
parser.add_argument('--seed', type=int, default=50010, help='', required=False)
parser.add_argument('--decay', type=float, default=0.90, help='decay rate of learning rate [0.97/0.92]')
parser.add_argument('--gcn_hid_feats', type=int, default=16, help='Dimension of GCN hidden feature')
parser.add_argument('--gcn_output_feats', type=int, default=4, help='Dimension of GCN output feature')
parser.add_argument('--att_input_feats', type=int, default=8, help='Dimension of attention input feature')
parser.add_argument('--att_output_feats', type=int, default=32, help='Dimension of attention output feature')
parser.add_argument('--att_heads_num', type=int, default=8, help='nums of attention head')
parser.add_argument('--wdecay', type=float, default=0.001, help='decay rates')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--data_dir', type=str, default='./data/nyc_taxi_2019_01', help='.......')
parser.add_argument('--num_nodes', type=int, default=66, help='nums of node,When dataset is Chicago,the values is 77')
parser.add_argument('--event_flag', type=int, default=1, help='event_flag')
parser.add_argument('--t2vDim', type=int, default=5, help='Dimension of time2vec feature')
parser.add_argument('--hawkes_epochs', type=int, default=200, help='Max epochs of training')
parser.add_argument('--type_size', type=int, default=2, help='event type size')
parser.add_argument('--hawkes_lr', type=int, default=0.01, help='learning rate of hawkes process train')
parser.add_argument('--earlyStop', type=int, default=20, help='Stop training after how many epoches are not optimized')
parser.add_argument('--save_dir', type=str, default='./data/nyc_taxi_2019_01/gcnn_parameters', help='parameters save load')
parser.add_argument('--plot', type=bool, default=True, help='Whether or not to draw')
parser.add_argument('--beta1', type=float, default=0.6, help='The proportion of OD loss')
parser.add_argument('--beta2', type=float, default=0.2, help='The proportion of in loss')
parser.add_argument('--beta3', type=float, default=0.2, help='The proportion of out loss')

FLAGS = parser.parse_args()

timestamp_start = datetime.now()
print("\nWorking start at ", timestamp_start, '\n')

def eval_model(net, val_loader, loss_function, device):
    net.eval()
    batch_loss_list = []
    batach_odACC_list = []

    for input_data, target_data, adj, intensity_score, t2v, target_od_matrix, target_in_degree, target_out_degree in tqdm(
            val_loader):
        input_data = input_data.to(device)  # (4,66,1)
        target_data = target_data.to(device)  # (4,66,1)
        target_od_matrix = target_od_matrix.to(device)  # (4,66,66)
        target_in_degree = target_in_degree.to(device)
        target_out_degree = target_out_degree.to(device)
        net.train()
        optimizer.zero_grad()

        finally_embedding, predict_od_matrix, in_degree, out_degree = net(device, adj, intensity_score,
                                                                          input_data, t2v,
                                                                          neighbors_list)  # (4,66,66)

        loss_od = loss_function(predict_od_matrix, target_od_matrix)
        acc_od = cal_acc(predict_od_matrix, target_od_matrix)
        loss_in, loss_out = In_Out_loss(in_degree, target_in_degree), In_Out_loss(out_degree,
                                                                                  target_out_degree)
        loss_od = FLAGS.beta1 * loss_od + FLAGS.beta2 * loss_in + FLAGS.beta3 * loss_out

        batch_loss_list.append(loss_od.item())
        batach_odACC_list.append(acc_od.item())

    epoch_loss = np.mean(batch_loss_list)
    epoch_ACC = np.mean(batach_odACC_list)

    return epoch_loss, epoch_ACC

def predict(net, data_loader, device):
    net.eval()
    output = []
    y = []

    for input_data, target_data, adj, intensity_score, t2v, target_od_matrix, target_in_degree, target_out_degree in tqdm(
            data_loader):
        input_data = input_data.to(device)  # (4,66,1)
        target_data = target_data.to(device)  # (4,66,1)
        target_od_matrix = target_od_matrix.to(device)  # (4,66,66)
        target_in_degree = target_in_degree.to(device)
        target_out_degree = target_out_degree.to(device)
        net.train()
        optimizer.zero_grad()

        finally_embedding, predict_od_matrix, in_degree, out_degree = net(device, adj, intensity_score,
                                                                          input_data, t2v,
                                                                          neighbors_list)  # (4,66,66)

        out_batch = predict_od_matrix.detach().cpu().numpy()
        y_batch = target_od_matrix.detach().cpu().numpy()
        output.append(out_batch)
        y.append(y_batch)

    output = np.vstack(output).squeeze()
    y = np.vstack(y).squeeze()

    return y, output

def tst_model(net, test_loader, device, log):
    net.eval()
    print_log("--------- Test ---------", log=log)

    start = time()
    y_true, y_pred = predict(net, test_loader, device)
    end = time()

    # rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    # out_str = "All Steps MAE = %.5f, RMSE = %.5f, MAPE = %.5f\n" % (
    #     mae_all,
    #     rmse_all,
    #     mape_all,
    # )
    rmse, mae, mape, mse = RMSE_MAE_MAPE_MSE(y_true, y_pred)
    out_str = " MAE = %.5f, MSE = %.5f, RMSE = %.5f, MAPE = %.5f\n" % (
        mae,
        mse,
        rmse,
        mape
    )

    print_log(out_str, log=log)
    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == "__main__":

    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    dgl.seed(FLAGS.seed)

    data_dir = FLAGS.data_dir
    save_dir = FLAGS.save_dir
    num_nodes = FLAGS.num_nodes
    event_flag = FLAGS.event_flag
    device = FLAGS.device
    batch_size = FLAGS.batch_size
    earlystop = FLAGS.earlyStop
    plot = FLAGS.plot

    t2v_feats = FLAGS.t2vDim


    hawkesEpoch = FLAGS.hawkes_epochs
    typeSize = FLAGS.type_size
    hawkesLr = FLAGS.hawkes_lr


    model_name = "gcnn"
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"./logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}--{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    phy_adj, od_matrxi_list, trafficFlow, t2v, event_list, distance, newDiagA_list = generate_data(data_dir, num_nodes, t2v_feats, event_flag, log)

    parameter_path = os.path.join(data_dir, 'hawkes_parameters/epoch_%s' % hawkesEpoch)
    if not os.path.exists(parameter_path):
        print_log('\nstart train hawkes model', log=log)
        start_time_train = time()
        hawkes_train(device, num_nodes, hawkesEpoch, event_list, newDiagA_list, trafficFlow, parameter_path)
        end_time_train = time()
        print_log('\nhawkes model train finished, time is %.2fs' % (end_time_train - start_time_train), log=log)
    else:
        print_log('\nhawkes model train finished,next step is load parameters', log=log)

    generate_A, intensity_score_list = cal_intensityAdj(device, num_nodes, event_list, newDiagA_list, trafficFlow, parameter_path)

    train_loader, val_loader, test_loader = load_data(data_dir, generate_A, intensity_score_list, od_matrxi_list, trafficFlow, t2v, batch_size, log)

    neighbors_list = processNeighbors(phy_adj)

    net = FCGHPN(trafficFlow.shape[-1], FLAGS.gcn_hid_feats, FLAGS.gcn_output_feats, FLAGS.att_input_feats,
                    FLAGS.att_output_feats, FLAGS.att_heads_num, FLAGS.num_nodes)
    net.to(device)

    loss_function = OD_loss()
    optimizer = optim.AdamW(net.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.wdecay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, FLAGS.decay)
    # lookhead = Lookahead(optimizer=optimizer)
    early_stopper = EarlyStopMonitor(max_round=FLAGS.patience, higher_better=False)  # 10
    #early_stopper = EarlyStopMonitor(max_round=FLAGS.patience, higher_better=False)  # 10

    wait = 0
    min_eval_loss = np.inf

    train_loss_list = []
    train_ACC_list = []

    eval_loss_list = []
    eval_ACC_list = []

    for epoch in range(FLAGS.max_epoch):

        start_time_train = time()
        batch_loss_list = []
        batach_odACC_list = []

        for input_data, target_data, adj, intensity_score, t2v, target_od_matrix, target_in_degree, target_out_degree in tqdm(
                train_loader):
            input_data = input_data.to(FLAGS.device)  # (4,66,1)
            target_data = target_data.to(FLAGS.device)  # (4,66,1)
            target_od_matrix = target_od_matrix.to(FLAGS.device)  # (4,66,66)
            target_in_degree = target_in_degree.to(FLAGS.device)
            target_out_degree = target_out_degree.to(FLAGS.device)
            net.train()
            optimizer.zero_grad()

            finally_embedding, predict_od_matrix, in_degree, out_degree = net(FLAGS.device, adj, intensity_score,
                                                                              input_data, t2v,
                                                                              neighbors_list)  # (4,66,66)

            loss_od = loss_function(predict_od_matrix, target_od_matrix)
            acc_od = cal_acc(predict_od_matrix, target_od_matrix)
            loss_in, loss_out = In_Out_loss(in_degree, target_in_degree), In_Out_loss(out_degree,
                                                                                              target_out_degree)
            loss_od = FLAGS.beta1 * loss_od + FLAGS.beta2 * loss_in + FLAGS.beta3 * loss_out

            loss_od.backward()
            optimizer.step()

            batch_loss_list.append(loss_od.item())
            batach_odACC_list.append(acc_od.item())
        scheduler.step()

        train_loss = np.mean(batch_loss_list)
        train_ACC = np.mean(batach_odACC_list)
        train_loss_list.append(train_loss)
        train_ACC_list.append(train_ACC)

        eval_loss, eval_ACC = eval_model(net, val_loader, loss_function, device)
        eval_loss_list.append(eval_loss)
        eval_ACC_list.append(eval_ACC)

        print_log(datetime.now(), "Epoch", epoch + 1, " \tTrain Loss = %.5f" % train_loss, "Val Loss = %.5f" % eval_loss, log=log)
        rmse, mae, mape, mse = RMSE_MAE_MAPE_MSE(*predict(net, test_loader, device))
        print('\t current epoch: %s \t' % epoch, end='')
        print('MAE: %.2f \t' % (mae), 'MSE: %.2f\t' % (mse), 'RMSE: %.2f\t' % (rmse),'MAPE: %.2f' % (mape))


        if eval_loss < min_eval_loss:
            wait = 0
            min_eval_loss = eval_loss
            best_epoch = epoch
            best_state_dict = net.state_dict()
        else:
            wait += 1
            if wait >= earlystop:
                break


    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # save_path = os.path.join(save_dir, f"{model_name}-{now}.pt")
    # torch.save(best_state_dict, save_path)
    #
    # # best_state_dict = './data/nyc_taxi_2019_01/gcnn_parameters/gcnn-2024-01-23-12-01-53.pt'
    # # epoch = 10
    # # best_epoch = 10
    # net.load_state_dict(best_state_dict)
    # #net.load_state_dict(torch.load(best_state_dict))
    # train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(net, train_loader, device))
    # val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(net, val_loader, device))
    #
    # out_str = f"Early stopping at epoch: {epoch + 1}\n"
    # out_str += f"Best at epoch {best_epoch + 1}:\n"
    # out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    # out_str += "Train ACC = %.5f\n" % train_ACC_list[best_epoch]
    # out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
    #     train_rmse,
    #     train_mae,
    #     train_mape,
    # )
    # out_str += "Val Loss = %.5f\n" % eval_loss_list[best_epoch]
    # out_str += "Val ACC = %.5f\n" % eval_ACC_list[best_epoch]
    # out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
    #     val_rmse,
    #     val_mae,
    #     val_mape,
    # )
    #
    # print_log(out_str, log=log)

    if plot:
        plt.subplot(1, 2, 1)
        plt.plot(range(0, epoch + 1), train_loss_list, 'b', label="Train Loss")
        plt.plot(range(0, epoch + 1), eval_loss_list, 'g--', label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Epoch-Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(0, epoch + 1), train_ACC_list, 'b', label="Train ACC")
        plt.plot(range(0, epoch + 1), eval_ACC_list, 'g--', label="Val ACC")
        plt.xlabel("Epoch")
        plt.ylabel("ACC")
        plt.title("Epoch-ACC")
        plt.legend()
        plt.show()

    tst_model(net, test_loader, device, log)






