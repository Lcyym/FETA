import os
import time
import socket
import pickle
import argparse
import asyncio
import concurrent.futures
import threading
import math
import copy
import numpy as np
import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from pulp import *
import random
from config import ClientConfig, CommonConfig
from client_comm_utils import *
from training_utils import train, test
import datasets, models, models_grow
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="127.0.0.1",
                    help='IP address for controller or ps')
parser.add_argument('--master_port', type=int, default=58000, metavar='N',
                    help='')
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()

if args.visible_cuda == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str((int(args.idx)) % 4 + 0)
    # if args.idx == '0':
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '2'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda
# if int(args.idx) == 0:
#     os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

def main():
    client_config = ClientConfig(
        common_config=CommonConfig()
    )
    # recorder = SummaryWriter("log_"+str(args.idx))
    # receive config
    master_socket = connect_get_socket(args.master_ip, args.master_port)
    config_received = get_data_socket(master_socket)
    #这里跟服务器通信然后获取配置文件，get_data_socket是堵塞的。
    for k, v in config_received.__dict__.items():
        setattr(client_config, k, v)

    computation = client_config.custom["computation"]
    dynamics=client_config.custom["dynamics"]

    common_config = CommonConfig()
    common_config.model_type = client_config.common_config.model_type
    common_config.dataset_type = client_config.common_config.dataset_type
    common_config.batch_size = client_config.common_config.batch_size
    common_config.data_pattern=client_config.common_config.data_pattern
    common_config.lr = client_config.common_config.lr
    common_config.decay_rate = client_config.common_config.decay_rate
    common_config.min_lr=client_config.common_config.min_lr
    common_config.epoch = client_config.common_config.epoch
    common_config.momentum = client_config.common_config.momentum
    common_config.weight_decay = client_config.common_config.weight_decay
    

    # init config
    print(common_config.__dict__)

    local_model = models_grow.get_model()
    # local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    #local_model.load_state_dict(client_config.para)
    torch.nn.utils.vector_to_parameters(client_config.para, local_model.parameters())
    local_model.to(device)
    #para_nums = torch.nn.utils.parameters_to_vector(local_model.parameters()).nelement()
    # create dataset
    print(len(client_config.custom["label_data_idxes"]))
    print(len(client_config.custom["unlabel_data_idxes"]))
    label_selected_idxs=client_config.custom["label_data_idxes"]
    unlabel_selected_idxs=client_config.custom["unlabel_data_idxes"]
    train_dataset, test_dataset = datasets.load_datasets(common_config.dataset_type)
    label_train_loader = datasets.create_dataloaders(train_dataset, batch_size=common_config.batch_size, shuffle=True, selected_idxs=label_selected_idxs)
    unlabel_train_loader = datasets.create_dataloaders(train_dataset, batch_size=32, shuffle=False, selected_idxs=unlabel_selected_idxs)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=16, shuffle=False)

    count_dataset_total(label_train_loader)

    epoch_lr = common_config.lr
    local_steps=25
    sample_thd=0.2
    tf_batch_size=32
    for epoch in range(1, 1+common_config.epoch):
        
        # if epoch%100==0:
        #     def update_model_construct(old_model,epoch_idx):
        #         new_model = models_grow.get_model(int(epoch/5)).to(device)
        #         # new_model = get_model(int(epoch_idx/5))
        #         old_model_state = old_model.state_dict()
        #         for key in old_model_state.keys():
        #             if key.find("avgpool") !=-1:
        #                 continue
        #             new_model.state_dict()[key] = new_model.state_dict()[key]*0.0 + old_model_state[key]
    
        #         return new_model

        #     local_model = update_model_construct(local_model,epoch)
        # local_steps,compre_ratio=get_data_socket(master_socket)
        get_model_para(local_model,master_socket)

        print("***")
        start_time = time.time()
        if common_config.momentum<0:
            optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=common_config.weight_decay)
        else:
            optimizer = optim.SGD(local_model.parameters(),momentum=common_config.momentum, lr=epoch_lr, weight_decay=common_config.weight_decay)
        train_loss = train(local_model, label_train_loader, optimizer, local_iters=local_steps, device=device, model_type=common_config.model_type)

        prediction_results, t_acc, label_time = get_results_val(unlabel_train_loader, local_model, epoch=epoch)
        avg_labels, selected_indices = select_unlabelled_data(prediction_results, sample_thd)

        if epoch > 1 and epoch % 1 == 0:
            epoch_lr = max((common_config.decay_rate * epoch_lr, common_config.min_lr))
        print("epoch-{} lr: {}".format(epoch, epoch_lr))
        print("local steps: ", local_steps)
        train_time=0.1
        if len(selected_indices) >= tf_batch_size:
            start_time = time.time()
            # global_para_backup = torch.nn.utils.parameters_to_vector(local_model.parameters()).clone().detach()
            optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=common_config.weight_decay)
            local_model, tf_train_loss, total_num, right_num, train_acc = train_on_pseduo(local_model, train_dataset, np.array(unlabel_selected_idxs), selected_indices, avg_labels, optimizer, tf_batch_size)
            train_time=time.time()-start_time
            print("pseduo time: {}".format(train_time))

        # train_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))
        # while train_time>10 or train_time<1:
        #      train_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))
        # train_time=train_time/10
        # print("train time: ", train_time)
        
        #print(train_time/computation)
        acc,test_loss=0,0
        test_loss, acc = test(local_model, test_loader, device, model_type=common_config.model_type)
        print("after aggregation, epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(epoch, train_loss, test_loss, acc))
        print("send para")

        send_model_para(local_model,master_socket)

        # send_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))
        # while send_time>10 or send_time<1:
        #      send_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))

        # print("send time: ",send_time)
        send_data_socket((train_time,label_time), master_socket)

    master_socket.shutdown(2)
    master_socket.close()

def get_results_val(data_loader, label_model, device=torch.device("cuda"),epoch=0):
    setResults = list()
    label_model.eval()
    data_loader = data_loader.loader

    correct = 0
    start_time = time.time()
    with torch.no_grad():
        for data_idx, (data, target) in enumerate(data_loader):
            #print(data_idx)
            # if data_idx >= ((epoch-1)*400)%4000 and data_idx<=(epoch*400)%4000:
            if data_idx >= 0 and data_idx<=4000 :
                data, target = data.to(device), target.to(device)
                output = label_model(data)
                softmax1 = F.softmax(output, dim=1).cpu().detach().numpy()
                avg_pred = softmax1.copy()
                # print("avg_pred")
                # print(avg_pred)
                # print(type(avg_pred))
                setResults.extend(softmax1.copy())
            else:
                avg_pred=np.array([[0,0,0,0,0,0,0,0,0,0]])
                setResults.extend(avg_pred)
            pred = avg_pred.argmax(1)
            correct = correct + np.sum(pred == target.cpu().detach().numpy().reshape(pred.shape))
    compute_time = time.time() - start_time

    t_acc = correct/len(data_loader.dataset)
    print("teachers' acc in val: {}".format(t_acc))
    print("compute label time: {}".format(compute_time))
    return np.array(setResults), t_acc, compute_time

def select_unlabelled_data(prediction_results, p_th):
    avg_labels = prediction_results
    max_label = np.argmax(avg_labels, axis=1)

    selected_indices = list()
    print("prediction_results len: {}".format(len(prediction_results)))
    for data_idx in range(len(prediction_results)):
        if avg_labels[data_idx][max_label[data_idx]] >= p_th:
            selected_indices.append(data_idx)

    print("num of selected samples: ", len(selected_indices))
    return np.array(avg_labels)[selected_indices], np.array(selected_indices)

def train_on_pseduo(model, train_dataset, unlabelled_indices, selected_indices, soft_labels, optimizer, tf_batch_size, device=torch.device("cuda")):
    if len(selected_indices) <= 0:
        return
    model.train()
    model = model.to(device)
    selected_shuffle = [i for i in range(len(selected_indices))]
    np.random.shuffle(selected_shuffle)
    selected_indices = selected_indices[selected_shuffle]
    soft_labels = soft_labels[selected_shuffle]

    train_loader = datasets.create_dataloaders(train_dataset, batch_size=tf_batch_size, selected_idxs=unlabelled_indices[selected_indices].tolist(), shuffle=False)
    # total_num, right_num = count_dataset(train_loader, soft_labels, tf_batch_size)
    total_num, right_num = 0, 0

    data_loader = train_loader.loader
    samples_num = 0

    train_loss = 0.0
    correct = 0

    correct_s = 0
    
    for data_idx, (data, label) in enumerate(data_loader):

        target = torch.from_numpy(soft_labels[data_idx*tf_batch_size:(data_idx+1)*tf_batch_size])
        data, target, label = data.to(device), target.to(device), label.to(device)
        
        output = model(data)

        optimizer.zero_grad()
        
        loss_func = nn.CrossEntropyLoss() 
        loss =loss_func(output, target.argmax(1))
        # loss = F.cross_entropy(output, target.argmax(1))
        # loss = F.cross_entropy(output, label)

        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * data.size(0))
        samples_num += data.size(0)

        pred = target.argmax(1, keepdim=True)
        batch_correct = pred.eq(label.view_as(pred)).sum().item()
        correct += batch_correct

        pred_s = output.argmax(1, keepdim=True)
        batch_correct_s = pred.eq(pred_s).sum().item()
        correct_s += batch_correct_s

    if samples_num != 0:
        train_loss /= samples_num
        # print("sample num: ", samples_num)
        test_accuracy = np.float(1.0 * correct / samples_num)
        print("teacher's acc : {}".format(test_accuracy))
        test_accuracy = np.float(1.0 * correct_s / samples_num)
        print("student's training acc : {}".format(test_accuracy))
    
    return model,train_loss, total_num, right_num, test_accuracy

def count_dataset(loader, soft_labels, tf_batch_size):
    counts = np.zeros(len(loader.loader.dataset.classes))
    right = np.zeros(len(loader.loader.dataset.classes))

    st_matrix = np.zeros((len(loader.loader.dataset.classes), len(loader.loader.dataset.classes)))
    for data_idx, (_, target) in enumerate(loader.loader):
        predu = torch.from_numpy(soft_labels[data_idx*tf_batch_size:(data_idx+1)*tf_batch_size])
        predu = predu.argmax(1)
        batch_correct = predu.eq(target.view_as(predu))

        labels = target.view(-1).numpy()
        predu = predu.view(-1).numpy()
        for label_idx, label in enumerate(labels):
            counts[label] += 1
            st_matrix[label][predu[label_idx]] += 1
            if batch_correct[label_idx] == True:
                right[label] += 1
    print(st_matrix.astype(np.int))
    print("class counts: ", counts.astype(np.int))
    print("total data count: ", np.sum(counts))
    print("right class counts: ", right.astype(np.int))
    print("total right data count: ", np.sum(right))

    return np.sum(counts), np.sum(right)


def send_model_dict(local_model,master_socket):
    model_dict = dict()
    for para in local_model.state_dict().keys():
        model_dict[para] = copy.deepcopy(local_model.state_dict()[para])
    
    start_time = time.time()
    send_data_socket(model_dict, master_socket)
    send_time=time.time()-start_time
    pass

def get_model_dict(local_model,master_socket):
    local_para = get_data_socket(master_socket)
    local_model.load_state_dict(local_para)
    local_model.to(device)

def send_model_para(local_model,master_socket):
    local_paras = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
    #print(local_paras)
    send_data_socket(local_paras, master_socket)

def send_compressed_model(local_model,master_socket,ratio):
    local_paras = torch.nn.utils.parameters_to_vector(local_model.parameters()).nelement()
    compress_paras=compress_model_top(local_paras, ratio)
    send_data_socket(compress_paras, master_socket)

def send_compressed_gradient(local_model,master_socket,ratio,old_para,memory_para):
    local_paras = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
    memory_para,compress_paras=compress_gradient_top(local_paras, old_para, memory_para,ratio)
    send_data_socket(compress_paras, master_socket)
    return memory_para

def get_model_para(local_model,master_socket):
    local_para = get_data_socket(master_socket)
    local_para.to(device)
    torch.nn.utils.vector_to_parameters(local_para, local_model.parameters())
    return local_para

def compress_model_rand(local_para, ratio):
    start_time = time.time()
    with torch.no_grad():
        send_para = local_para.detach()
        select_n = int(send_para.nelement() * ratio)
        rd_seed = np.random.randint(0, np.iinfo(np.uint32).max)
        rng = np.random.RandomState(rd_seed)
        indices = rng.choice(send_para.nelement(), size=select_n, replace=False)
        select_para = send_para[indices]
    print("compressed time: ", time.time() - start_time)
    return (select_para, select_n, rd_seed)

def compress_model_top(local_para, ratio):
    start_time = time.time()
    with torch.no_grad():
        send_para = local_para.detach()
        topk = int(send_para.nelement() * ratio)
        _, indices = torch.topk(local_para.abs(), topk, largest=True, sorted=False)
        select_para = send_para[indices]
    print("compressed time: ", time.time() - start_time)

    return (select_para, indices)

def compress_gradient_top(local_para, old_para, memory_para,ratio):
    start_time = time.time()
    with torch.no_grad():
        # print("local_para:",local_para)
        # print("old_para:",old_para)
        old_para=local_para-old_para+memory_para
        # print("local_para:",local_para)
        send_para = old_para.detach()
        topk = int(send_para.nelement() * ratio)
        _, indices = torch.topk(old_para.abs(), topk, largest=True, sorted=False)
        select_para = send_para[indices]
    print("compressed time: ", time.time() - start_time)
    restored_model = torch.zeros(send_para.nelement()).to(device)
    restored_model[indices] = select_para
    memory_para=old_para - restored_model
    model_size = select_para.nelement() * 4 / 1024 / 1024
    print("model_size:",model_size)
    # print("memory_para:",memory_para)
    return (memory_para,(select_para, indices))

def count_dataset_total(loader):
    counts = np.zeros(10)
    for _, target in loader.loader:
        labels = target.view(-1).numpy()
        for label in labels:
            counts[label] += 1
    print("class counts: ", counts)
    print("total data count: ", np.sum(counts))

if __name__ == '__main__':
    main()
