#-*- coding: utf-8 -*-
import os
os.environ["DEVICE"] = "cuda:0"
run_id = 'test-0'
multiplier = 1

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np


import sys
import os

__WORKSPACE__ = Path.cwd().parent.parent
sys.path.append(str(__WORKSPACE__))
# os.chdir(__WORKSPACE__)
from src.clone_detection.data.data_gathering import get_id2code, get_labels

from tqdm import tqdm, trange
import argparse
from jsonparse import saveAllDataToRam, saveAllDataToRam_v2
from jsonparse import getCodePairDataList
from myModels.GAT_Edgepool_clone_detection import CodeCloneDetection
from layers.Focal_loss import FocalLoss
import random
from sklearn.metrics import recall_score,precision_score,f1_score
from myModels.GAT_Edgepool_graphEmb import graphEmb
from myModels.GAT_Edgepool_bi_lstm import bi_lstm_detect


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--hidden', type=int, default=16*multiplier)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--nheads', type=int, default=16)
parser.add_argument('--dropout', type=int, default=0.1)
parser.add_argument('--alpha', type=int, default=0.2)
parser.add_argument("--threshold", default=0)
args = parser.parse_args()
device = os.getenv("DEVICE", "cuda") if torch.cuda.is_available() else "cpu"
print(device)

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(fold, epoch, val_loss, test_acc))

#----------------------------------------------------------------


def getBatch(line_list, batch_size, batch_index, ramData):
    start_line = batch_size*batch_index
    end_line = start_line+batch_size
    dataList = getCodePairDataList(ramData,line_list[start_line:end_line])
    return dataList



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, size_average=None, reduce=None)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, size_average=None, reduce=None)
        
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

criterion=FocalLoss().to(device)

def graph_emb(data, epoch, model_save_dir):
    model = graphEmb(args.num_layers, args.hidden, args.nheads, args.num_classes, args.dropout, args.alpha, False).to(device)
    # saveModel = torch.load('./saveModel/epoch'+str(epoch)+'.pkl')
    saveModel = torch.load(model_save_dir / f'epoch{epoch}.pkl')
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in saveModel.items() if k in model_dict.keys()}
    #print(state_dict.keys())
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    #print("loaded "+ 'epoch'+str(epoch)+'.pkl')
    model.eval()
    features, edge_index, edgesAttr, adjacency, node2node_features = data
    data = features, edge_index, edgesAttr, adjacency, node2node_features
    data = map(lambda tensor: tensor.to(device), data)
    h = model(data)
    return h
def bi_lstm_detection(data,epoch, model_save_dir):
    model = bi_lstm_detect(args.num_layers, args.hidden, args.nheads, args.num_classes, args.dropout, args.alpha, False).to(device)
    # saveModel = torch.load('./saveModel/epoch'+str(epoch)+'.pkl')
    saveModel = torch.load(model_save_dir / f'epoch{epoch}.pkl')
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in saveModel.items() if k in model_dict.keys()}
    #print(state_dict.keys())
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    #print("loaded "+ 'epoch'+str(epoch)+'.pkl')
    model.eval()
    h1, h2 = data
    out = model(h1, h2)
    return out

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

def test(testlist, model_index, ramData, batch_size, model_save_dir):
    graphEmbDict = {}
    print("save graphEmbDict... ", end="")
    for codeID in tqdm(ramData, disable=True):
        data = ramData[codeID]
        graphEmbDict[codeID] = graph_emb(data, model_index, model_save_dir).tolist()
    print(f"Done.")

    notFound = 0
    testCount = 0
    y_preds = []
    y_trues = []
    batches = split_batch(testlist, batch_size)
    Test_data_batches = trange(len(batches), leave=True, desc = "Test", disable=True)
    print(f"Predicting on {len(batches)} batches. ", end="")
    for i in Test_data_batches:
        h1_batch = []
        h2_batch = []
        label_batch = []
        for codepair in batches[i]:
            try:
                test_data = codepair.split()
                graphEmbDict[test_data[0]]
                graphEmbDict[test_data[1]]
                testCount+=1
            except:
                notFound+=1
                continue

            test_data = codepair.split()
            h1 = torch.as_tensor(graphEmbDict[test_data[0]]).to(device)
            h2 = torch.as_tensor(graphEmbDict[test_data[1]]).to(device)
            label = int(test_data[2])
            
            h1_batch.append(h1)
            h2_batch.append(h2)
            label_batch.append(label)

        h1_batch_t = torch.stack(h1_batch, dim=1).squeeze(0)
        h2_batch_t = torch.stack(h2_batch, dim=1).squeeze(0)
        #print("h1_batch",h1_batch.shape)
        data = h1_batch_t, h2_batch_t
        outputs = bi_lstm_detection(data, model_index, model_save_dir)
        _, predicted = torch.max(outputs.data, 1)
        y_preds += predicted.tolist()
        y_trues += label_batch

        h1_batch = []
        h2_batch = []
        label_batch = []

        r_a=recall_score(y_trues, y_preds, average='macro')
        p_a=precision_score(y_trues, y_preds, average='macro')
        f_a=f1_score(y_trues, y_preds, average='macro')
        print(f"Precision: {p_a:.6f}\tRecall: {r_a:.5f}\tF1: {f_a:.6f}")

        Test_data_batches.set_description("Test (p_a=%.4g,r_a=%.4g,f_a=%.4g)" % (p_a, r_a, f_a))
    print("testCount",testCount)
    print("notFound",notFound)
    return p_a, r_a, f_a

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def train(model_save_dir, ramData, trainlist, validlist, testlist):
    #start_train_model_index = 7
    addNum = 0
    model = CodeCloneDetection(args.num_layers, args.hidden, args.nheads, args.num_classes, args.dropout, args.alpha, True).to(device)
    #model.load_state_dict(torch.load('./saveModel/epoch'+str(start_train_model_index)+'.pkl'))
    #优化器使用Adam
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    #print("loaded ", './saveModel/epoch'+str(start_train_model_index)+'.pkl')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("Model Parameters：", get_parameter_number(model))
    print("nheads ", args.nheads," batch_size ", args.batch_size)
    print("dropout = ",args.dropout)
    random.shuffle(trainlist)
    epochs = trange(args.epochs, leave=True, desc = "Epoch", disable=True)
    iterations = 0
    for epoch in epochs:
        #print(epoch)
        totalloss=0.0
        main_index=0.0
        #batches = create_batches(trainDataList)
        #for index, batch in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
        count = 0
        right = 0
        acc = 0
        total_steps = int(len(trainlist)/args.batch_size)
        for batch_index in tqdm(range(total_steps), disable=True):
            batch = getBatch(trainlist, args.batch_size, batch_index, ramData)
            optimizer.zero_grad()
            batchloss= 0
            recoreds = open("recoreds.txt", 'a')
            
            for data in batch:

                # data = map(lambda tensor: tensor.to(device), data)
                
                features1, features2, edge_index1, edge_index2, edgesAttr1, edgesAttr2, adjacency1, adjacency2, node2node_features1, node2node_features2, label = data
                
                data = features1, features2, edge_index1, edge_index2, edgesAttr1, edgesAttr2, adjacency1, adjacency2, node2node_features1, node2node_features2
                
                data = map(lambda tensor: tensor.to(device), data)
                label=torch.Tensor([[0,1]]).to(device) if label==1 else torch.Tensor([[1,0]]).to(device)
                #print("label ",label.device," ",label)
                output = model(data)
                #print("criterion",criterion(prediction,label))
                #cossim=F.cosine_similarity(h1,h2)
                #print("output",output)
                #print("label",label)
                batchloss = batchloss + criterion(output, label)
                count += 1
                right += torch.sum(torch.eq(torch.argmax(output, dim=1), torch.argmax(label, dim=1)))
            #print("batchloss",batchloss)
            acc = right*1.0/count
            batchloss.backward(retain_graph = True)
            optimizer.step()
            loss = batchloss.item()
            totalloss += loss
            main_index = main_index + len(batch)
            loss = totalloss/main_index
            epochs.set_description("Epoch (Loss=%g) (Acc = %g)" % (round(loss,5) , acc))
            if batch_index % 10 == 0 or batch_index+1 == total_steps:
                print(f"Epoch [{epoch+1}/{args.epochs}]\t : Batch [{batch_index+1}/{total_steps}]\t : ", end="")
                print(f"Batch Loss: {loss:.6f}\t: Accuracy: {acc:.6f}")
            iterations += 1
            recoreds.write(str(iterations+addNum*14078) +" "+ str(acc.item()) +" "+ str(loss)+"\n")
            recoreds.close()
        #if(epoch%10==0 and epoch>0):
        torch.save(model.state_dict(), model_save_dir / f'epoch{epoch+addNum}.pkl')
        val_recoreds = open("val_recoreds.txt", 'a')
        tmplist = np.random.choice(validlist, size=1_000, replace=False).tolist()
        p,r,f1 = test(tmplist, epoch+addNum, ramData, 15000, model_save_dir)
        val_recoreds.write(str(epoch+addNum) +" "+ str(p) +" "+ str(r) +" "+ str(f1)+"\n")
        val_recoreds.close()

        test_recoreds = open("test_recoreds.txt", 'a')
        tmplist = np.random.choice(testlist, size=1_000, replace=False).tolist()
        p,r,f1 = test(tmplist, epoch+addNum, ramData, 15000, model_save_dir)
        test_recoreds.write(str(epoch+addNum) +" "+ str(p) +" "+ str(r) +" "+ str(f1)+"\n")
        test_recoreds.close()


if __name__ == '__main__':
    pass

    indexdir='DataSetJsonVec/GCJ/javadata/'
    id = 'my_data'
    jsonVecPath = "DataSetJsonVec/GCJ/dataSetCfgGCJ16/"
    sourceCodePath = "googlejam4_src/"
    data_dir = Path("/home/ec22263/ataa/data")
    data_dir = Path("/Users/ataago/Documents/data")
    model_save_dir = data_dir / "model_bin" / 'CodeGraph4CCDetector' / run_id
    if not model_save_dir.exists(): model_save_dir.mkdir(parents=True)
    print(f"{model_save_dir=}")
    #jsonVecPath = "DataSetJsonVec/BCB/BCB-CFG-16v/"
    print(jsonVecPath, " ", id)
    if id=='0':
        trainfile=open(indexdir+'trainall.txt')
        validfile = open(indexdir+'valid.txt')
        testfile = open(indexdir+'test.txt')
    elif id=='13':
        trainfile = open(indexdir+'train13.txt')
        validfile = open(indexdir+'valid.txt')
        testfile = open(indexdir+'test.txt')
    elif id=='11':
        trainfile = open(indexdir+'train11.txt')
        validfile = open(indexdir+'valid.txt')
        testfile = open(indexdir+'test.txt')
    elif id=='0small':
        trainfile = open(indexdir+'trainsmall.txt')
        validfile = open(indexdir+'valid.txt')
        testfile = open(indexdir+'test.txt')
    elif id == '13small':
        trainfile = open(indexdir+'train13small.txt')
        validfile = open(indexdir+'validsmall.txt')
        testfile = open(indexdir+'testsmall.txt')
    elif id=='11small':
        trainfile = open(indexdir+'train11small.txt')
        validfile = open(indexdir+'validsmall.txt')
        testfile = open(indexdir+'testsmall.txt')

    elif id=='my_data':
        DATA_DIR = Path("/Users/ataago/Documents/data")
        # FILTERED_FILES_CSV = Path("/Users/ataago/Documents/data/preprocessed/bcb/meta.filtered_static_metrics.v2.csv")
        DATASET_TYPE = 'bcb'

        # Dataset Dir 
        DATASET_RAW_DIR = DATA_DIR / "raw" / DATASET_TYPE
        DATASET_PROCESSED_DIR = DATA_DIR / "preprocessed" / DATASET_TYPE


        # Raw dataset files
        DATASET_JSON = DATASET_RAW_DIR / 'data.jsonl'
        TRAIN_CSV = DATASET_RAW_DIR / 'train.csv'
        TEST_CSV = DATASET_RAW_DIR / 'test.csv'
        VALID_CSV = DATASET_RAW_DIR / 'valid.csv'

        # Processed dataset files
        FILTERED_FILES_CSV = DATASET_PROCESSED_DIR / 'meta.filtered_static_metrics.v2.csv'

        # Processed data dirs
        CPG_VECTOR_BIN = DATASET_PROCESSED_DIR / 'cpg_vector_bin'

        # Data Gathering
        filtered_static_metric_df = pd.read_csv(FILTERED_FILES_CSV)
        filtered_ids = filtered_static_metric_df['file_id'].tolist()
        id2code = get_id2code(data_type=DATASET_TYPE, json_file=DATASET_JSON)
        id2code = dict(filter(lambda x: x[0] in filtered_ids, id2code.items())) # Filtered id2code

        # len(id2code)

        print(f"{FILTERED_FILES_CSV=}")
        filtered_static_metric_df = pd.read_csv(FILTERED_FILES_CSV)
        filtered_ids = filtered_static_metric_df['file_id'].tolist()


        train_df = get_labels(csv_file=TRAIN_CSV, filtered_ids=filtered_ids)
        test_df = get_labels(csv_file=TEST_CSV, filtered_ids=filtered_ids)
        valid_df = get_labels(csv_file=VALID_CSV, filtered_ids=filtered_ids)

        trainlist = [' '.join(map(str, vals)) for vals in train_df.values.tolist()]
        testlist = [' '.join(map(str, vals)) for vals in test_df.values.tolist()]
        validlist = [' '.join(map(str, vals)) for vals in valid_df.values.tolist()]

    else:
        print('file not exist')
        quit()

    try:
        trainlist=trainfile.readlines()
        validlist=validfile.readlines()
        testlist=testfile.readlines()
    except:
        pass

    print("trainlist",len(trainlist))
    print("validlist",len(validlist))
    print("testlist",len(testlist))



    print("add all data to ram...")
    # ramData = saveAllDataToRam(sourceCodePath,jsonVecPath)
    ramData = saveAllDataToRam_v2(id2code=id2code, jsonVecPath=CPG_VECTOR_BIN)


    train()

