import torch
import importlib
import yaml
import os
import sys
from pathlib import Path
from tqdm.auto import tqdm
import pprint
import argparse
import numpy as np
import pandas as pd
import pickle
from time import gmtime, strftime
import time


class DLRunner():

    def __init__(self, configpath, logfile=None, inference_=False):
        if(logfile is not None):
            sys.stdout = logfile
        pp = pprint.PrettyPrinter(indent=4)
        with open(configpath, 'r') as file:
            config = yaml.safe_load(file)
            print(f'Configuration : {pp.pprint(config)}', flush=True)
        self.modelsetting = config.get('model',None)
        self.dataset = self.modelsetting.get('dataset',None)
        self.dataloaderparams = self.modelsetting.get('datasetloader',None)
        self.dataset = getattr(importlib.import_module('utils.dataloader'), self.dataset)(configpath, inference_)
        self.datapostprocessing = config['data'].get('postprocessing',None)
        if (self.modelsetting is None or self.dataset is None or self.dataloaderparams is None):
            print(f'Missing Model/Dataset/Dataloader Section in the config file.', flush=True)
            raise ValueError
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f'Find Device: {self.device}', flush=True)

        self.modelname = list(self.modelsetting.keys())[0]
        self.lossname = list(config['model'].get('loss',{}).keys())[0]
        self.weightexport = self.modelsetting.get('weightexport', {'path':'./weight', 'selection':'best'})
        self.experimentresult = self.modelsetting.get('experimentresult',{'path':'./result', 'exportSample': False})
        self.loadweightpath = self.modelsetting.get('weight',None)
        self.resultpath = self.experimentresult['path'] if (type(self.experimentresult['path']) == str) else '/'.join(self.experimentresult['path'])
        self.weightexportpath = self.weightexport['path'] if (type(self.weightexport['path']) == str) else '/'.join(self.weightexport['path'])
        self.exportSample =self.experimentresult['exportSample']
        self.Model = getattr(importlib.import_module(f'model.{".".join(self.modelname.split(".")[:-1])}'), self.modelname.split('.')[-1])
        self.model = self.Model(outputActivate=False if ('logits' in self.lossname.lower()) else True, **self.modelsetting[self.modelname])

        if (not inference_):
            self.optname = list(config['model'].get('opt',{}).keys())[0]
            self.epoch = config['model'].get('epoch',100)
            if (self.modelsetting.get('accuracymetric')):
                self.accythreshold = self.modelsetting['accuracymetric'].get('threshold',0.5)
                del self.modelsetting['accuracymetric']['threshold']
                self.accuracymetric = self.modelsetting['accuracymetric']
            else:
                self.accuracymetric = {}
            if (self.modelsetting.get('custommetric')):
                self.cust_accuracymodule = self.modelsetting['custommetric']['module']
                del self.modelsetting['custommetric']['module']
                self.cust_accuracymetric = self.modelsetting['custommetric']
            else:
                self.cust_accuracymetric = {}
            self.exportweightcriterion = self.weightexport.get('additional_criterion')
            self.Loss = getattr(importlib.import_module('torch.nn'), self.lossname) if (self.lossname != 'custom') else getattr(importlib.import_module(config['model']['loss']['custom']['module']),config['model']['loss']['custom']['function'])
            self.Opt = getattr(importlib.import_module('torch.optim'), self.optname)
            self.accuracymetricF = [getattr(importlib.import_module('torcheval.metrics'), metric) for metric in self.accuracymetric.keys()]
            self.cust_accuracymetricF = [getattr(importlib.import_module(self.cust_accuracymodule), cmetric) for cmetric in self.cust_accuracymetric.keys()]
            print(self.accuracymetricF, flush=True)
            print(self.cust_accuracymetricF, flush=True)
            self.accuracymetricF = [metric(device=self.model.device, threshold=self.accythreshold, **self.accuracymetric[metric.__name__] if (self.accuracymetric[metric.__name__] is not None) else {})
                               for metric in self.accuracymetricF]
            if (self.exportweightcriterion is not None):
                self.exportweightcriterion = [i for i, metric in enumerate(self.accuracymetricF) if (metric.__class__.__name__ == self.exportweightcriterion)]
                self.exportweightcriterion = self.exportweightcriterion[0]
            self.criterion = self.Loss(**config['model']['loss']['custom']['param'] if (self.lossname == 'custom') else {})
            self.additionloss = config['model']['loss']['custom']['additionloss' ] if (self.lossname == 'custom') else []
            self.rlevel = config['model']['loss']['custom']['param']['lteReconstructionLoss'] if (self.lossname == 'custom') else 0
            self.optimizer = self.Opt(self.model.parameters(), **self.modelsetting['opt'][self.optname] if(self.modelsetting['opt'][self.optname] is not None) else {} )
            self.trainloader = torch.utils.data.DataLoader(self.dataset.trainset, **self.dataloaderparams)
            self.testloader = torch.utils.data.DataLoader(self.dataset.testset, **self.dataloaderparams)
            self.valloader = None if (self.dataset.valset is None) else torch.utils.data.DataLoader(self.dataset.valset, **self.dataloaderparams)
            self.pdcolumn = ['Epoch', 'Train_Test', 'loss', 'time(s)'] + self.additionloss + [name for name in self.accuracymetric.keys()] + [ name for name in self.cust_accuracymetric.keys() ]
            with open(f'{self.resultpath}/split.pkl', 'wb') as f:
                d = {'train': self.dataset.trainDataPath, 'test': self.dataset.testDataPath}
                if (self.valloader is not None):
                    d['val'] = self.dataset.valDataPath
                pickle.dump(d, f)
            self.avgresultdata = pd.DataFrame(columns=self.pdcolumn)
            if (not os.path.exists(self.weightexportpath)):
                path = Path(self.weightexportpath)
                path.mkdir(parents=True)
            if (not os.path.exists(self.resultpath)):
                path = Path(self.resultpath)
                path.mkdir(parents=True)
        else:
            # Inference Setting
            path = self.resultpath.split("/")
            path = "/".join(path[:-1])+"/inference_result"
            self.resultpath = path
            if (not os.path.exists(path)):
                path = Path(path)
                path.mkdir(parents=True)
            self.inferenceloader = torch.utils.data.DataLoader(self.dataset.inferenceset, **self.dataloaderparams)

        if (self.datapostprocessing is not None):
            self.datapostprocessing = getattr(importlib.import_module(f'utils.{".".join(self.datapostprocessing.split(".")[:-1])}'), self.datapostprocessing.split('.')[-1])(self.resultpath)

        print(self.model, flush=True)
        if (self.loadweightpath is not None):
            print(f'Loading weight from path :{self.loadweightpath}', flush=True)
            checkpoint = torch.load(self.loadweightpath)
            if (checkpoint.get('model_state_dict')):
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        with open(f'{"/".join(self.resultpath.split("/")[:-1])}/config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


    def train(self):
    # Training
        testbesloss = torch.inf
        additionalcriterion = 0
        for i in range(self.epoch):
            self.model.train()
            trainingloss, testloss, valloss = 0, 0, 0
            trainingaddition_loss = [ 0 for i in range(len(self.additionloss)) ]
            testaddition_loss = [ 0 for i in range(len(self.additionloss)) ]
            valaddition_loss = [ 0 for i in range(len(self.additionloss)) ]
            batchtrainingresult = []
            batch_cust_trainingresult=[]
            batchtestresult = []
            batch_cust_testresult = []
            batchvalresult = []
            batch_cust_valresult = []
            trainstarttime = time.time()
            for batch, data in enumerate(tqdm(self.trainloader, desc=f'Epoch {i:3d}, Training',position=0, leave=True)):
                indata, groundtrue = data[0], data[1]
                groundtrue = groundtrue.to(self.device)
                output = self.model(indata)
                loss = self.criterion(output, groundtrue)
                trainingloss += loss
                trainingaddition_loss = list(map(lambda x: trainingaddition_loss[x[0]] + getattr(self.criterion, x[1]).detach().cpu().numpy(), enumerate(self.additionloss)))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                output = torch.nn.functional.sigmoid(output) if ("logits" in self.lossname.lower()) else output
                if (type(output) is tuple):
                    output = output[0]
                    output = torch.nn.functional.sigmoid(output)
                [metric.update(torch.flatten(output), torch.flatten(groundtrue).type(torch.IntTensor)) for metric in self.accuracymetricF]
                if (len(self.cust_accuracymetricF)>0):
                    foutput = torch.flatten(output).detach().cpu()
                    fgroundtrue = torch.flatten(groundtrue).type(torch.IntTensor).detach().cpu()
                    batch_cust_trainingresult.append([metric({'y_pred': foutput, 'y_true': fgroundtrue, 'threshold': self.accythreshold}) for metric in self.cust_accuracymetricF])
            trainendtime = time.time()
            metric_ = [metric.compute() for metric in self.accuracymetricF]
            batchtrainingresult = [m.detach().cpu().numpy() for m in metric_]
            batch_cust_trainingresult = np.mean(np.array(batch_cust_trainingresult),axis=0) if (len(self.cust_accuracymetricF)>0) else batch_cust_trainingresult
            [metric.reset() for metric in self.accuracymetricF]
            self.model.eval()
            with torch.no_grad():
                if (self.valloader):
                    valstarttime = time.time()
                    for valbatch, data in enumerate(tqdm(self.valloader, desc=f'Epoch {i:3d}, Val',position=0, leave=True)):
                        indata, groundtrue, filename = data[0], data[1], data[2]
                        groundtrue = groundtrue.to(self.device)
                        output = self.model(indata)
                        loss = self.criterion(output, groundtrue)
                        valloss += loss
                        valaddition_loss = list(map(lambda x: valaddition_loss[x[0]] + getattr(self.criterion, x[1]).detach().cpu().numpy(), enumerate(self.additionloss)))
                        output = torch.nn.functional.sigmoid(output) if ("logits" in self.lossname.lower()) else output
                        if (type(output) is tuple):
                            output = output[0]
                            output = torch.nn.functional.sigmoid(output)
                        [metric.update(torch.flatten(output), torch.flatten(groundtrue).type(torch.IntTensor)) for metric in self.accuracymetricF]
                        if (len(self.cust_accuracymetricF)>0):
                            foutput = torch.flatten(output).detach().cpu()
                            fgroundtrue = torch.flatten(groundtrue).type(torch.IntTensor).detach().cpu()
                            batch_cust_valresult.append([metric({'y_pred': foutput, 'y_true': fgroundtrue, 'threshold': self.accythreshold}) for metric in self.cust_accuracymetricF])
                    valendtime = time.time()
                    metric_ = [metric.compute() for metric in self.accuracymetricF]
                    batchvalresult = [m.detach().cpu().numpy() for m in metric_]
                    batch_cust_valresult = np.mean(np.array(batch_cust_valresult),axis=0) if (len(self.cust_accuracymetricF)>0) else batch_cust_valresult
                    [metric.reset() for metric in self.accuracymetricF]
                teststarttime = time.time()
                for testbatch, data in enumerate(tqdm(self.testloader, desc=f'Epoch {i:3d}, Testing',position=0, leave=True)):
                    indata, groundtrue, filename = data[0], data[1], data[2]
                    groundtrue = groundtrue.to(self.device)
                    output = self.model(indata)
                    loss = self.criterion(output, groundtrue)
                    testloss += loss
                    testaddition_loss = list(map(lambda x: testaddition_loss[x[0]] + getattr(self.criterion, x[1]).detach().cpu().numpy(), enumerate(self.additionloss)))
                    output = torch.nn.functional.sigmoid(output) if ("logits" in self.lossname.lower()) else output
                    if (type(output) is tuple):
                        output = output[0]
                        output = torch.nn.functional.sigmoid(output)
                    [metric.update(torch.flatten(output), torch.flatten(groundtrue).type(torch.IntTensor)) for metric in self.accuracymetricF]
                    if (len(self.cust_accuracymetricF)>0):
                        foutput = torch.flatten(output).detach().cpu()
                        fgroundtrue = torch.flatten(groundtrue).type(torch.IntTensor).detach().cpu()
                        batch_cust_testresult.append([metric({'y_pred': foutput, 'y_true': fgroundtrue, 'threshold': self.accythreshold}) for metric in self.cust_accuracymetricF])
                    if (self.exportSample):
                        if (self.datapostprocessing is None):
                            with open(f'{self.resultpath}/Epoch{i}_TestSet_batch_{testbatch}_{strftime("%Y-%m-%d %H:%M:%S", gmtime())}', 'wb') as f:
                                pickle.dump({'file': filename, 'data': output}, f)
                        else:
                            p = self.resultpath + f"/Epoch_{i}_batch_{testbatch}"
                            os.mkdir(p)
                            self.datapostprocessing.exportpath = p
                            data[0] = output.cpu().detach().numpy()
                            self.datapostprocessing.export2file(data)
                testendtime = time.time()
            metric_ = [metric.compute() for metric in self.accuracymetricF]
            batchtestresult = [m.detach().cpu().numpy() for m in metric_]
            batch_cust_testresult = np.mean(np.array(batch_cust_testresult),axis=0) if (len(self.cust_accuracymetricF)>0) else batch_cust_testresult
            [metric.reset() for metric in self.accuracymetricF]
            trainingloss = trainingloss.detach().cpu().numpy() / (batch+1)
            testloss = testloss.detach().cpu().numpy() / (testbatch+1)
            trainingaddition_loss = [ i/(batch+1) for i in trainingaddition_loss]
            testaddition_loss = [ i/(testbatch+1) for i in testaddition_loss]
            c = batchtestresult[self.exportweightcriterion] if (self.exportweightcriterion is not None) else 0
            if (len(self.cust_accuracymetricF)>0):
                batch_cust_trainingresult = batch_cust_trainingresult.tolist()
                batch_cust_testresult = batch_cust_testresult.tolist()
            batchtrainingresult = [i, 'Train', trainingloss, (trainendtime-trainstarttime)] + trainingaddition_loss + batchtrainingresult + batch_cust_trainingresult
            batchtestresult = [i, 'Test', testloss, (testendtime-teststarttime) ] + testaddition_loss + batchtestresult + batch_cust_testresult
            self.avgresultdata.loc[len(self.avgresultdata)] = batchtrainingresult
            self.avgresultdata.loc[len(self.avgresultdata)] = batchtestresult
            if (self.valloader):
                valloss = valloss.detach().cpu().numpy() / (valbatch + 1)
                valaddition_loss = [i / (valbatch + 1) for i in valaddition_loss]
                if (len(self.cust_accuracymetricF)>0):
                    batch_cust_valresult = batch_cust_valresult.tolist()
                batchvalresult = [i, 'Val', valloss, (valendtime-valstarttime)] + valaddition_loss + batchvalresult + batch_cust_valresult
                self.avgresultdata.loc[len(self.avgresultdata)] = batchvalresult
            self.avgresultdata.to_pickle(f'{self.resultpath}/result.pkl')
            if (self.lossname == 'custom' and (self.criterion.beta > 0 and self.criterion.beta < 1)):
                mloss = trainingaddition_loss[self.additionloss.index('reconstructionloss')]
                self.criterion.beta = (1 / (mloss * self.rlevel))
            if (testloss < testbesloss):
                testbesloss = testloss
            if (self.weightexport['selection'].lower() == 'best'):
                torch.save({ 'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict() }, f'{self.weightexportpath}/model_weight_Epoch_{i}_testloss_{testloss:0.3f}.pkl')
            elif (self.weightexport['selection'].lower() == 'all'):
                torch.save({ 'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict() }, f'{self.weightexportpath}/model_weight_Epoch_{i}_testloss_{testloss:0.3f}.pkl')
            if (self.exportweightcriterion is not None):
                if (c > additionalcriterion):
                    additionalcriterion = c
                    torch.save({ 'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict() }, f'{self.weightexportpath}/model_weight_Epoch_{i}_testloss_{testloss:0.3f}.pkl')


            # print(f'Training loss:{(trainingloss/batch):.4f} Testing loss:{(testloss/testbatch):.4f}')
            idx = -3 if (self.valloader) else -2
            print(self.avgresultdata[idx:].to_string(), flush=True)

    def inference(self):
        self.model.eval()
        with torch.no_grad():
            teststarttime = time.time()
            for testbatch, data in enumerate(tqdm(self.inferenceloader, desc='Inference', position=0, leave=True)):
                indata, _, filename = data[0], data[1], data[2]
                output = self.model(indata)
                output = torch.nn.functional.sigmoid(output) if ("logits" in self.lossname.lower()) else output
                if (type(output) is tuple):
                    output = output[0]
                    output = torch.nn.functional.sigmoid(output)
                output = output.detach().cpu().numpy()
                if (self.datapostprocessing is None):
                    with open(f'{self.resultpath}/prediction_TestSet_batch_{testbatch}_{strftime("%Y-%m-%d %H:%M:%S", gmtime())}', 'wb') as f:
                        pickle.dump({'file': filename, 'data': output}, f)
                else:
                    data[0] = output
                    self.datapostprocessing.export2file(data)
            testendtime = time.time()
        print(f"Inference Time: {testendtime-teststarttime}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str, default='./config.yml')
    ap.add_argument("-inference", type=bool, default=False)
    ap.add_argument('-mp', type=int, default=1)
    ap.add_argument('-logdir', type=str, default='.')
    args = ap.parse_args()
    configpath = args.config
    inference = args.inference
    logdir = args.logdir
    mp = args.mp
    if (os.path.isdir(configpath)):
        import pymp
        config = [configpath + '/' + c for c in os.listdir(configpath) if (c.endswith('.yaml') or c.endswith('yml'))]
        print(f'Find Config : {config}', flush=True)
        with pymp.Parallel(mp) as p:
            for i in p.range(len(config)):
                with open(f'{config[i]}', 'r') as file:
                    c = yaml.safe_load(file)
                logname = c['experimentname']
                orig_stdout = sys.stdout
                f = open(f'{logdir}/log_{logname}', 'w')
                runner = DLRunner(config[i], f, inference)
                if (inference):
                    runner.inference()
                else:
                    runner.train()
    else:
        runner = DLRunner(configpath, inference_=inference)
        if (inference):
            runner.inference()
        else:
            runner.train()
