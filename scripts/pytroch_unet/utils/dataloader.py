import os
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose
from skimage.transform import resize
import importlib
from operator import itemgetter
import importlib.util
import yaml
import numpy as np
import math
from copy import deepcopy


class TrainTestDataSet():

    def __init__(self, dataset_config=None, inference=False):
        with open(dataset_config, 'r') as file:
            dataset_config = yaml.safe_load(file)
        dataconfig = dataset_config['data']

        self.dataloadername = dataconfig['loader']
        self.filetype = dataconfig['filetype']
        self.tranformation = dataconfig.get('transforms',None)
        self.testsetApplyTransformation = dataconfig.get('testsettransforms',True)
        self.Dataloader = getattr(importlib.import_module('utils.dataloader'), self.dataloadername)
        self.trainDataPath = None
        self.testDataPath = None
        self.valDataPath = None
        self.trainset = None
        self.testset = None
        self.valset = None
        self.inferenceset = None
        self.inference = inference
        if (not inference):
            print(f"{__class__}: Load Training and Testing dataset")
            if (dataconfig.get('testing') is None):
                # Perform Train Test split
                datasetpath = dataconfig['training']['data']
                labelpath = dataconfig['training']['label']
                trainratio = dataconfig.get('trainratio', 0.8)
                print(f"{__class__}: Testing Set not defined, using TrainTest split with train ratio: {trainratio}")
                if ((os.path.isdir(datasetpath))==False or (os.path.isdir(labelpath))==False):
                    raise Exception(f"{__class__}: Training data path or label path is not a valid directory.")
                datasetpath = [f'{datasetpath}/{file}' for file in os.listdir(datasetpath) if (file.split('.')[-1] == self.filetype)]
                split = int(np.ceil(trainratio * len(datasetpath)))
                idx = np.arange(len(datasetpath),dtype=int)
                np.random.shuffle(idx)
                self.trainset = self.Dataloader(itemgetter(*idx[:split])(datasetpath), labelpath, self.testsetApplyTransformation, self.tranformation)
                self.testset = self.Dataloader(itemgetter(*idx[split:])(datasetpath), labelpath, self.testsetApplyTransformation, self.tranformation)
                self.trainDataPath = itemgetter(*idx[:split])(datasetpath)
                self.testDataPath = itemgetter(*idx[split:])(datasetpath)
            else:
                # load train test data set individually
                traindatasetpath = dataconfig['training']['data']
                trainlabelpath = dataconfig['training']['label']
                testdatasetpath = dataconfig['testing']['data']
                testlabelpath = dataconfig['testing']['label']
                if ((os.path.isdir(traindatasetpath))==False or (os.path.isdir(trainlabelpath))==False):
                    raise Exception(f"{__class__}: Training data path or label path is not a valid directory.")
                traindatasetpath = [f'{traindatasetpath}/{file}' for file in os.listdir(traindatasetpath) if (file.split('.')[-1] == self.filetype)]
                if (dataconfig.get('trainratio')):
                    trainratio = dataconfig.get('trainratio', 0.8)
                    print(f"{__class__}: Validation set split with train ratio: {trainratio}")
                    split = int(np.ceil(trainratio * len(traindatasetpath)))
                    idx = np.arange(len(traindatasetpath),dtype=int)
                    np.random.shuffle(idx)
                    self.trainset = self.Dataloader(itemgetter(*idx[:split])(traindatasetpath), trainlabelpath, self.testsetApplyTransformation, self.tranformation, self.inference)
                    self.valset = self.Dataloader(itemgetter(*idx[split:])(traindatasetpath), trainlabelpath, self.testsetApplyTransformation, self.tranformation, self.inference)
                    self.trainDataPath = itemgetter(*idx[:split])(traindatasetpath)
                    self.valDataPath = itemgetter(*idx[split:])(traindatasetpath)
                else:
                    self.trainset = self.Dataloader(traindatasetpath, trainlabelpath, self.testsetApplyTransformation, self.tranformation, self.inference)
                    self.trainDataPath = traindatasetpath
                    if (dataconfig.get('validation')):
                        print(f"{__class__}: Load Validation dataset")
                        valdatapath = dataconfig['validation']['data']
                        vallabelpath = dataconfig['validation']['label']
                        valdatapath = [f'{valdatapath}/{file}' for file in os.listdir(valdatapath) if (file.split('.')[-1] == self.filetype)]
                        self.valset = self.Dataloader(valdatapath, vallabelpath, self.testsetApplyTransformation, self.tranformation, self.inference)
                        self.valDataPath = valdatapath
                if ((os.path.isdir(testdatasetpath))==False or (os.path.isdir(testlabelpath))==False):
                    raise Exception(f"{__class__}: Test data path or label path is not a valid directory.")
                testdatasetpath = [f'{testdatasetpath}/{file}' for file in os.listdir(testdatasetpath) if (file.split('.')[-1] == self.filetype)]
                self.testset = self.Dataloader(testdatasetpath, testlabelpath, self.testsetApplyTransformation, self.tranformation,self.inference)
                self.testDataPath = testdatasetpath
        else:
            print(f"{__class__}: Load Inference dataset")
            inferencesetpath = dataconfig['inference']['data']
            if (os.path.isdir(inferencesetpath)==False):
                raise Exception(f"{__class__}: inference data path is not a valid directory")
            inferencesetpath = [f'{inferencesetpath}/{file}' for file in os.listdir(inferencesetpath) if (file.split('.')[-1] == self.filetype)]
            self.inferenceset = self.Dataloader(inferencesetpath,testsetApplyTransformation=self.testsetApplyTransformation, tranformations=self.tranformation,inference=self.inference)



class ImgDataLoader(Dataset):

    def __init__(self, datasetpath, datasetlabelpath=None, testsetApplyTransformation=True, tranformations=None, inference=False):
        super().__init__()

        self.datasetpath = datasetpath
        self.datasetlabelpath = datasetlabelpath
        self.tranformation = tranformations
        self.tranforms = []
        self.apply = testsetApplyTransformation
        self.module = None
        self.inference = inference
        if (self.tranformation is not None):
            self.module =  self.tranformation['module']
            for key in self.tranformation.keys():
                if (key != 'module'):
                    t = getattr(importlib.import_module(self.module), key)
                    t = t(**self.tranformation[key]) if ('torchvision' in self.module.lower()) else {'fun':t, 'param':self.tranformation[key]}
                    self.tranforms.append(t)
            if ('torchvision' in self.module.lower()):
                self.tranforms = Compose(self.tranforms)

    def __len__(self):
        return len(self.datasetpath)

    def __getitem__(self, index):
        filename = self.datasetpath[index].split('/')[-1]
        profile = None
        labelimg = np.array([0])
        with rasterio.open(f'{self.datasetpath[index]}') as ds:
            dataimg = ds.read()
            profile = deepcopy(ds.profile.data)
            profile['crs'] = profile['crs'].to_dict()
            profile['transform'] = profile['transform'].to_gdal()
            if (profile['nodata'] is None):
                profile['nodata'] = "None"
            elif (math.isnan(profile['nodata'])):
                profile['nodata'] = "nan"
        if (not self.inference):
            with rasterio.open(f'{self.datasetlabelpath}/{filename}') as ds:
                labelimg = ds.read()
        if (self.tranformation is not None):
            if ('torchvision' in self.module.lower()):
                dataimg = torch.from_numpy(dataimg)
                dataimg = self.tranforms(dataimg)
                if (self.inference):
                    labelimg = torch.from_numpy(labelimg)
                    labelimg = self.tranforms(labelimg) if (self.apply) else labelimg
            else:
                for transform in self.tranforms:
                    dataimg = transform['fun'](dataimg,**transform['param'])
                    labelimg = transform['fun'](labelimg,**transform['param']) if (not self.inference) else labelimg
                dataimg = torch.from_numpy(dataimg)
                labelimg = torch.from_numpy(labelimg)
            return (dataimg, labelimg, filename, profile)
        else:
            dataimg = torch.from_numpy(dataimg)
            labelimg = torch.from_numpy(np.float32(labelimg)) if (not self.inference) else labelimg
            return (dataimg, labelimg, filename, profile)

class SKImgDataLoader(Dataset):

    def __init__(self, datasetpath, datasetlabelpath=None, testsetApplyTransformation=True, tranformations=None, inference=False):
        super().__init__()
        print('Fouce to use skimage for resizing to (1,512,512)')
        self.datasetpath = datasetpath
        self.datasetlabelpath = datasetlabelpath
        self.tranformation = tranformations
        self.tranforms = []
        self.apply = testsetApplyTransformation
        self.module = None
        self.inference = inference

    def __len__(self):
        return len(self.datasetpath)

    def __getitem__(self, index):
        filename = self.datasetpath[index].split('/')[-1]
        profile = None
        labelimg = np.array([0])
        with rasterio.open(f'{self.datasetpath[index]}') as ds:
            dataimg = ds.read()
            dataimg = resize(dataimg,output_shape=(1,512,512),mode='constant', preserve_range=True)
            dataimg = torch.FloatTensor(dataimg)
            profile = deepcopy(ds.profile.data)
            profile['crs'] = profile['crs'].to_dict()
            profile['transform'] = profile['transform'].to_gdal()
            if (profile['nodata'] is None):
                profile['nodata'] = "None"
            elif (math.isnan(profile['nodata'])):
                profile['nodata'] = "nan"
        if (not self.inference):
            with rasterio.open(f'{self.datasetlabelpath}/{filename}') as ds:
                labelimg = ds.read()
                labelimg = resize(labelimg,output_shape=(1,512,512), preserve_range=True, order=0)
                labelimg = torch.FloatTensor(labelimg)
        return (dataimg, labelimg, filename, str(profile))

if __name__ == "__main__":
    d = ImgDataLoader(mode="Train")
    d.__getitem__(1)
