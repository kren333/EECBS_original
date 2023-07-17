from shutil import which
import numpy as np
from argparse import ArgumentParser
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import pdb
from collections import defaultdict
import pickle
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn as nn
# import pandas as pd
import wandb

## Importing ML stuff
from model_utils import identity, relativeErrorLossFunction, MultiHeadActivation, \
            simple_CNN, simple_CNN2, simple_MLP, simple_MLP2, log_cosh_loss
import pytorch_utils as ptu
import visualizer
# from createDataset import LocalHeuristicDataset
from parse_paths_and_map import PipelineDataset
from custom_timer import CustomTimer


def get_dataset3(filePaths, size, combineIntoSingle):
    """ Input:
        filePaths: list of strings
        size: int
        combineIntoSingle: boolean
    Output:
        if combineIntoSinge: single combined dataset
        else: dictionary with key=string name, value=dataset
    """
    datasets = dict()
    for aFilePath in filePaths:
        dataName = Path(aFilePath).stem
        #TODO add size back in to PipelineDataset
        datasets[dataName] = PipelineDataset(aFilePath, 4) #TODO change to non-hard coded window size = 4
    if combineIntoSingle:
        dataset = torch.utils.data.ConcatDataset(datasets.values())
        print("Combined dataset size: {}".format(len(dataset)))
        return dataset
    else:
        for aName, aDataset in datasets.items():
            print("Name: {}, Dataset size: {}".format(aName, len(aDataset)))
        return datasets

class TrainingLoop:
    def __init__(self, train_data_loader, eval_data_loaders,
                        my_model, loss_function,
                        outputFolder, batch_size):
        """ Inputs:
            train_data_loader: 1 data loader
            eval_data_loaders: dict with [name]=[data_loader]
            my_model: NN Model
            loss function: Relative or raw
            outputFolder: Save directory
            batchsize: Int for training
        """
        # Note: We can access dataset from dataloader by doing dataloader.dataset
        self.train_data_loader = train_data_loader
        # assert("" in test_data_loaders.keys() and "" in val_data_loaders.keys())
        # self.test_data_loaders = test_data_loaders
        # self.val_data_loaders = val_data_loaders

        # assert("val" not in eval_data_loaders.keys())
        ### Check that there is at least one dataloader with "val" in the key name
        atLeastOneValSet = True
        for aKey in eval_data_loaders.keys():
            assert(aKey != "val")
            atLeastOneValSet = atLeastOneValSet or ("val" in aKey)
        assert(atLeastOneValSet)

        self.eval_data_loaders = eval_data_loaders
        self.my_model = my_model
        self.loss_function = loss_function
        self.outputFolder = outputFolder
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.my_model.parameters(), lr=1e-3)

        self.logDict = defaultdict(list)
        self.logDict["trainingSize"] = len(train_data_loader.dataset)
        self.timer = CustomTimer()
        # self.relFunc = lambda pred, true: torch.log(pred + 1)-torch.log(true + 1) # relativeErrorLossFunction(1, reduction="none")
        self.relFunc = lambda pred, true: torch.log(pred[:,0] + 1)-torch.log(true[0] + 1)
        self.curStep = 0

    def prepare_inputs(self, sampled_batch, isNumpy=False):
        def prep(x):
            return x.to(ptu.device).float()
        # obs, hvals, newHVal = sampled_batch
        # if isNumpy:
        #     obs, hvals, newHVal = ptu.from_numpy(obs), ptu.from_numpy(hvals), ptu.from_numpy(newHVal)
        if isNumpy:
            obs, backward_d, helper_bd, nearby_agents, state = \
                        [ptu.from_numpy(x) for x in sampled_batch]
        else:
            obs, backward_d, helper_bd, nearby_agents, state = sampled_batch

        #TODO Fix everything below, things should be loaded in properly (deal with nearby agents)

        inputs = torch.stack([obs, backward_d, helper_bd], dim=1) # (B,3,D,D)
        if isinstance(self.my_model, simple_MLP) or \
                isinstance(self.my_model, simple_MLP2): # Flatten if model is an MLP
            inputs = inputs.reshape((obs.shape[0], -1)) # (B,3,D,D) -> (B,3*D*D)

        targets = prep(state)

        return (prep(inputs), prep(nearby_agents)), targets

    def train_on_dataloader(self, dataloader, key):
        """
        Input: dataloader
        Output: mean loss
        Side effects: Updates my_model
        """
        self.my_model.train()
        all_losses, raw_non_mse = [], []
        for i_batch, sampled_batch in enumerate(dataloader):
            inputs, targets = self.prepare_inputs(sampled_batch)  # [(B,3,D,D),optional=?], (5)

            self.optimizer.zero_grad()
            preds = self.my_model.forward(*inputs)  # preds (float)
            loss = self.loss_function(preds,targets)
            # aVar = torch.var(self.relFunc(preds.detach(), targets.detach()))
            # all_vars.append(aVar.detach().item())
            # pdb.set_trace()

            loss.backward()
            self.optimizer.step()
            all_losses.append(loss.detach().item())  # Regular number
            raw_non_mse.append(self.relFunc(preds, targets).mean().detach().item())

        self.logDict[key+"_Loss"].append(np.mean(all_losses))
        self.logDict[key+"_Mean"].append(np.mean(raw_non_mse))
        self.logDict[key+"_Var"].append(np.var(raw_non_mse))
        for ending in ["Loss", "Mean", "Var"]:
            keyName = "{}_{}".format(key, ending)
            wandb.log({keyName: self.logDict[keyName][-1]}, step=self.curStep)
        return np.mean(all_losses), np.mean(raw_non_mse), np.var(raw_non_mse)

    def test_on_dataloader(self, dataloader, key):
        """
        Input: dataloader
        Output: mean loss
        Side effects: None
        """
        self.logDict[key+"_XVals"] = []
        self.logDict[key+"_YVals"] = []
        self.my_model.eval()
        all_losses, raw_non_mse = [], []
        with torch.no_grad(): # No gradients needed
            for i_batch, sampled_batch in enumerate(dataloader):
                inputs, targets = self.prepare_inputs(sampled_batch)  #(B,2,D,D), (B)
                preds = self.my_model.forward(*inputs)  # preds (float)
                loss = self.loss_function(preds,targets)
                all_losses.append(loss.detach().item())  # Regular number
                raw_non_mse.append(self.relFunc(preds, targets).mean().detach().item())

                raw_log_error = ptu.get_numpy(self.relFunc(preds, targets))
                self.logDict[key+"_XVals"].extend(ptu.get_numpy(targets[0]))
                self.logDict[key+"_YVals"].extend(raw_log_error)

        self.logDict[key+"_XVals"] = np.array(self.logDict[key+"_XVals"])
        self.logDict[key+"_YVals"] = np.array(self.logDict[key+"_YVals"])
        self.logDict[key+"_Loss"].append(np.mean(all_losses))
        self.logDict[key+"_Mean"].append(np.mean(raw_non_mse))
        self.logDict[key+"_Var"].append(np.var(raw_non_mse))

        # wandb.log({key+"_Loss": np.mean(all_losses)})
        for ending in ["Loss", "Mean", "Var"]:
            keyName = "{}_{}".format(key, ending)
            wandb.log({keyName: self.logDict[keyName][-1]}, step=self.curStep)
        return np.mean(all_losses), np.mean(raw_non_mse), np.var(raw_non_mse)

    def train_loop(self, numEpochs):
        maxTimeTrain = 60*60  # 20 minutes in seconds
        numViz = 10
        for i in tqdm(range(numEpochs)):
            self.curStep = i
            with self.timer("train"):
                trainLoss, trainMean, trainVar = self.train_on_dataloader(self.train_data_loader, "train")

            for aKey, aDataLoader in self.eval_data_loaders.items():
                _, _, _ = self.test_on_dataloader(aDataLoader, aKey)

            ### Aggregegate net "val" statistics
            valDict = defaultdict(list)
            for aKey in self.logDict.keys():
                if "val" in aKey and not aKey.startswith("totalVal_"):
                    whichVal = aKey.split("_")[-1]
                    # if whichVal in ["XVals", "YVals"]:
                        # pdb.set_trace()
                        # valDict[whichVal].extend(self.logDict[aKey])
                    if whichVal in ["Loss", "Mean", "Var"]:
                        valDict[whichVal].append(self.logDict[aKey][-1])
            for aKey in valDict.keys():
                if aKey in ["Loss", "Mean", "Var"]:
                    self.logDict["totalVal_"+aKey].append(np.mean(valDict[aKey]))
                    wandb.log({"totalVal_"+aKey: np.mean(valDict[aKey])}, step=self.curStep)
                # else:
                    # self.logDict["totalVal_"+aKey] = valDict[whichVal]

            self.saveModel("totalVal_Loss")
            self.saveModel("train_Loss")

            ### Plot loss curves
            plt.figure()
            plt.plot(self.logDict["train_Loss"], label="TrainLoss")
            plt.plot(self.logDict["totalVal_Loss"], label="TotalValLoss")
            for aKey in self.eval_data_loaders.keys():
                plt.plot(self.logDict[aKey+"_Loss"], label=aKey+"Loss")
            # plt.plot(self.logDict["testLoss"], label="Test")
            if i > 10:
                plt.ylim(top=self.logDict["train_Loss"][4])  # Ignore initial spike in plot
            plt.legend(loc="best")
            plt.savefig("{}/lossCurves.png".format(self.outputFolder))
            plt.close('all')


            if i % numViz == 0:
                self.visualize_values(i, self.train_data_loader, "train")
                for aKey, aDataLoader in self.eval_data_loaders.items():
                    self.visualize_values(i, aDataLoader, aKey)

                if i > 5:
                    plt.figure()
                    normVal = self.batch_size
                    plt.plot(np.array(self.logDict["train_Var"][5:])*normVal, label="TrainVar")
                    plt.plot(np.array(self.logDict["totalVal_Var"][5:])*normVal, label="TotalValVar")
                    # plt.plot(np.array(self.logDict["testVar"][5:])*normVal, label="TestVar")
                    for aKey in self.eval_data_loaders.keys():
                        plt.plot(np.array(self.logDict[aKey+"_Var"][5:])*normVal, label=aKey+"Var")

                    plt.legend(loc="best")
                    plt.savefig("{}/lossCurvesVar.png".format(self.outputFolder))
                    plt.close('all')

                plt.figure()
                plt.plot(np.array(self.logDict["train_Mean"]), label="TrainMean")
                plt.plot(np.array(self.logDict["totalVal_Mean"]), label="TotalValMean")
                # plt.plot(np.array(self.logDict["testMean"]), label="TestMean")
                for aKey in self.eval_data_loaders.keys():
                    plt.plot(np.array(self.logDict[aKey+"_Mean"]), label=aKey+"Mean")
                plt.legend(loc="best")
                plt.savefig("{}/lossCurvesMean.png".format(self.outputFolder))
                plt.close('all')

                plt.figure()
                # plt.scatter(self.logDict["trainXVals"], self.logDict["trainYVals"], label="trainRelative",
                    # alpha=0.5)
                numSamples = 1000
                for aKey in self.eval_data_loaders.keys():
                    plt.scatter(self.logDict[aKey+"_XVals"][:numSamples], self.logDict[aKey+"_YVals"][:numSamples],
                        label=aKey+"Relative", alpha=0.2)
                plt.legend(loc="best")
                plt.savefig("{}/lossBreakDown.png".format(self.outputFolder))
                plt.close('all')

                if i > 2:
                    plt.figure()
                    # pdb.set_trace()
                    normVal = self.batch_size
                    plt.plot(self.logDict["train_Loss"][2:], np.array(self.logDict["train_Var"][2:])*normVal, label="TrainVar")
                    plt.plot(self.logDict["totalVal_Loss"][2:], np.array(self.logDict["totalVal_Var"][2:])*normVal, label="TotalValVar")
                    # plt.plot(self.logDict["testLoss"][2:], np.array(self.logDict["testVar"][2:])*normVal, label="TestVar")
                    for aKey in self.eval_data_loaders.keys():
                        plt.plot(self.logDict[aKey+"_Loss"][2:], np.array(self.logDict[aKey+"_Var"][2:])*normVal, label=aKey+"Var")
                    plt.legend(loc="best")
                    plt.savefig("{}/lossCurvesRegression.png".format(self.outputFolder))
                    plt.close('all')
            # print("Ave time per train epoch: {}".format(self.timer.getTimes("train")/(i+1)))

            with open('{}/logDict.pkl'.format(self.outputFolder), 'wb') as f:
                pickle.dump(self.logDict, f)

            # If our last 10 epochs were all worse than the 10 epochs before
            if (i > 20 and np.min(self.logDict["train_Loss"][-10:]) > np.max(self.logDict["train_Loss"][-20:-10])) or \
                    self.timer.getTimes("train") > maxTimeTrain or \
                    (i > 20 and np.mean(self.logDict["totalVal_Loss"][-10:]) < 0.01): # If validation loss less than 1 percent
                tqdm.write("Early finish: iteration {}".format(i))
                break


    def saveModel(self, key):
        losses = np.array(self.logDict[key])

        # If our last loss was the lowest, save the model
        if losses[-1] == np.min(losses):
            sm = torch.jit.script(self.my_model)
            # sm.save("{}/my_module_model.pt".format(self.outputFolder))
            sm.save("{}/best_{}_model.pt".format(self.outputFolder, key))


    def visualize_values(self, iter, dataLoader, key):
        """ Plot 10 random example images from dataLoader """
        outputFolder = "{}/images".format(self.outputFolder)
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        dataset = dataLoader.dataset
        K = 10
        rand_indices = np.random.randint(0, len(dataset), K)
        sampled_batch = [dataset[i] for i in rand_indices]  # np, ((),(),())xK
        sampled_batch = list(map(np.stack, zip(*sampled_batch)))  # np, (K,...)
        # sampled_batch: obs (B,D,D), hval (B,D,D), state (B,3)
        inputs, targets = self.prepare_inputs(sampled_batch, isNumpy=True)  # torch
        preds = self.my_model.forward(*inputs)  # torch (B,3)

        # pdb.set_trace()
        # sampled_batch.append(ptu.get_numpy(preds[:,0].flatten()))
        sampled_batch.append(ptu.get_numpy(preds))
        data = list(zip(*sampled_batch))
        visualizer.plotCustomMulti(visualizer.plotLH2, data, (2, K//2), "{}/{}_{}.png".format(
                                                                        outputFolder, key, iter))



def practice_classifier(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # pdb.set_trace()
    if "/" in args.outputFolder: # If absolute path
        outputFolder = args.outputFolder
    else:  # Else relative to logs
        # outputFolder = "/home/rishi/Desktop/CMU/Research/search-zoo/logs/{}".format(args.outputFolder)
        outputFolder = "../logs/{}".format(args.outputFolder)

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    # train_dataset = get_dataset(args.folderpath, "train", args.size)
    # val_dataset = get_dataset(args.folderpath, "val", -1)  # Should be 10k
    # test_dataset = get_dataset(args.folderpath, "test", 10000)

    # trainDataset = [PipelineDataset('', 0, '')]
    trainDataset = get_dataset3(args.trainData.split(","), args.size, True)
    evalDatasetsDict = get_dataset3(args.evalData.split(","), 10000, False)
    # pdb.set_trace()
    # train_dev_sets = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    # tmp1 = get_dataset2("", ["Berlin", "Paris"], "train", 100)
    # tmp2 = get_dataset2("", ["Berlin", "Paris"], "train", 1000)
    # pdb.set_trace()
    windowHalfSize = 4

    # See https://github.com/pytorch/pytorch/issues/43094 for more info if wanted
    print("Using lhK value: {}".format(windowHalfSize))


    using_gpu = args.which_gpu >= 0
    batch_size = args.batch_size #32
    ### Create MLP model
    # mlp_info = dict(
    #     input_size = pow(windowHalfSize*2+1,2)*2,
    #     sizes=[100, 100, 1],
    #     hidden_activation=[nn.ReLU(), nn.ELU()][0],
    #     output_activation=[identity, torch.abs, nn.ReLU(), multiOuput][-1]
    # )
    # my_model = simple_MLP(**mlp_info)

    # mlp_info = dict(
    #     input_size = pow(windowHalfSize*2+1,2)*2,
    #     sizes=[100, 100, 100, 1],
    #     hidden_activation=[nn.ReLU(), nn.ELU()][0],
    #     output_activation=[nn.ReLU(), identity, nn.Sigmoid()] # Extra cost, deltaK, binary
    # )
    # my_model = simple_MLP2(**mlp_info)

    ### Create MLP model
    # cnn_info = dict(
    #     input_dim = windowHalfSize*2+1, # 41->21->11
    #     input_channels = 2,
    #     kernel_sizes = [5, 5],
    #     n_channels = [8, 8],
    #     strides = [2, 2],
    #     paddings = [2, 2],
    #     max_pool_sizes = [1, 1],
    #     hidden_activation = [nn.ReLU(), nn.ELU()][0],
    #     output_activation = [identity, torch.abs, nn.ReLU()][1],
    #     output_flatten = True,
    #     mlp_sizes = [100, 1],  # 6x6x3=~100->100->1
    # )
    # my_model = simple_CNN(**cnn_info)

    # def multiHeadActivation(preds):
    #     """preds: (B,3) extraCost, deltaK, blockedProb
    #     Output: stack((B),(B),(B))->(B,3)
    #     """
    #     f1, f2, f3 = nn.ReLU(), identity, nn.Sigmoid()
    #     return torch.stack([f1(preds[:,0]), f2(preds[:,1]), f3(preds[:,2])], dim=1)
    # multiOuput = MultiHeadActivation(maxPred)
    # cnn_info = dict(
    #     input_dim = windowHalfSize*2+1, # 41->21->11
    #     input_channels = 2,
    #     kernel_sizes = [5, 5],
    #     n_channels = [8, 8],
    #     strides = [2, 2],
    #     paddings = [2, 2],
    #     max_pool_sizes = [1, 1],
    #     hidden_activation = [nn.ReLU(), nn.ELU()][0],
    #     output_activation = [identity, torch.abs, nn.ReLU(), multiOuput][3],
    #     output_flatten = True,
    #     mlp_sizes = [100, 3],  # 6x6x3=~100->100->3 (extraCost, deltaK, blockedProb)
    #     mlp_extra_input_size = 5, # (x,y,theta,vel,t)->(theta,vel,t)
    # )
    cnn_info = dict(
        input_dim = windowHalfSize*2+1, # 41->21->11
        input_channels = 3,
        kernel_sizes = [3],
        n_channels = [8],
        strides = [2],
        paddings = [2],
        max_pool_sizes = [1],
        hidden_activation = [nn.ReLU(), nn.ELU()][1],
        output_activation = nn.Softmax(),
        output_flatten = True,
        mlp_sizes = [100, 100, 5],  # 6x6x3=~100->100->5 (one hot vector of size 5)
        mlp_extra_input_size = 8, # flattened (4,2) for the neighboring agents' positions
    )
    my_model = simple_CNN2(**cnn_info)

    ## Generic model stuff
    if isinstance(my_model, simple_MLP) or isinstance(my_model, simple_MLP2):
        modelType = "MLP"
    else:
        modelType = "CNN"
    my_model.to(ptu.device)

    train_data_loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size,
                                                shuffle=True, num_workers=0, pin_memory=using_gpu)
    evalDataLoaders = dict()
    for aName, aDataset in evalDatasetsDict.items():
        evalDataLoaders[aName] = torch.utils.data.DataLoader(aDataset, batch_size=batch_size,
                                    shuffle=True, num_workers=0, pin_memory=using_gpu)

    ### Supervised training loop: num_epochs, objective

    lossFunc = nn.CrossEntropyLoss()

    ### Set up wandb logging
    configDict = dict(
        batchSize=batch_size,
        totalTrainDataSize=len(trainDataset),
        outputFolder=outputFolder,
        seed=args.seed,
        arch=cnn_info,
        # arch=mlp_info,
        modelType=modelType,
    )
    if args.wandbOff:
        wandb_mode = "disabled"
    else:
        wandb_mode = "online"
    tags = ["k{}".format(windowHalfSize), modelType]
    wandb.init(project="initial-test-k{}".format(windowHalfSize), mode=wandb_mode,
                config=configDict, tags=tags)
    # wandb.run.name = outputFolder
    wandb.watch(my_model, log_freq=len(train_data_loader))

    my_trainer = TrainingLoop(train_data_loader, evalDataLoaders,
                                    my_model, lossFunc, outputFolder, batch_size)
    my_trainer.train_loop(101)

####### Main interface #######
## Example run: CUDA_VISIBLE_DEVICES=0 python trainer.py
if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("folderpath", help="folder path with train, val, test data splits", type=str)
    parser.add_argument("outputFolder", help="output folder", type=str)
    parser.add_argument("--trainData", help=".npz paths, split by ;", type=str)
    parser.add_argument("--evalData", help=".npz paths, split by ;", type=str)
    parser.add_argument('-gpu', '--which_gpu', type=int, default=-1, required=False)
    parser.add_argument('-s', '--size', type=int, default=10000, required=False)
    parser.add_argument('-seed', '--seed', type=int, default=1, required=False)
    parser.add_argument('-bs', '--batch_size', type=int, default=32, required=False)
    parser.add_argument('-wandbOff', '--wandbOff', type=bool, default=False, required=False)
    args = parser.parse_args()

    ptu.set_device(args.which_gpu)  # Set gpu or cpu device

    practice_classifier(args)
