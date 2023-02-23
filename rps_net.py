'''
RPS network script with resnet-18
Copyright (c) Jathushan Rajasegaran, 2019
'''
from __future__ import print_function

# import argparse
# import os
# import shutil
# import torch
# import pdb
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
# import torch.utils.data as data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable
# from torch.autograd import gradcheck


class MultiHeadRPS_net(nn.Module):

    def __init__(self, args):
        super(MultiHeadRPS_net, self).__init__()
        self.args = args
        self.final_layers = []
        self.init(None)

    def init(self, best_path):

        """Initialize all parameters"""
        self.conv1 = []
        self.conv2 = []
        self.conv3 = []
        self.conv4 = []
        self.conv5 = []
        self.conv6 = []
        self.conv7 = []
        self.conv8 = []
        self.conv9 = []
        self.fc1 = []

        # conv1
        for i in range(self.args.M):
            exec("self.m1" + str(
                i) + " = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),nn.BatchNorm2d(64),nn.ReLU())")
            exec("self.conv1.append(self.m1" + str(i) + ")")

        # conv2
        for i in range(self.args.M):
            exec("self.m2" + str(
                i) + " = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64))")
            exec("self.conv2.append(self.m2" + str(i) + ")")

        # conv3
        for i in range(self.args.M):
            exec("self.m3" + str(
                i) + " = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64))")
            exec("self.conv3.append(self.m3" + str(i) + ")")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv4
        for i in range(self.args.M):
            exec("self.m4" + str(
                i) + " = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128))")
            exec("self.conv4.append(self.m4" + str(i) + ")")
        exec("self.m4" + str(
            "x") + " = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU())")
        exec("self.conv4.append(self.m4" + str("x") + ")")

        # conv5
        for i in range(self.args.M):
            exec("self.m5" + str(
                i) + " = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128))")
            exec("self.conv5.append(self.m5" + str(i) + ")")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv6
        for i in range(self.args.M):
            exec("self.m6" + str(
                i) + " = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256),nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256))")
            exec("self.conv6.append(self.m6" + str(i) + ")")
        exec("self.m6" + str(
            "x") + " = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256),nn.ReLU())")
        exec("self.conv6.append(self.m6" + str("x") + ")")

        # conv7
        for i in range(self.args.M):
            exec("self.m7" + str(
                i) + " = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256),nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256))")
            exec("self.conv7.append(self.m7" + str(i) + ")")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv8
        for i in range(self.args.M):
            exec("self.m8" + str(
                i) + " = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512))")
            exec("self.conv8.append(self.m8" + str(i) + ")")
        exec("self.m8" + str(
            "x") + " = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU())")
        exec("self.conv8.append(self.m8" + str("x") + ")")

        # conv9
        for i in range(self.args.M):
            exec("self.m9" + str(
                i) + " = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512))")
            exec("self.conv9.append(self.m9" + str(i) + ")")

        # avg pool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if len(self.final_layers) < 1:
            if self.args.return_task_id:    # task-IL
                for i in range(self.args.num_train_task):
                    exec(f"self.final_layer{i+1} = nn.Linear(512, self.args.class_per_task)")    # 10
                    exec(f"self.final_layers.append(self.final_layer{i+1})")

                exec(f"self.final_layer_test = nn.Linear(512, self.args.num_test_class)")  # 10
                exec(f"self.final_layers.append(self.final_layer_test)")

                # for i in range(self.args.num_train_task, self.args.num_train_task + self.args.num_test_task):
                #     exec(f"self.final_layer{i+1} = nn.Linear(512, self.args.num_test_class)")    # 10
                #     exec(f"self.final_layers.append(self.final_layer{i+1})")
            else:    # class-IL
                exec(f"self.final_layer1 = nn.Linear(512, self.args.class_per_task*self.args.num_train_task)")    # 100
                exec(f"self.final_layers.append(self.final_layer1)")

                exec(f"self.final_layer_test = nn.Linear(512, self.args.num_test_class)")  # 10
                exec(f"self.final_layers.append(self.final_layer_test)")

                # for i in range(1, 1+self.args.num_test_task):
                #     exec(f"self.final_layer{i+1} = nn.Linear(512, self.args.num_test_class)")    # 10
                #     exec(f"self.final_layers.append(self.final_layer{i+1})")

        self.cuda()

    def freeze_feature_extractor(self):
        params_set = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5,
                      self.conv6, self.conv7, self.conv8, self.conv9]
        for j, params in enumerate(params_set):
            for i, param in enumerate(params):
                param.requires_grad = False

    def forward(self, x, path, last):

        M = self.args.M
        div = 1
        p = 0.5

        y = self.conv1[0](x)
        for j in range(1, self.args.M):
            if (path[0][j] == 1):
                y += self.conv1[j](x)
        x = F.relu(y)

        y = self.conv2[0](x)
        for j in range(1, self.args.M):
            if (path[1][j] == 1):
                y += self.conv2[j](x)
        x = y + x
        x = F.relu(x)

        y = self.conv3[0](x)
        for j in range(1, self.args.M):
            if (path[2][j] == 1):
                y += self.conv3[j](x)
        x = y + x
        x = F.relu(x)
        x = self.pool1(x)

        y = self.conv4[-1](x)
        for j in range(self.args.M):
            if (path[3][j] == 1):
                y += self.conv4[j](x)
        x = y
        x = F.relu(x)

        y = self.conv5[0](x)
        for j in range(1, self.args.M):
            if (path[4][j] == 1):
                y += self.conv5[j](x)
        x = y + x
        x = F.relu(x)
        x = self.pool2(x)

        y = self.conv6[-1](x)
        for j in range(self.args.M):
            if (path[5][j] == 1):
                y += self.conv6[j](x)
        x = y
        x = F.relu(x)

        y = self.conv7[0](x)
        for j in range(1, self.args.M):
            if (path[6][j] == 1):
                y += self.conv7[j](x)
        x = y
        x = F.relu(x)
        x = self.pool3(x)

        y = self.conv8[-1](x)
        for j in range(self.args.M):
            if (path[7][j] == 1):
                y += self.conv8[j](x)
        x = y
        x = F.relu(x)

        y = self.conv9[0](x)
        for j in range(1, self.args.M):
            if (path[8][j] == 1):
                y += self.conv9[j](x)
        x = y + x
        x = F.relu(x)
        x = self.pool4(x)

        # x = F.avg_pool2d(x, (7, 7), stride=(1, 1))
        x = self.avgpool(x)
        x = x.view(-1, 512)

        if type(last) is int:
            x = self.final_layers[last](x)
        else:
            o = []
            for x_idx in range(x.shape[0]):
                if last[x_idx] >= len(self.final_layers):  # for few-shot test
                    idx = -1
                else:       # for continual train
                    idx = last[x_idx]
                o.append(self.final_layers[idx](x[x_idx: x_idx+1]))   # forward classifier sample by sample
            x = torch.cat(o)

        return x


class RPS_net(nn.Module):

        def __init__(self, args):
            super(RPS_net, self).__init__()
            self.args = args
            self.final_layers = []
            self.init(None)

        def init(self, best_path):


            """Initialize all parameters"""
            self.conv1 = []
            self.conv2 = []
            self.conv3 = []
            self.conv4 = []
            self.conv5 = []
            self.conv6 = []
            self.conv7 = []
            self.conv8 = []
            self.conv9 = []
            self.fc1 = []


            # conv1
            for i in range(self.args.M):
                exec("self.m1" + str(i) + " = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),nn.BatchNorm2d(64),nn.ReLU())")
                exec("self.conv1.append(self.m1" + str(i) + ")")


            # conv2
            for i in range(self.args.M):
                exec("self.m2" + str(i) + " = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64))")
                exec("self.conv2.append(self.m2" + str(i) + ")")

            # conv3
            for i in range(self.args.M):
                exec("self.m3" + str(i) + " = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64))")
                exec("self.conv3.append(self.m3" + str(i) + ")")
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)


            
            
            

            # conv4
            for i in range(self.args.M):
                exec("self.m4" + str(i) + " = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128))")
                exec("self.conv4.append(self.m4" + str(i) + ")")
            exec("self.m4" + str("x") + " = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU())")
            exec("self.conv4.append(self.m4" + str("x") + ")")

            # conv5
            for i in range(self.args.M):
                exec("self.m5" + str(i) + " = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128))")
                exec("self.conv5.append(self.m5" + str(i) + ")")
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
              
                

            # conv6
            for i in range(self.args.M):
                exec("self.m6" + str(i) + " = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256),nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256))")
                exec("self.conv6.append(self.m6" + str(i) + ")")
            exec("self.m6" + str("x") + " = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256),nn.ReLU())")
            exec("self.conv6.append(self.m6" + str("x") + ")")

            # conv7
            for i in range(self.args.M):
                exec("self.m7" + str(i) + " = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256),nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256))")
                exec("self.conv7.append(self.m7" + str(i) + ")")
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            
            

            # conv8
            for i in range(self.args.M):
                exec("self.m8" + str(i) + " = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512))")
                exec("self.conv8.append(self.m8" + str(i) + ")")
            exec("self.m8" + str("x") + " = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU())")
            exec("self.conv8.append(self.m8" + str("x") + ")")

            # conv9
            for i in range(self.args.M):
                exec("self.m9" + str(i) + " = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512))")
                exec("self.conv9.append(self.m9" + str(i) + ")")

            if len(self.final_layers) < 1:
                self.final_layer1 = nn.Linear(512, self.args.num_class)
                self.final_layers.append(self.final_layer1)

            
            self.cuda()

        def forward(self, x, path, last):

            M = self.args.M
            div =1
            p=0.5
            
            y = self.conv1[0](x)
            for j in range(1,self.args.M):
                if(path[0][j]==1):
                    y += self.conv1[j](x)
            x = F.relu(y)

            
            
            
            
        
            y = self.conv2[0](x)
            for j in range(1,self.args.M):
                if(path[1][j]==1):
                    y += self.conv2[j](x)
            x = y + x
            x = F.relu(x)
            
            y = self.conv3[0](x)
            for j in range(1,self.args.M):
                if(path[2][j]==1):
                    y += self.conv3[j](x)
            x = y + x
            x = F.relu(x)
            x = self.pool1(x)

            
            
            
            y = self.conv4[-1](x)
            for j in range(self.args.M):
                if(path[3][j]==1):
                    y += self.conv4[j](x)
            x = y
            x = F.relu(x)

            y = self.conv5[0](x)
            for j in range(1, self.args.M):
                if(path[4][j]==1):
                    y += self.conv5[j](x)
            x = y + x
            x = F.relu(x)
            x = self.pool2(x)
            
            
            
            
            
            y = self.conv6[-1](x)
            for j in range(self.args.M):
                if(path[5][j]==1):
                    y += self.conv6[j](x)
            x = y 
            x = F.relu(x)

            y = self.conv7[0](x)
            for j in range(1, self.args.M):
                if(path[6][j]==1):
                    y += self.conv7[j](x)
            x = y
            x = F.relu(x)
            x = self.pool3(x)
            
            
            
            
            
            y = self.conv8[-1](x)
            for j in range(self.args.M):
                if(path[7][j]==1):
                    y += self.conv8[j](x)
            x = y 
            x = F.relu(x)

            y = self.conv9[0](x)
            for j in range(1, self.args.M):
                if(path[8][j]==1):
                    y += self.conv9[j](x)
            x = y + x
            x = F.relu(x)
            x = self.pool4(x)
            

            x = F.avg_pool2d(x, (7,7), stride=(1,1))
            x = x.view(-1, 512)
            x = self.final_layers[last](x)
            
            return x
        
        
        
class RPS_net_cifar(nn.Module):

        def __init__(self, args):
            super(RPS_net_cifar, self).__init__()
            self.args = args
            self.final_layers = []
            self.init(None)

        def init(self, best_path):


            """Initialize all parameters"""
            self.conv1 = []
            self.conv2 = []
            self.conv3 = []
            self.conv4 = []
            self.conv5 = []
            self.conv6 = []
            self.conv7 = []
            self.conv8 = []
            self.conv9 = []
            self.fc1 = []

            div = 1
            a1 = 64//div
            
            a2 = 64//div
            a3 = 128//div
            a4 = 256//div
            a5 = 512//div
            
            self.a5 =a5
            # conv1
            for i in range(self.args.M):
                exec("self.m1" + str(i) + " = nn.Sequential(nn.Conv2d(3, "+str(a1)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a1)+"),nn.ReLU())")
                exec("self.conv1.append(self.m1" + str(i) + ")")


            # conv2
            for i in range(self.args.M):
                exec("self.m2" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a1)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a2)+"),nn.ReLU(), nn.Conv2d("+str(a2)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a1)+"))")
                exec("self.conv2.append(self.m2" + str(i) + ")")

            # conv3
            for i in range(self.args.M):
                exec("self.m3" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a2)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a2)+"),nn.ReLU(), nn.Conv2d("+str(a2)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a2)+"))")
                exec("self.conv3.append(self.m3" + str(i) + ")")
            


            # conv4
            for i in range(self.args.M):
                exec("self.m4" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a2)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"),nn.ReLU(), nn.Conv2d("+str(a3)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"))")
                exec("self.conv4.append(self.m4" + str(i) + ")")
            exec("self.m4" + str("x") + " = nn.Sequential(nn.Conv2d("+str(a2)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"),nn.ReLU())")
            exec("self.conv4.append(self.m4" + str("x") + ")")

            # conv5
            for i in range(self.args.M):
                exec("self.m5" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a3)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"),nn.ReLU(), nn.Conv2d("+str(a3)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"))")
                exec("self.conv5.append(self.m5" + str(i) + ")")
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
              
                

            # conv6
            for i in range(self.args.M):
                exec("self.m6" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a3)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"),nn.ReLU(), nn.Conv2d("+str(a4)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"))")
                exec("self.conv6.append(self.m6" + str(i) + ")")
            exec("self.m6" + str("x") + " = nn.Sequential(nn.Conv2d("+str(a3)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"),nn.ReLU())")
            exec("self.conv6.append(self.m6" + str("x") + ")")

            # conv7
            for i in range(self.args.M):
                exec("self.m7" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a4)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"),nn.ReLU(), nn.Conv2d("+str(a4)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"))")
                exec("self.conv7.append(self.m7" + str(i) + ")")
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            
            

            # conv8
            for i in range(self.args.M):
                exec("self.m8" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a4)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"),nn.ReLU(), nn.Conv2d("+str(a5)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"))")
                exec("self.conv8.append(self.m8" + str(i) + ")")
            exec("self.m8" + str("x") + " = nn.Sequential(nn.Conv2d("+str(a4)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"),nn.ReLU())")
            exec("self.conv8.append(self.m8" + str("x") + ")")

            # conv9
            for i in range(self.args.M):
                exec("self.m9" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a5)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"),nn.ReLU(), nn.Conv2d("+str(a5)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"))")
                exec("self.conv9.append(self.m9" + str(i) + ")")
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            
            
            if len(self.final_layers) < 1:
                self.final_layer1 = nn.Linear(a5, 100)
                self.final_layers.append(self.final_layer1)

            
            self.cuda()

        def forward(self, x, path, last):

            M = self.args.M
            div =1
            p=0.5
            
            y = self.conv1[0](x)
            for j in range(1,self.args.M):
                if(path[0][j]==1):
                    y += self.conv1[j](x)
            x = F.relu(y)

            
            
            
            
        
            y = self.conv2[0](x)
            for j in range(1,self.args.M):
                if(path[1][j]==1):
                    y += self.conv2[j](x)
            x = y + x
            x = F.relu(x)
            
            y = self.conv3[0](x)
            for j in range(1,self.args.M):
                if(path[2][j]==1):
                    y += self.conv3[j](x)
            x = y + x
            x = F.relu(x)


            
            
            
            y = self.conv4[-1](x)
            for j in range(self.args.M):
                if(path[3][j]==1):
                    y += self.conv4[j](x)
            x = y
            x = F.relu(x)
            
            y = self.conv5[0](x)
            for j in range(1, self.args.M):
                if(path[4][j]==1):
                    y += self.conv5[j](x)
            x = y + x
            x = F.relu(x)
            x = self.pool1(x)  
            
            
            
            
            
            y = self.conv6[-1](x)
            for j in range(self.args.M):
                if(path[5][j]==1):
                    y += self.conv6[j](x)
            x = y 
            x = F.relu(x)
            
            y = self.conv7[0](x)
            for j in range(1, self.args.M):
                if(path[6][j]==1):
                    y += self.conv7[j](x)
            x = y
            x = F.relu(x)
            x = self.pool2(x)
            
            
            
            
            
            y = self.conv8[-1](x)
            for j in range(self.args.M):
                if(path[7][j]==1):
                    y += self.conv8[j](x)
            x = y 
            x = F.relu(x)
            
            y = self.conv9[0](x)
            for j in range(1, self.args.M):
                if(path[8][j]==1):
                    y += self.conv9[j](x)
            x = y + x
            x = F.relu(x)
#             x = self.pool3(x)
            

            x = F.avg_pool2d(x, (8, 8), stride=(1,1))
            x = x.view(-1, self.a5)
            x = self.final_layers[last](x)
            
            return x
        
        
class RPS_net_mlp(nn.Module):

        def __init__(self, args):
            super(RPS_net_mlp, self).__init__()
            self.args = args
            self.final_layers = []
            self.init(None)

        def init(self, best_path):


            """Initialize all parameters"""

            self.mlp1 = []
            self.mlp2 = []
            self.fc1 = []

            #mlp1
            for i in range(self.args.M):
#                 exec("self.m1" + str(i) + " = nn.Sequential(nn.Linear(784, 256),nn.BatchNorm2d(256),nn.ReLU())")
                exec("self.m1" + str(i) + " = nn.Linear(784, 400)")
                exec("self.mlp1.append(self.m1" + str(i) + ")")
                
                
            #mlp2
            for i in range(self.args.M):
                exec("self.m2" + str(i) + " = nn.Linear(400, 128)")
                exec("self.mlp2.append(self.m2" + str(i) + ")")
                

            if len(self.final_layers) < 1:
                self.final_layer1 = nn.Linear(128, 10)
                self.final_layers.append(self.final_layer1)
            
            self.cuda()

        def forward(self, x, path, last):

            M = self.args.M
           
            y = self.mlp1[0](x)
            for j in range(1,self.args.M):
                if(path[0][j]==1):
                    y += self.mlp1[j](x)
            x = F.relu(y)

            
            
            y = self.mlp2[0](x)
            for j in range(1,self.args.M):
                if(path[1][j]==1):
                    y += self.mlp2[j](x)
            x = F.relu(y)
            
            x = self.final_layers[last](x)
            
            return x