import os
# import shutil

import torch
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.optim as optim
import time
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb




class Learner():
    def __init__(self,model,args,trainloader,testloader,old_model,use_cuda, path, fixed_path, train_path, infer_path):
        self.model=model
        self.args=args
        self.title='cifar-100-' + self.args.arch
        self.trainloader=trainloader 
        self.use_cuda=use_cuda
        self.state= {key:value for key, value in self.args.__dict__.items() if not key.startswith('__') and not callable(key)} 
        self.best_acc = 0 
        self.testloader=testloader
        self.start_epoch=self.args.start_epoch
        self.test_loss=0.0
        self.path = path
        self.fixed_path = fixed_path
        self.train_path = train_path
        self.infer_path = infer_path
        self.test_acc=0.0
        self.train_loss, self.train_acc=0.0,0.0
        self.old_model=old_model
        if self.args.sess > 0: self.old_model.eval()


        trainable_params = []

        if self.args.sess < self.args.num_train_task:       # continual train
            if(self.args.dataset=="MNIST"):
                params_set = [self.model.mlp1, self.model.mlp2]
            elif self.args.arch == 'res-18':
                params_set = [self.model.conv1, self.model.conv2, self.model.conv3, self.model.conv4, self.model.conv5, self.model.conv6, self.model.conv7, self.model.conv8, self.model.conv9]
            elif self.args.arch == 'vit':
                assert self.args.L == 9
                params_set = [self.model.l1, self.model.l2, self.model.l3, self.model.l4, self.model.l5, self.model.l6, self.model.l7, self.model.l8, self.model.l9]
            else:
                raise Exception(f'unimplement arch {self.args.arch}')

            if self.args.arch == 'vit':      # encoder
                p = {'params': self.model.encoder.parameters()}
                trainable_params.append(p)

                if hasattr(self.model, 'attn1'):
                    p = {'params': self.model.attn1.parameters()}
                    trainable_params.append(p)
                    p = {'params': self.model.attn2.parameters()}
                    trainable_params.append(p)
                    p = {'params': self.model.attn3.parameters()}
                    trainable_params.append(p)
                    p = {'params': self.model.attn4.parameters()}
                    trainable_params.append(p)
                    p = {'params': self.model.attn5.parameters()}
                    trainable_params.append(p)
                    p = {'params': self.model.attn6.parameters()}
                    trainable_params.append(p)
                    p = {'params': self.model.attn7.parameters()}
                    trainable_params.append(p)
                    p = {'params': self.model.attn8.parameters()}
                    trainable_params.append(p)
                    p = {'params': self.model.attn9.parameters()}
                    trainable_params.append(p)

            for j, params in enumerate(params_set):
                for i, param in enumerate(params):
                    if(i==self.args.M):     # M_skip
                        p = {'params': param.parameters()}
                        trainable_params.append(p)
                    else:
                        if(self.train_path[j,i]==1):
                            p = {'params': param.parameters()}
                            trainable_params.append(p)
                        else:
                            # param.requires_grad = False
                            for p in param.parameters():
                                p.requires_grad = False

        if self.args.sess < self.args.num_train_task:       # continual train
            if self.args.return_task_id:        # task-IL
                for j in range(len(self.model.final_layers)):
                    if j < self.args.sess+1:
                        p = {'params': self.model.final_layers[j].parameters()}     # all trained classifiers since memory-based
                        trainable_params.append(p)
                    else:
                        # self.model.final_layers[j].requires_grad = False
                        for p in self.model.final_layers[j].parameters():
                            p.requires_grad = False
            else:       # class-IL
                assert len(self.model.final_layers) == 2
                p = {'params': self.model.final_layers[0].parameters()}  # classifier
                trainable_params.append(p)
                # self.model.final_layers[1].requires_grad = False
                for p in self.model.final_layers[1].parameters():  # classifier for fewshot test
                    p.requires_grad = False

        else:       # fewshot test
            # if self.args.return_task_id:        # task-IL
            #     offset = self.args.num_train_task
            # else:       # class-IL
            #     offset = 1
            #
            # if args.sess < args.num_train_task + 1 * args.num_test_task:  # sys
            #     sess_offset = args.num_train_task
            #     mode = 'sys'
            # elif args.sess < args.num_train_task + 2 * args.num_test_task:  # pro
            #     sess_offset = args.num_train_task + 1 * args.num_test_task
            #     mode = 'pro'
            # elif args.sess < args.num_train_task + 3 * args.num_test_task:  # sub
            #     sess_offset = args.num_train_task + 2 * args.num_test_task
            #     mode = 'sub'
            # elif args.sess < args.num_train_task + 4 * args.num_test_task:  # non
            #     sess_offset = args.num_train_task + 3 * args.num_test_task
            #     mode = 'non'
            # elif args.sess < args.num_train_task + 5 * args.num_test_task:  # noc
            #     sess_offset = args.num_train_task + 4 * args.num_test_task
            #     mode = 'noc'
            #
            # elif args.sess < args.num_train_task + 6 * args.num_test_task:  # sys    no freeze fe
            #     sess_offset = args.num_train_task + 5 * args.num_test_task
            #     mode = 'sys'
            # elif args.sess < args.num_train_task + 7 * args.num_test_task:  # pro
            #     sess_offset = args.num_train_task + 6 * args.num_test_task
            #     mode = 'pro'
            # elif args.sess < args.num_train_task + 8 * args.num_test_task:  # sub
            #     sess_offset = args.num_train_task + 7 * args.num_test_task
            #     mode = 'sub'
            # elif args.sess < args.num_train_task + 9 * args.num_test_task:  # non
            #     sess_offset = args.num_train_task + 8 * args.num_test_task
            #     mode = 'non'
            # elif args.sess < args.num_train_task + 10 * args.num_test_task:  # noc
            #     sess_offset = args.num_train_task + 9 * args.num_test_task
            #     mode = 'noc'
            # else:
            #     raise Exception(f'sess error: {args.sess}.')
            # p = {'params': self.model.final_layers[self.args.sess - sess_offset + offset].parameters()}

            p = {'params': self.model.final_layers[-1].parameters()}
            trainable_params.append(p)
            for j in range(len(self.model.final_layers)-1):
                # self.model.final_layers[j].requires_grad = False
                for p in self.model.final_layers[j].parameters():
                    p.requires_grad = False

        print("Number of layers being trained : " , len(trainable_params))

        print('    Total trainable params: %.2fM' % (
                sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1024 / 1024))

        
#         self.optimizer = optim.Adadelta(trainable_params)
#         self.optimizer = optim.SGD(trainable_params, lr=self.args.lr, momentum=0.96, weight_decay=0)
        self.optimizer = optim.Adam(trainable_params, lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        if hasattr(self.args, 'schedule_mode') and self.args.schedule_mode == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.epochs, 1e-6)
        else:
            self.scheduler = None



    def learn(self):
        if self.args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isfile(self.args.resume), 'Error: no checkpoint directory found!'
            self.args.checkpoint = os.path.dirname(self.args.resume)
            checkpoint = torch.load(self.args.resume)
            self.best_acc = checkpoint['best_acc']
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(self.args.checkpoint, 'log.txt'), title=self.title, resume=True)
        else:
            logger = Logger(os.path.join(self.args.checkpoint, 'session_'+str(self.args.sess)+'_'+str(self.args.test_case)+'_log.txt'), title=self.title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Best Acc'])
        if self.args.evaluate:
            print('\nEvaluation only')
            self.test(self.start_epoch)
            print(' Test Loss:  %.8f, Test Acc:  %.2f' % (self.test_loss, self.test_acc))
            return


        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.adjust_learning_rate(epoch)

            print('\nEpoch: [%d | %d] LR: %f Sess: %d' % (epoch + 1, self.args.epochs, self.state['lr'],self.args.sess))
            self.train(epoch, self.infer_path, -1)
            self.test(epoch, self.infer_path, -1)
            

            # append logger file
            logger.append([self.state['lr'], self.train_loss, self.test_loss, self.train_acc, self.test_acc, self.best_acc])

            # save model
            is_best = self.test_acc > self.best_acc
            self.best_acc = max(self.test_acc, self.best_acc)
            if not hasattr(self.args, 'num_train_task') or self.args.sess < self.args.num_train_task:
                # only training task need to store checkpoint
                self.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'acc': self.test_acc,
                        'best_acc': self.best_acc,
                        'optimizer' : self.optimizer.state_dict(),
                }, is_best, checkpoint=self.args.savepoint,filename='session_'+str(self.args.sess)+'_' + str(self.args.test_case)+'_checkpoint.pth.tar',session=self.args.sess, test_case=self.args.test_case)

            # early stop for 50 epochs
            if is_best:
                self.epochs_overfitting = epoch

            if self.epochs_overfitting <= epoch - 10:  # patience for 50 epochs
                print(f'early stop at epoch: {epoch}')
                break

        logger.close()
        logger.plot()
        savefig(os.path.join(self.args.checkpoint, 'log.eps'))

        print('Best acc:')
        print(self.best_acc)

    def train(self, epoch, path, last):
        # switch to train mode
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        if self.args.sess >= self.args.num_train_task:      # for novel testing
            num_class = self.args.num_test_class    # for task-IL and class-IL num_class = 10
            num_old_class = self.args.num_test_class if self.args.sess > self.args.num_train_task else self.args.class_per_task
            num_total_class = num_class
        else:   # continual training
            if self.args.return_task_id:        # task-IL
                num_class = self.args.class_per_task        # 10
                num_old_class = self.args.class_per_task
                num_total_class = self.args.class_per_task
            else:       # class-IL
                num_class = self.args.class_per_task * (self.args.sess + 1)
                num_old_class = self.args.class_per_task * self.args.sess
                num_total_class = self.args.class_per_task * self.args.num_train_task       # 100


        bar = Bar('Processing', max=len(self.trainloader))
        for batch_idx, (inputs, targets, task_ids) in enumerate(self.trainloader):
            # measure data loading time
            data_time.update(time.time() - end)

            # '''DEBUG'''
            # print(f'DEBUG targets: {targets}')
            # '''DEBUG'''
            targets = targets.long()
            targets_one_hot = torch.FloatTensor(inputs.shape[0], num_total_class)
            targets_one_hot.zero_()
            targets_one_hot.scatter_(1, targets[:,None], 1)

            # '''DEBUG'''
            # print(f'DEBUG targets_one_hot: {targets_one_hot}')
            # '''DEBUG'''



            if self.use_cuda:
                inputs, targets_one_hot,targets = inputs.cuda(), targets_one_hot.cuda(),targets.cuda()
            inputs, targets_one_hot,targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_one_hot),torch.autograd.Variable(targets)



            # compute output
            outputs = self.model(inputs, path, task_ids)
            preds=outputs.masked_select(targets_one_hot.eq(1))
            
            tar_ce=targets
            pre_ce=outputs.clone()

            pre_ce=pre_ce[:,0:num_class]

            loss =   F.cross_entropy(pre_ce,tar_ce)
            loss_dist = 0
            ## distillation loss
            if self.args.sess > 0 and self.args.sess < self.args.num_train_task:        # only for continual training
                outputs_old=self.old_model(inputs, path, task_ids)

                t_one_hot=targets_one_hot.clone()

                t_one_hot[:,0:num_old_class]=outputs_old[:,0:num_old_class]
                
                
                if(self.args.sess in range(1+self.args.jump)):
                    cx = 1
                else:
                    cx = self.args.rigidness_coff*(self.args.sess-self.args.jump)
                loss_dist = ( cx/self.args.train_batch*1.0)* torch.sum(F.kl_div(F.log_softmax(outputs/2.0,dim=1),F.softmax(t_one_hot/2.0,dim=1),reduce=False).clamp(min=0.0))

            loss+=loss_dist 



            # '''DEBUG'''
            # print(f'DEBUG outputs: {outputs.data}')
            # print(f'DEBUG targets: {targets.data}')
            # '''DEBUG'''

            # measure accuracy and record loss
            if(self.args.dataset=="MNIST"):
                prec1, prec5 = accuracy(output=outputs.data[:,0:num_class], target=targets.cuda().data, topk=(1, 1))
            elif(self.args.dataset=="SYSGQA"):
                prec1, prec5 = accuracy(output=outputs.data[:,0:num_class], target=targets.cuda().data, topk=(1, 1))
            elif(self.args.dataset=="SUBGQA"):
                prec1, prec5 = accuracy(output=outputs.data[:,0:num_class], target=targets.cuda().data, topk=(1, 1))
            else:
                prec1, prec5 = accuracy(output=outputs.data[:,0:num_class], target=targets.cuda().data, topk=(1, 1))
                # topk=(1, 5)
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))


            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) | Total: {total:} | Loss: {loss:.4f} | Dist: {loss_dist:.4f} | top1: {top1: .4f} | top5: {top5: .4f} '.format(
                        batch=batch_idx + 1,
                        size=len(self.trainloader),
#                         data=data_time.avg,
#                         bt=batch_time.avg,
                        total=bar.elapsed_td,
#                         eta=bar.eta_td,
                        loss=losses.avg,
                        loss_dist=loss_dist,
                        top1=top1.avg,
                        top5=top5.avg
                        )
            bar.next()
        bar.finish()
        self.train_loss,self.train_acc=losses.avg, top1.avg

        if self.scheduler is not None:
            self.scheduler.step()
   
    
    def test(self, epoch, path, last):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        if self.args.sess >= self.args.num_train_task:      # for novel testing
            num_class = self.args.num_test_class    # for task-IL and class-IL num_class = 10
            num_total_class = num_class
        else:   # continual training
            if self.args.return_task_id:        # task-IL
                num_class = self.args.class_per_task        # 10
                num_total_class = self.args.class_per_task
            else:       # class-IL
                num_class = self.args.class_per_task * (self.args.sess + 1)
                num_total_class = self.args.class_per_task * self.args.num_train_task       # 100


        bar = Bar('Processing', max=len(self.testloader))
        for batch_idx, (inputs, targets, task_ids) in enumerate(self.testloader):
            # measure data loading time
            data_time.update(time.time() - end)


            targets = targets.long()
            targets_one_hot = torch.FloatTensor(inputs.shape[0], num_total_class)
            targets_one_hot.zero_()
            targets_one_hot.scatter_(1, targets[:,None], 1)

            if self.use_cuda:
                inputs, targets_one_hot,targets = inputs.cuda(), targets_one_hot.cuda(),targets.cuda()
            inputs, targets_one_hot, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_one_hot) ,torch.autograd.Variable(targets)


            

            outputs = self.model(inputs, path, task_ids)

            loss = F.cross_entropy(outputs, targets)



            # measure accuracy and record loss
            if(self.args.dataset=="MNIST"):
                prec1, prec5 = accuracy(output=outputs.data[:,0:num_class], target=targets.cuda().data, topk=(1, 1))
            elif(self.args.dataset=="SYSGQA"):
                prec1, prec5 = accuracy(output=outputs.data[:,0:num_class], target=targets.cuda().data, topk=(1, 1))
            elif(self.args.dataset=="SUBGQA"):
                prec1, prec5 = accuracy(output=outputs.data[:,0:num_class], target=targets.cuda().data, topk=(1, 1))
            else:
                prec1, prec5 = accuracy(output=outputs.data[:,0:num_class], target=targets.cuda().data, topk=(1, 1))
                # topk=(1, 5)
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size})  Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(self.testloader),
#                         data=data_time.avg,
#                         bt=batch_time.avg,
                        total=bar.elapsed_td,
#                         eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg
                        )
            bar.next()
        bar.finish()
        self.test_loss= losses.avg;self.test_acc= top1.avg

    def save_checkpoint(self,state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar',session=0, test_case=0):
#         filepath = os.path.join(checkpoint, filename)
#         torch.save(state, filepath)
        if is_best:
            torch.save(state, os.path.join(checkpoint, 'session_'+str(session)+'_'+str(test_case)+'_model_best.pth.tar'))
#             shutil.copyfile(filepath, os.path.join(checkpoint, 'session_'+str(session)+'_'+str(test_case)+'_model_best.pth.tar') )

    def adjust_learning_rate(self, epoch):
        if self.scheduler is None and epoch in self.args.schedule:       # only for step schedule
            self.state['lr'] *= self.args.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.state['lr']
        elif self.scheduler is not None:
            self.state['lr'] = self.scheduler.get_last_lr()[0]


    def get_confusion_matrix(self, path):
        
        confusion_matrix = torch.zeros(100, 100)
        with torch.no_grad():
            for i, (inputs, targets, task_ids) in enumerate(self.testloader):
                inputs = inputs.cuda()
                targets = targets.long()
                targets = targets.cuda()
                outputs = self.model(inputs, path, task_ids)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(targets.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

        print(confusion_matrix)
        return confusion_matrix

