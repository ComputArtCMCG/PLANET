import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader,RandomSampler
from PLANET_datautils import ProLigDataset
from PLANET_model import PLANET
import argparse,math,rdkit,os,sys,pickle
import numpy as np
from rdkit import RDLogger

if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--train', required=True)
    parser.add_argument('-v','--valid',required=True)
    parser.add_argument('-te','--test',required=True)
    parser.add_argument('-d','--save_dir', required=True)

    parser.add_argument('-s','--feature_dims', type=int, default=300)
    parser.add_argument('-n','--nheads', type=int, default=8)
    parser.add_argument('-k','--key_dims', type=int, default=300)
    parser.add_argument('-va','--value_dims', type=int, default=300)

    parser.add_argument('-pu','--pro_update_inters', type=int, default=3)
    parser.add_argument('-lu','--lig_update_iters',type=int,default=10)
    parser.add_argument('-pl','--pro_lig_update_iters',type=int,default=1)
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip_norm', type=float, default=200.0)

    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--batch_size',type=int,default=16)
    #parser.add_argument('--factor', type=float, default=0.3)  #optimizer lr factor
    parser.add_argument('--anneal_iter', type=int, default=20000)  #optimizer patience

    parser.add_argument('--load_epoch',type=int,default=0)
    parser.add_argument('--initial_step',type=int,default=0)

    parser.add_argument('--save_iter', type=int, default=5000)
    parser.add_argument('--print_iter', type=int, default=200)

    args = parser.parse_args()
    print(args)

    valid_dataset = ProLigDataset(args.valid,args.batch_size,shuffle=False,decoy_flag=False)
    valid_loader = DataLoader(valid_dataset,batch_size=1,shuffle=False,num_workers=4,drop_last=False,collate_fn=lambda x:x[0])

    test_dataset = ProLigDataset(args.test,args.batch_size,shuffle=False,decoy_flag=False)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=4,drop_last=False,collate_fn=lambda x:x[0])


    feature_dims,nheads,key_dims,value_dims,pro_update_inters,lig_update_iters,pro_lig_update_iters = args.feature_dims,args.nheads,args.key_dims,args.value_dims,\
        args.pro_update_inters,args.lig_update_iters,args.pro_lig_update_iters
    PLANET = PLANET(feature_dims,nheads,key_dims,value_dims,pro_update_inters,lig_update_iters,pro_lig_update_iters,'cuda').cuda()

    for param in PLANET.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_uniform_(param)

    print ("Model #Params: {:d}K".format(sum([x.nelement() for x in PLANET.parameters()]) // 1000))
    print(PLANET)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, PLANET.parameters()), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer,0.8)
    total_step = args.initial_step
    meters = np.zeros(6)
    affinity_all,lr_drop = False,False  #flag for fix param
    beta = 0

    PLANET.train()
    for epoch in range(1,1+args.epoch):
        train_dataset = ProLigDataset(args.train,args.batch_size,shuffle=True,decoy_flag=True)
        train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=4,drop_last=False,collate_fn=lambda x:x[0])
        
        for (res_feature_batch,mol_feature_batch,targets) in train_loader:
            optimizer.zero_grad()
            predictions = PLANET(res_feature_batch,mol_feature_batch)
            lig_interaction_loss,pro_lig_interaction_loss,affinity_loss = PLANET.compute_loss(predictions,targets,res_feature_batch,mol_feature_batch)
            total_loss = lig_interaction_loss +  pro_lig_interaction_loss +  beta * affinity_loss 
            total_loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, PLANET.parameters()), args.clip_norm)
            optimizer.step()

            lig_interaction_acc,pro_lig_interaction_acc,affinity_mae = PLANET.compute_metrics(predictions,targets)
            
            meters = meters + np.array([lig_interaction_loss.item(),lig_interaction_acc.item(),pro_lig_interaction_loss.item(),\
                pro_lig_interaction_acc.item(),affinity_loss.item(),affinity_mae.item()])
            total_step += 1

            
            if  total_step > 0 and total_step <= 500:
                beta = 0.0
            else:
                beta = 1.0

            if  total_step % args.print_iter == 0:
                meters /= args.print_iter
                print ("[{}]\tLig_L:{:.3f}\tLig_ACC:{:.3f}\tProLig_L:{:.3f}\tProLig_ACC:{:.3f}\tAffinity_L:{:.3f}\tMAE:{:.3f}" \
                    .format(total_step,meters[0], meters[1],meters[2],meters[3],meters[4],meters[5]))
                sys.stdout.flush()
                meters *= 0.

            
            if total_step > 60000 and total_step % args.anneal_iter == 0:
                scheduler.step()
                print ("[learning rate]: {:.6f}".format(scheduler.get_last_lr()[0]))
            '''
            if total_step > 40000 and total_step < 60000:
                PLANET.requires_grad_(requires_grad=False)
                PLANET.prolig.linear_affinity.requires_grad_(requires_grad=True)
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, PLANET.parameters()), lr=args.lr)
                affinity_all = True
                beta = 1

            if total_step >= 60000 and not lr_drop:
                PLANET.requires_grad_(requires_grad=True)
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, PLANET.parameters()))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-6
                affinity_all = False
                lr_drop = True
            '''

            if total_step > 0 and total_step % args.save_iter == 0:
                PLANET.eval()
                with torch.no_grad():
                    valid_meters = np.zeros(6)
                    valid_batch_count = 0            
                    for (res_batch,mol_batch,targets) in valid_loader:
                        try:
                            predictions = PLANET(res_batch,mol_batch)
                            lig_interaction_loss,pro_lig_interaction_loss,affinity_loss = PLANET.compute_loss(predictions,targets,res_batch,mol_batch)
                            lig_interaction_acc,pro_lig_interaction_acc,affinity_mae = PLANET.compute_metrics(predictions,targets)
                            valid_batch_count += 1
                            valid_meters = valid_meters + np.array([lig_interaction_loss.item(),lig_interaction_acc.item(),pro_lig_interaction_loss.item(),\
                                pro_lig_interaction_acc.item(),affinity_loss.item(),affinity_mae.item()])
                        except:
                            continue
                    valid_meters /= valid_batch_count
                    print ("[Valid_{}]\tLig_L:{:.3f}\tLig_ACC:{:.3f}\tProLig_L:{:.3f}\tProLig_ACC:{:.3f}\tAffinity_L:{:.3f}\tMAE:{:.3f}" \
                        .format(total_step,valid_meters[0], valid_meters[1],valid_meters[2],valid_meters[3],valid_meters[4],valid_meters[5]))

                torch.save(PLANET.state_dict(), args.save_dir + "/PLANET.iter-" + str(total_step))
                
                with torch.no_grad():
                    test_meters = np.zeros(6)
                    test_batch_count = 0            
                    for (res_batch,mol_batch,targets) in test_loader:
                        try:
                            predictions = PLANET(res_batch,mol_batch)
                            lig_interaction_loss,pro_lig_interaction_loss,affinity_loss = PLANET.compute_loss(predictions,targets,res_batch,mol_batch)
                            lig_interaction_acc,pro_lig_interaction_acc,affinity_mae = PLANET.compute_metrics(predictions,targets)
                            test_batch_count += 1
                            test_meters =  test_meters + np.array([lig_interaction_loss.item(),lig_interaction_acc.item(),pro_lig_interaction_loss.item(),\
                                pro_lig_interaction_acc.item(),affinity_loss.item(),affinity_mae.item()])
                        except:
                            continue
                    test_meters /= test_batch_count
                    print ("[Test_{}]\tLig_L:{:.3f}\tLig_ACC:{:.3f}\tProLig_L:{:.3f}\tProLig_ACC:{:.3f}\tAffinity_L:{:.3f}\tMAE:{:.3f}" \
                        .format(total_step,test_meters[0], test_meters[1],test_meters[2],test_meters[3],test_meters[4],test_meters[5]))
                PLANET.train()
        

