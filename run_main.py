import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import gc

gc.collect()
torch.cuda.empty_cache()

from models import Autoformer, DLinear, TimeLLM, ST_SepNet, baselines32, TimeLLM_maxmin,TimeLLM_navie

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content, vali_save

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')


parser.add_argument('--model_comment', type=str, required=False, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=False, default='pool',
                    help='model name, options: [Autoformer, DLinear, pool, mtgnn, agcrn, stgcn, TGCN]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=False, default='outflow', help='dataset type')
parser.add_argument('--root_path', type=str, default='dataset/Bike', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='outflow.csv', help='data file')
parser.add_argument('--adjacency_path', type=str, default='adj.csv', help='data file')

# dataloader: datetime to the data split
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')

parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='5min',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')


# forecasting task
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')


# model define
parser.add_argument('--enc_in', type=int, default=295, help='encoder input size') #default 7
parser.add_argument('--dec_in', type=int, default=295, help='decoder input size') #default 7
parser.add_argument('--c_out', type=int, default=295, help='output size')   #default 7
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')  
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')


parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='learned',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='deepseek2b', help='LLM model')  # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='1536', help='LLM model dimension') # LLama7b:4096; GPT2-small:768; BERT-base:768
parser.add_argument('--node_num', type=int, default=295, help = 'the node number of the network ')  # Bike 295 PV 69 WT  66   DIDI 166
parser.add_argument('--fusion_gate', type=str, default='hyperstgnn', help='style of module fusion')
# optimization

parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')



parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')  #LLAMA 32  GPT2 256
parser.add_argument('--eval_batch_size', type=int, default=16, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')



parser.add_argument('--learning_rate', type=float, default=0.00005, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='multstep', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)


parser.add_argument('--gcn_true',type=bool, default=False, help='Whether to use GCN module')
parser.add_argument('--hgcn_true',type=bool, default=True,  help='Whether to use HGCN module')
parser.add_argument('--hgat_true', type=bool, default=True, help='Whether to use HGAT module')
parser.add_argument('--scale_hyperedges', type=int, default=3)
parser.add_argument('--alpha', type=float, default=0.1, help = 'use adjustable parameter to control hyperSTLLM or STLLM')
parser.add_argument('--beta', type=float, default=0.2, help = 'use adjustable parameter to control hyperSTLLM or STLLM')
parser.add_argument('--gamma', type=float, default=0.5, help = 'use adjustable parameter to control hyperSTLLM or STLLM')



args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='/ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)


    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    args.min_values = np.min(train_loader.dataset.data_x,0)
    args.max_values = np.max(train_loader.dataset.data_x,0)


    baselines_models = ['mtgnn','agcrn','stgcn','dmstgcn','gmsdr','mstgcn','stsgcn','gman','TGCN']

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    elif args.model == 'pool':
        model = ST_SepNet.Model(args).float()
    elif args.model == 'maxmin':
        model = TimeLLM_maxmin.Model(args).float()    
    elif args.model in baselines_models:
        model = baselines32.Model(args).float()    
    else:
        model = TimeLLM_navie.Model(args).float()
        

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)

    elif args.lradj == 'multstep':  #lr_decay_step = [5,20,40,70] 
        lr_decay_steps = [5,20,40,70]  
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=model_optim,
                                                            milestones=lr_decay_steps,
                                                            gamma=0.3)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=train_steps,
                                        pct_start=args.pct_start,
                                        epochs=args.train_epochs,
                                        max_lr=args.learning_rate)
    

    criterion = nn.MSELoss()   #MSE loss
    mae_metric = nn.L1Loss()   #MAE
    criterion_rmse = lambda y_pred, y_true: torch.sqrt(nn.MSELoss()(y_pred, y_true))  # RMSE loss
    criterion_mape = lambda y_pred, y_true: torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0  # MAPE loss

    

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)



    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    saved = False

    preds = []
    trues = []

    print(f'{args.llm_model}   {args.data}     Traning----------------------------------------')
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                
                
                # inverse transform
                outputs_inverse = outputs.reshape(args.batch_size*args.seq_len,args.node_num)
                batch_y_inverse = batch_y.reshape(args.batch_size*args.seq_len,args.node_num)

                outputs_inverse = test_data.inverse_transform(outputs_inverse.detach().cpu().numpy())
                batch_y_inverse = test_data.inverse_transform(batch_y_inverse.detach().cpu().numpy())

                # outputs_inverse = outputs_inverse.to(torch.bffloat16)
                # batch_y_inverse = batch_y_inverse.to(torch.bffloat16)
                preds.append(outputs_inverse)
                trues.append(batch_y_inverse)


            if (i + 1) % 10 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        
        # Loss
        train_loss = np.average(train_loss)
        vali_loss, vali_mae_loss, vali_mape_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric, criterion_mape)
        test_loss, test_mae_loss, test_mape_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric, criterion_mape)


        torch.cuda.empty_cache()
        # Output the error of each step, each predicted step is an error
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} | Vali Loss: {2:.7f} | Vali MAE Loss: {3:.7f} | Vali MAPE Loss: {4:.7f} | Test Loss: {5:.7f} | Test MAE Loss: {6:.7f} | Test MAPE Loss: {7:.7f}".format(
                epoch + 1, train_loss, vali_loss, vali_mae_loss, vali_mape_loss, test_loss, test_mae_loss, test_mape_loss))
        
        early_stopping(vali_loss, model, path)
        
        # Save the data model stops early
        if early_stopping.early_stop:
            accelerator.print("Early stopping")

            np.save(f'results/data_pred/{args.data}_{args.llm_model}_pred.npy', preds)
            np.save(f'results/data_pred/{args.data}_{args.llm_model}_true.npy', trues)
            vali_save(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
            saved = True
            break
        
        scheduler.step()    

    torch.cuda.empty_cache()
    checkpoint_path = path + '/' + 'checkpoint'
    state_dict = torch.load(checkpoint_path)
    if args.model in baselines_models:
        model.load_state_dict(state_dict)
        
    elif args.model == 'DLinear':
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('module.', '')   
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
    else:
        new_state_dict = {}
        for key in state_dict:
            new_key = 'module.' + key   
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
    model.eval()
    vali_save(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
accelerator.wait_for_everyone()

if accelerator.is_local_main_process:
    path = './checkpoints'  # unique checkpoint saving path
    del_files(path)  # delete checkpoint files
    accelerator.print('success delete checkpoints')