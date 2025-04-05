from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom,  Dataset_inflow_hour,Dataset_PV_minute, Dataset_PEMS_minute
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'bike_inflow':Dataset_inflow_hour,
    'bike_outflow':Dataset_inflow_hour,
    'inflow_didi':Dataset_inflow_hour,
    'outflow_didi':Dataset_inflow_hour,    
    'PV': Dataset_inflow_hour,
    'WT': Dataset_inflow_hour,
    'PEMS03':Dataset_PEMS_minute,
    'PEMS04':Dataset_PEMS_minute,
    'PEMS07':Dataset_PEMS_minute,
    'PEMS08':Dataset_PEMS_minute,
    'BJ':Dataset_PEMS_minute,
    'METR':Dataset_PEMS_minute,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True

        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        # seasonal_patterns=args.seasonal_patterns
    )
    # print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
