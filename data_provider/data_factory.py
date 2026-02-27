from data_provider.data_loader import Dataset_Electricity, Dataset_Sunspot, Dataset_CDD_HDD
from torch.utils.data import DataLoader

data_dict = {
    'electricity': Dataset_Electricity,
    'sunspot':Dataset_Sunspot,
    'cdd_hdd':Dataset_CDD_HDD
}


def data_provider(args, flag):
    """
    Provides dataset and DataLoader for time series prediction tasks.

    Args:
        args: An object containing various configuration parameters.
        flag (str): Indicates the type of dataset split ('train', 'val', 'test').

    Returns:
        tuple: A tuple containing the dataset object and the DataLoader object.
    """
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq
    print(f"Loading data for prediction task: {args.data} data for {flag} flag.")
    
    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=args.scale,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )
    print(f"{flag} set size: {len(data_set)}")

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader