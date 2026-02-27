import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
from utils.augmentation import run_augmentation_single
from utils.timefeatures import time_features
warnings.filterwarnings('ignore')

class Dataset_Electricity(Dataset):
    def __init__(self, args, root_path, flag, size,
                 features, data_path,
                 target, scale, timeenc=1, freq='h',
                 seasonal_patterns=None):
        """
        initialize the Electricity dataset class.

        Args:
            args: an object containing various configuration parameters, such as data augmentation settings.
            root_path (str): data file root path.
            flag (str): the type of dataset ('train', 'val', 'test').
            size (list, optional): [seq_len, label_len, pred_len].
                seq_len: the length of the input sequence.
                label_len: the length of the label sequence (for decoder input).
                pred_len: the length of the prediction sequence.
            features (str): the type of features ('S' for univariate, 'M' or 'MS' for multivariate).
            data_path (str): the name of the data file.
            target (str): the name of the target variable.
            scale (bool): whether to scale the data.
            timeenc (int): the type of time encoding (0: discrete features, 1: continuous periodic features).
            freq (str): the frequency of the data ('h' for hourly, 'd' for daily etc.).
        """
        self.args = args

        if size is None:
            self.seq_len = 24 * 4 * 4  
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]


        assert flag in ['train', 'test', 'val'], f"Flag must be 'train', 'test', or 'val', but got {flag}"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        """
        read the original data, standardize it, and divide it into training/validation/test sets according to flag.
        extract time features.
        """
        self.scaler = StandardScaler()
        df_raw = pd.read_excel(os.path.join(self.root_path, self.data_path))
        df_raw['date'] = pd.to_datetime(df_raw['date'], format='%d.%m.%Y %H:%M', errors='raise')
        assert self.target in df_raw.columns, f"Target '{self.target}' not in columns: {list(df_raw.columns)}"


        num_rows = len(df_raw)
        train_border = int(num_rows * 0.6)
        val_border = int(num_rows * 0.8)

        if self.set_type == 0:
            border1 = 0
            border2 = train_border
        elif self.set_type == 1:
            border1 = train_border - self.seq_len
            border2 = val_border
        else:  
            border1 = val_border - self.seq_len
            border2 = num_rows

        print(f"Loading data for {['train', 'val', 'test'][self.set_type]} set from {border1} to {border2} (total_rows: {num_rows})")

        df_data = df_raw[[self.target]]

        if self.scale:
            train_data_for_scaler = df_data[0:train_border].values
            self.scaler.fit(train_data_for_scaler)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
            print(f"Data augmentation applied to training set: {augmentation_tags}")

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        get a sliding window sample based on the index.

        Args:
            index (int): the starting index of the sample.

        Returns:
            tuple: contains (seq_x, seq_y, seq_x_mark, seq_y_mark) four NumPy arrays.
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end].astype('float32')
        seq_y = self.data_y[r_begin:r_end].astype('float32')
        seq_x_mark = self.data_stamp[s_begin:s_end].astype('float32')
        seq_y_mark = self.data_stamp[r_begin:r_end].astype('float32')

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        return the total number of sliding window samples available in the dataset.
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        convert the standardized data back to the original scale.

        Args:
            data (numpy.ndarray): the standardized data.

        Returns:
            numpy.ndarray: the data converted back to the original scale.
        """
        return self.scaler.inverse_transform(data)

class Dataset_Sunspot(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path=None,
                 target='sunspot', scale=True, timeenc=0, freq='m',
                 seasonal_patterns=None):
        """
        initialize the Sunspot dataset class.
        specifically designed to handle the Sunspot dataset, and maintain the same interface and output format as the ETT dataset class.

        Args:
            args: an object containing various configuration parameters, such as data augmentation settings.
            root_path (str): the root path of the data file.
            flag (str): the type of dataset ('train', 'val', 'test').
            size (list, optional): [seq_len, label_len, pred_len].
                seq_len: the length of the input sequence.
                label_len: the length of the label sequence (for decoder input).
                pred_len: the length of the prediction sequence.
            features (str): the type of features ('S' for univariate). this implementation forces 'S'.
            data_path (str): the name of the data file (default is 'sunspot.csv').
            target (str): the name of the target variable (default is 'sunspot').
            scale (bool): whether to scale the data.
            timeenc (int): the type of time encoding (0: discrete features, 1: continuous periodic features).
            freq (str): the frequency of the data (default is 'm' for monthly).
        """
        self.args = args
        if size is None:
            self.seq_len = 24 * 4 * 4 
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val'], f"Flag must be 'train', 'test', or 'val', but got {flag}"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = 'S'
        self.target = 'sunspot'
        self.scale = scale
        self.timeenc = timeenc
        self.freq = 'm'

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        """
        read the original Sunspot data, standardize it, and divide it into training/validation/test sets according to flag.
        combine the 'year' and 'month' columns to create the 'date' column, and extract time features.
        """
        self.scaler = StandardScaler()
        df_raw = pd.read_excel(os.path.join(self.root_path, self.data_path))
        
        df_raw['date'] = pd.to_datetime(df_raw['year'].astype(str) + '-' + \
                                        df_raw['month'].astype(str) + '-01')
        
        df_data = df_raw[['sunspot']]

        num_rows = len(df_raw)
        train_border = int(num_rows * 0.7)
        val_border = int(num_rows * 0.8)

        if self.set_type == 0:
            border1 = 0
            border2 = train_border
        elif self.set_type == 1:
            border1 = train_border - self.seq_len
            border2 = val_border
        else:
            border1 = val_border - self.seq_len
            border2 = num_rows

        print(f"Loading data for {['train', 'val', 'test'][self.set_type]} set from {border1} to {border2} (total_rows: {num_rows})")

        if self.scale:
            train_data_for_scaler = df_data[0:train_border].values
            self.scaler.fit(train_data_for_scaler)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
            print(f"Data augmentation applied to training set: {augmentation_tags}")

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
            get a sliding window sample based on the index.

        Args:
            index (int): the starting index of the sample.

        Returns:
            tuple: contains (seq_x, seq_y, seq_x_mark, seq_y_mark) four NumPy arrays.
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end].astype('float32')
        seq_y = self.data_y[r_begin:r_end].astype('float32')
        seq_x_mark = self.data_stamp[s_begin:s_end].astype('float32')
        seq_y_mark = self.data_stamp[r_begin:r_end].astype('float32')

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        return the total number of sliding window samples available in the dataset.
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        convert the standardized data back to the original scale.

        Args:
            data (numpy.ndarray): the standardized data.

        Returns:
            numpy.ndarray: the data converted back to the original scale.
        """
        return self.scaler.inverse_transform(data)

class Dataset_CDD_HDD(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path=None, # Default data file
                 target='HDDs', scale=True, timeenc=0, freq='d', # Default target, freq is 'd' for daily
                 seasonal_patterns=None):
        """
        Initializes the CDD/HDD Dataset class.
        Specifically designed to handle CDD/HDD datasets, maintaining consistency
        with ETT dataset class interface and output format.

        Args:
            args: Object containing various configuration parameters, such as data augmentation settings.
            root_path (str): Root directory path where the data file is located.
            flag (str): Dataset type ('train', 'val', 'test').
            size (list, optional): [seq_len, label_len, pred_len].
                seq_len: Length of the input sequence.
                label_len: Length of the label sequence (for decoder input).
                pred_len: Length of the prediction sequence.
            features (str): Feature type ('S' for univariate). This implementation forces 'S'.
            data_path (str): Filename of the CSV data file (defaults to 'cdd_hdd.csv').
            target (str): Specifies the column name to be used as the target variable ('HDDs' or 'CDDs').
            scale (bool): Whether to standardize the data.
            timeenc (int): Time feature encoding method (0: discrete features, 1: continuous periodic features).
            freq (str): Data frequency (defaults to 'd' for daily).
        """
        self.args = args
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val'], f"Flag must be 'train', 'test', or 'val', but got {flag}"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = 'S'
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = 'd'

        self.root_path = root_path
        self.data_path = data_path

        self.all_possible_targets = ['HDDs', 'CDDs']
        assert self.target in self.all_possible_targets, \
            f"Target '{self.target}' not found. Choose from {self.all_possible_targets}"

        self.__read_data__()

    def __read_data__(self):
        """
        Reads raw CDD/HDD data, standardizes it, and splits it into train/validation/test sets
        based on the flag. It also extracts time features from the 'DATE' column.
        """
        self.scaler = StandardScaler()
        df_raw = pd.read_excel(os.path.join(self.root_path, self.data_path))
                
        df_raw['DATE'] = pd.to_datetime(df_raw['DATE'], format='%d/%m/%Y')
        df_raw.rename(columns={'DATE': 'date'}, inplace=True) 
        df_data = df_raw[[self.target]]

        num_rows = len(df_raw)
        train_border = int(num_rows * 0.7)
        val_border = int(num_rows * 0.8)

        if self.set_type == 0:
            border1 = 0
            border2 = train_border
        elif self.set_type == 1:
            border1 = train_border - self.seq_len
            border2 = val_border
        else:
            border1 = val_border - self.seq_len
            border2 = num_rows

        print(f"Loading data for {['train', 'val', 'test'][self.set_type]} set from {border1} to {border2} (total_rows: {num_rows})")

        if self.scale:
            train_data_for_scaler = df_data[0:train_border].values
            self.scaler.fit(train_data_for_scaler)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
            print(f"Data augmentation applied to training set: {augmentation_tags}")

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        get a sliding window sample based on the index.

        Args:
            index (int): the starting index of the sample.

        Returns:
            tuple: contains (seq_x, seq_y, seq_x_mark, seq_y_mark) four NumPy arrays.
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end].astype('float32')
        seq_y = self.data_y[r_begin:r_end].astype('float32')
        seq_x_mark = self.data_stamp[s_begin:s_end].astype('float32')
        seq_y_mark = self.data_stamp[r_begin:r_end].astype('float32')

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        return the total number of available sliding window samples in the dataset.
        """
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        """
        convert the standardized data back to the original scale.

        Args:
            data (numpy.ndarray): the standardized data.

        Returns:
            numpy.ndarray: the data converted back to the original scale.
        """
        return self.scaler.inverse_transform(data)
