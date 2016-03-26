import pandas as pd
import os
from util import timethis


class DataLoader(object):
    """
    Load the data
    
    Five csv files must be provided in directory config.DATA_PATH. They are respectively
    'train.csv', 'test.csv', 'product_descriptions.csv', 'attributes.csv', 'wordmap.csv'
    """
    def __init__(self, config):
        # Configuration of the numbers of data to be loaded.
        self.conf_n_train = config.NUM_TRAIN
        self.conf_n_test = config.NUM_TEST
        self.conf_n_desc = config.NUM_DESC
        self.conf_n_attr = config.NUM_ATTR
        
        data_path = config.DATA_PATH
        # Path of the sample data
        self.train_path = os.path.join(data_path, 'train.csv')
        # Path of the test data
        self.test_path = os.path.join(data_path, 'test.csv')
        # Path of the product description data
        self.desc_path = os.path.join(data_path, 'product_descriptions.csv')
        # Path of the product attribute data
        self.attr_path = os.path.join(data_path, 'attributes.csv')
        # Dict of user-defined newname for the columns of input dataframes. 
        self.rename_dict = config.RENAME_DICT
        
    def _rename_column(self, df):
        df.rename(columns=self.rename_dict, inplace=True)

    def load_test_data(self, n_test=None):
        if n_test is None:
            df_test = pd.read_csv(self.test_path, encoding="ISO-8859-1")
        else:
            df_test = pd.read_csv(self.test_path, encoding="ISO-8859-1", nrows=n_test)
            assert df_test.shape[0] == n_test, 'Not enough test data.'
        self._rename_column(df_test)
        return df_test
            
    def load_train_data(self, n_train=None):
        if n_train is None:
            df_train = pd.read_csv(self.train_path, encoding="ISO-8859-1")
        else:
            df_train = pd.read_csv(self.train_path, encoding="ISO-8859-1", nrows=n_train)
            assert df_train.shape[0] == n_train, 'Not enough train data.'
        self._rename_column(df_train)
        return df_train
        
    def load_desc_data(self, n_desc=None):
        if n_desc is None:
            df_desc = pd.read_csv(self.desc_path)
        else:
            df_desc = pd.read_csv(self.desc_path, nrows = n_desc)
            assert df_desc.shape[0] == n_desc, 'Not enough product description data.'
        self._rename_column(df_desc)
        return df_desc
        
    def load_attr_data(self, n_attr=None):
        if n_attr is None:
            df_attr = pd.read_csv(self.attr_path)
        else:
            df_attr = pd.read_csv(self.attr_path, nrows = n_attr)
            assert df_attr.shape[0] == n_attr, 'Not enough product attribute data.'
        self._rename_column(df_attr)
        return df_attr

    def get_brand(self, df_attr):
        """
        Extract brand from attribute.
        """
        df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].\
                                            rename(columns={"value": "brand"})
        self._rename_column(df_brand)
        return df_brand

    def get_n_train_and_label(self):
        df_train = self.load_train_data(self.conf_n_train)
        nd_label = df_train['relevance'].values
        return df_train.shape[0], nd_label

    @timethis
    def load_and_merge_all(self):
        """
        Load all data, concat train and test, merge all into df_data.
        """
        df_train = self.load_train_data(self.conf_n_train)
        df_test = self.load_test_data(self.conf_n_test)
        df_desc = self.load_desc_data(self.conf_n_desc )
        df_attr = self.load_attr_data(self.conf_n_attr)

        # Extract brand from attribute
        df_brand = self.get_brand(df_attr)
        # Integrate the data
        df_data = pd.concat((df_train, df_test), axis=0, ignore_index=True)
        df_data = pd.merge(df_data, df_desc, how='left', on='product_uid')
        df_data = pd.merge(df_data, df_brand, how='left', on='product_uid')
        # Get labels
        nd_label = df_train['relevance'].values
        return df_data, df_train.shape[0], nd_label
        
        
