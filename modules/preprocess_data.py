import pandas as pd
import tensorflow as tf
from typing import Dict, List
import gc
from config.config import Variables


class PreprocessedData:
    def __init__(self,
                 train_ds: tf.data.Dataset,
                 nb_train_obs: int,
                 test_ds: tf.data.Dataset,
                 nb_test_obs: int,
                 lookups: Dict[str, tf.keras.layers.StringLookup],
                 all_articles: Dict[str, tf.Tensor]):
        self.train_ds = train_ds
        self.nb_train_obs = nb_train_obs
        self.test_ds = test_ds
        self.nb_test_obs = nb_test_obs
        self.lookups = lookups
        self.all_articles = all_articles

    def perform_lookups(self, inputs: Dict[str, tf.Tensor],
                    lookups: Dict[str, tf.keras.layers]) -> Dict[str, tf.Tensor]:
        
        return {key: tf.cast(lkp(inputs[key]), tf.int32) if key in Variables.CUSTOMER_VARIABLES_NUM else lkp(inputs[key]) for key, lkp in lookups.items()}

    def build_lookups(self, train_df) -> Dict[str, tf.keras.layers.StringLookup]:
        lookups = {}
        for var in Variables.ALL_VARIABLES:
            unique_values = train_df[var].unique()
            
            if var in Variables.CUSTOMER_VARIABLES_NUM:
                print('Int:', var)
                lookups[var] = tf.keras.layers.IntegerLookup(vocabulary=unique_values)
            else:
                print('String:', var)
                lookups[var] = tf.keras.layers.StringLookup(vocabulary=unique_values)
                
        return lookups

    def build_article_record(self, article_id: str,
                             articles_metadata: Dict[str, Dict[str, str]]):
        record = {
            'article_id': article_id
        }
        data = articles_metadata[article_id]
        for categ_variable in Variables.ARTICLE_VARIABLES:
            if categ_variable == 'article_id':
                continue
            record[categ_variable] = data[categ_variable]
        return record

    def build_articles_metadata(self, article_df) -> Dict[str, Dict[str, str]]:
        return article_df.set_index('article_id').to_dict('index')

    def build_train_article_df(self, article_df: pd.DataFrame,
                                article_lookup: tf.keras.layers.StringLookup) -> pd.DataFrame:
        all_articles = list(article_lookup.input_vocabulary)
        articles_metadata = self.build_articles_metadata(article_df)  # Use self to call the method
        articles_records = [self.build_article_record(article_id, articles_metadata) for article_id in all_articles]  # Use self to call the method
        return pd.DataFrame.from_records(articles_records)

    
def preprocess(train_df, test_df, article_df, batch_size) -> PreprocessedData:
    nb_train_obs = train_df.shape[0]
    nb_test_obs = test_df.shape[0]

    # Create an instance of PreprocessedData
    preprocessed_data_instance = PreprocessedData(
        train_ds=None,  # Placeholder, will be replaced later
        nb_train_obs=nb_train_obs,
        test_ds=None,   # Placeholder, will be replaced later
        nb_test_obs=nb_test_obs,
        lookups=None,   # Placeholder, will be replaced later
        all_articles=None  # Placeholder, will be replaced later
    )
    
    lookups = preprocessed_data_instance.build_lookups(train_df)
    
    print('Done preparing the lookups...')

    train_ds = tf.data.Dataset.from_tensor_slices(dict(train_df[Variables.ALL_VARIABLES])) \
                            .shuffle(1_000) \
                            .batch(batch_size) \
                            .map(lambda inputs: preprocessed_data_instance.perform_lookups(inputs, lookups)) \
                            .repeat() \
                            .prefetch(tf.data.experimental.AUTOTUNE)
                            
    # Call garbage collector
    gc.collect()
                            
    print('Done generating the training set...')
    
    test_ds = tf.data.Dataset.from_tensor_slices(dict(test_df[Variables.ALL_VARIABLES])) \
                     .batch(batch_size) \
                     .map(lambda inputs: preprocessed_data_instance.perform_lookups(inputs, lookups)) \
                     .repeat()
                            
    print('Done generating the test set...')

    article_lookup = lookups['article_id']
    train_article_df = preprocessed_data_instance.build_train_article_df(article_df, article_lookup)
    article_lookups = {key: lkp for key, lkp in lookups.items() if key in Variables.ARTICLE_VARIABLES}
    article_ds = tf.data.Dataset.from_tensor_slices(dict(train_article_df)) \
                                .batch(len(train_article_df)) \
                                .map(lambda inputs: preprocessed_data_instance.perform_lookups(inputs, article_lookups))
    all_articles = next(iter(article_ds))
    
    print('Done generating the article data...')

    # Now update the instance with actual datasets and lookups
    preprocessed_data_instance.train_ds = train_ds
    preprocessed_data_instance.test_ds = test_ds
    preprocessed_data_instance.lookups = lookups
    preprocessed_data_instance.all_articles = all_articles

    return preprocessed_data_instance