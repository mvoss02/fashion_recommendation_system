import tensorflow as tf
from config.config import Variables, Config
from modules.preprocess_data import PreprocessedData
from modules.single_tower_model import SingleTowerModel
from modules.recommendation_model import RecommendationModel


def get_callbacks():
    return [tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq=100), tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',  
                patience=1,          
                min_delta=0.1,
                mode='min',
                )]

def run_training(data: PreprocessedData):
    article_lookups = {key: lkp for key, lkp in data.lookups.items() if key in Variables.ARTICLE_VARIABLES}
    article_model = SingleTowerModel(article_lookups, Config.embedding_dimension)
    customer_lookups = {key: lkp for key, lkp in data.lookups.items() if key in Variables.ALL_CUSTOMER_VARIABLES}
    customer_model = SingleTowerModel(customer_lookups, Config.embedding_dimension)

    model = RecommendationModel(customer_model, article_model, data)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=Config.learning_rate), run_eagerly=True)

    return model.fit(x=data.train_ds,
                     epochs=Config.epochs,
                     batch_size=Config.batch_size,
                     steps_per_epoch=data.nb_train_obs // Config.batch_size,
                     validation_data=data.test_ds,
                     validation_steps=data.nb_test_obs // Config.batch_size,
                     callbacks=get_callbacks(),
                     verbose=1)