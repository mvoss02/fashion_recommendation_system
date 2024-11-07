import tensorflow as tf
from modules.preprocess_data import PreprocessedData
import tensorflow_recommenders as tfrs
from config.config import Variables


class RecommendationModel(tfrs.models.Model):
    def __init__(self,
                 query_model: tf.keras.models.Model,
                 candidate_model: tf.keras.models.Model,
                 data: PreprocessedData):
        super().__init__()
        
        self.query_model = query_model
        self.candidate_model =  candidate_model
        
        # Map each batch to embeddings using the candidate model
        candidate_data = data.article_ds.batch(128)
        candidate_data = candidate_data.map(self.candidate_model).cache().prefetch(tf.data.AUTOTUNE)
            
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            # metrics=tfrs.metrics.FactorizedTopK(
            #     candidates=candidate_data, 
            #     ks=(12,),
            # ),
            # num_hard_negatives=30,
        )


    def compute_loss(self, features, training=False):
        # Extract query features and item features dynamically
        query_inputs = {key: features[key] for key in Variables.ALL_CUSTOMER_VARIABLES}
        item_inputs = {key: features[key] for key in Variables.ARTICLE_VARIABLES}
        
        # Generate embeddings
        query_embeddings = self.query_model(query_inputs)
        item_embeddings = self.candidate_model(item_inputs)

        # Compute and return the retrieval task loss
        loss = self.retrieval_task(query_embeddings, item_embeddings, compute_metrics=not training)
        
        return loss