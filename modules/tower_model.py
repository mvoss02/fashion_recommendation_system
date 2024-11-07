import math
from typing import Dict
import tensorflow as tf


class TowerModel(tf.keras.models.Model):
    def __init__(self,
                 lookups: Dict[str, tf.keras.layers.StringLookup],
                 embedding_dimension: int):
        
        super().__init__()
        
        self._all_embeddings = {}
        
        for variable in lookups.keys():
            lookup = lookups[variable]
            vocab_size = lookup.vocabulary_size()
            
            # Assign embedding dimensions
            if variable == 'article_id' or variable == 'customer_id':
                var_emb_dim = 128
            else:
                var_emb_dim = max(8, int(3 * math.log(vocab_size, 2)))
            
            embedding_layer = tf.keras.layers.Embedding(vocab_size, var_emb_dim)
            self._all_embeddings[variable] = embedding_layer

        # Dense layers for further processing
        self._dense1 = tf.keras.layers.Dense(256, activation='relu')
        self._dense2 = tf.keras.layers.Dense(embedding_dimension, activation='relu')

    def call(self, inputs):
        all_embeddings = []
        
        # Loop over all variables and gather embeddings
        for variable, embedding_layer in self._all_embeddings.items():
            embeddings = embedding_layer(inputs[variable])
            
            # Flatten the embeddings using Flatten layer
            embeddings = tf.keras.layers.Flatten()(embeddings)
            
            all_embeddings.append(embeddings)
        
        # Concatenate all embeddings
        all_embeddings = tf.concat(all_embeddings, axis=1)
        
        # Pass through the dense layers
        outputs = self._dense1(all_embeddings)
        
        return self._dense2(outputs)