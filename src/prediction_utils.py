import tensorflow as tf

class EsmCEsmPDModel:
    
    def __init__(self,
                 tf_model_loc):
        self.tf_model_loc = tf_model_loc
        model = tf.saved_model.load(self.tf_model_loc )
        self.model_serving_func = model.signatures["serving_default"]

    def predict(self, X, C):
        prediction = tf.squeeze(self.model_serving_func(Adj=C, Input=X)['graph_attention_1']).numpy()
        return prediction

class AF2AF2PDModel:
    
    def __init__(self,
                 tf_model_loc):
        self.tf_model_loc = tf_model_loc
        model = tf.saved_model.load(self.tf_model_loc )
        self.model_serving_func = model.signatures["serving_default"]

    def predict(self, X, C):
        prediction = tf.squeeze(self.model_serving_func(Adj=C, Input=X)['graph_attention_1']).numpy()
        return prediction