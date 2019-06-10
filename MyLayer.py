
# coding: utf-8

# In[1]:


from keras import backend as K
from keras.engine.topology import Layer
#自作のレイヤー
class MyDense(Layer):
  def __init__(self, output_dim, activation, **kwargs):
    self.output_dim = output_dim
    self.activation = activation
    super(MyDense, self).__init__(**kwargs)

  def build(self, input_shape):
    self.kernel = self.add_weight(name='kernel',
                                  shape=(input_shape[1], self.output_dim),
                                  initializer='glorot_uniform')
    self.bias   = self.add_weight(name='bias',
                                  shape=(1, self.output_dim),
                                  initializer='zeros')
    super(MyDense, self).build(input_shape)

  def call(self, x):
    if self.activation == 'relu':
      return(K.relu(K.dot(x, self.kernel) + self.bias))
    elif self.activation == 'phase': #自作のアクティベーション
      return(K.sigmoid(K.dot(K.sin(x), self.kernel) + self.bias))
    elif self.activation == 'phase2': #自作のアクティベーション
      return(K.sigmoid(K.dot(K.cos(x), self.kernel) + self.bias))
    elif self.activation == 'softmax':
      return(K.softmax(K.dot(x, self.kernel) + self.bias))
    elif self.activation == 'sigmoid':
      return(K.sigmoid(K.dot(x, self.kernel) + self.bias))
    elif self.activation == 'dot':
      return(K.dot(x, self.kernel) + self.bias)
    else:
      return(K.dot(x, self.kernel) + self.bias)
  def compute_output_shape(self, input_shape):
    return(input_shape[0], self.output_dim)

