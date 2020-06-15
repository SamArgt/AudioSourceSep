import tensorflow as tf
import tensorflow_probability as tfp
from .flow_tfp_bijectors import *
from .flow_tfk_layers import *
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras



def Flowpp_Block(tfp.bijectors.Bijector):

	def __init__