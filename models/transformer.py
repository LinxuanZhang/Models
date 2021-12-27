'''
Transformer
https://arxiv.org/pdf/1706.03762v5.pdf
'''
import tensorflow as tf

class MultiHead_Attention(tf.keras.Model):
    def __init__(self, d_model, num_heads = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model//num_heads
        self.q_linear = tf.keras.layers.Dense(d_model)
        self.k_linear = tf.keras.layers.Dense(d_model)
        self.v_linear = tf.keras.layers.Dense(d_model)
        self.Dense = tf.keras.layers.Dense(d_model)
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        q = self.split_head(q, batch_size)
        k = self.split_head(k, batch_size)
        v = self.split_head(v, batch_size)
        attention, attention_weight = self.scaled_dot_product_attention(q, k, v, mask) # weight would have shape = (batch_size, num_head, seq_len_q, seq_len_k)
        attention = tf.transpose(attention, perm = [0, 2, 1, 3]) # shape = (batch_size, seq_len_q, num_heads, d_head)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.d_model)) # shape = (batch_size, seq_len, d_model)
        res = self.Dense(concat_attention)
        return res, attention_weight
    def split_head(self, x, batch_size):
        '''
        reshape x to be in shape(batch_size, num_heads, seq_len, d_head)
        '''
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_head))
        x = tf.transpose(x, perm = [0, 2, 1, 3])
        return x
    def scaled_dot_product_attention(self, q, k, v, mask = None):
        '''
        query, key must have the same 'depth'(same number of columns)
        key, value must have the same 'seq_len'(same number of rows)
        q: query, shape = (.., seq_len_q, dq)
        k: key, shape = (.., seq_len_k, dq) # dq = dk
        v: value, shape = (.., seq_len_k, dv) # seq_len_k = seq_len_v
        mask: shape = (.., seq_len_q, seq_len_k)
        '''
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        matmul_qk = tf.matmul(q, k, transpose_b = True)/tf.math.sqrt(dk)
        if mask is not None:
            matmul_qk += (mask * -1e9) # so that after softmax masked place is 0
        weight = tf.nn.softmax(matmul_qk) # (.., seq_len_q, seq_len_k)
        attention = tf.matmul(weight, v) # (.., seq_len_q, dv)
        return attention, weight


def position_wise_feed_forward_network(d_model = 512, dff = 2048):
    model = tf.keras.Sequential([
        tf.keras.Dense(dff, activation='relu'),
        tf.keras.Dense(d_model)
    ])
    return model


class EcoderLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()
    def call(self, input)


class Transformer(Model):
    def __init__(self):
        super().__init__()
        pass
    def call(self):
        pass
