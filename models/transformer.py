'''
Transformer
https://arxiv.org/pdf/1706.03762v5.pdf
'''
import tensorflow as tf
import numpy as np


def position_wise_feed_forward_network(d_model = 512, dff = 2048):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])
    return model


def get_angles(pos, i, d_model):
  res = pos / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return res


def position_encoding(posistion, d_model):
    angles = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.sin(angles[:, 1::2])
    pos_encode = angles[np.newaxis, ...]
    pos_encode = tf.cast(pos_encode, dtype=tf.float32)
    return pos_encode


def padding_mask(seq):
    '''
    mask all the pad token in the batch of the sequence, ensures the model does not treat padding as input
    '''
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32) # 1 where seq == 0, 0 otherwise
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # upper triangle of shape (seq_len, seq_len)


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


class EncoderLayer(tf.keras.Model):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = MultiHead_Attention(d_model, num_heads)
        self.ffn = position_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    def call(self, input, mask):
        x1, attn_weight = self.mha(input, input, input, mask) # shape = (batch_size, input_seq_len, d_model)
        x1 = self.dropout1(x1)
        x1 = self.layernorm1(x1 + input) # shape = (batch_size, input_seq_len, d_model)
        x2 = self.ffn(x1) # shape = (batch_size, input_seq_len, d_model)
        x2 = self.dropout2(x2)
        x = self.layernorm2(x2 + x1) # shape = (batch_size, input_seq_len, d_model)
        return x, attn_weight


class Encoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = position_encoding(maximum_position_encoding, d_model)
        for i in range(num_layers):
            vars(self)[f'enc_layer{i}'] = EncoderLayer(d_model, num_heads, dff, rate)
        self.dropout = tf.keras.layers.Dropout(rate)
    def call(self, input, mask):
        input_seq_len = tf.shape(input)[1]
        x = self.embedding(input) # shape = (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :input_seq_len, :]
        x = self.dropout(x)
        for i in range(self.num_layers):
            x, _ = vars(self)[f'enc_layer{i}'](x, mask)
        return x


class DecoderLayer(tf.keras.Model):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha1 = MultiHead_Attention(d_model, num_heads)
        self.mha2 = MultiHead_Attention(d_model, num_heads)
        self.ffn = position_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    def call(self, input, encoder_output, look_ahead_mask, padding_mask):
        x1, attn_weight1 = self.mha1(input, input, input, look_ahead_mask) # shape = (batch_size, target_seq_len, d_model)
        x1 = self.dropout1(x1)
        x1 = self.layernorm1(x1 + input)
        x2, attn_weight2 = self.mha2(x1, encoder_output, encoder_output, padding_mask) # shape = (batch_size, target_seq_len, d_model)
        x2 = self.dropout2(x2)
        x2 = self.layernorm2(x2 + x1)
        x3 = self.ffn(x2)
        x3 = self.dropout3(x3)
        x = self.layernorm3(x3 + x2) # shape = (batch_size, target_seq_len, d_model)
        return x, attn_weight1, attn_weight2


class Decoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = position_encoding(maximum_position_encoding, d_model)
        for i in range(num_layers):
            vars(self)[f'dec_layer{i}'] = DecoderLayer(d_model, num_heads, dff, rate)
        self.dropout = tf.keras.layers.Dropout(rate)
    def call(self, input, enc_output, look_ahead_mask, padding_mask):
        input_seq_len = tf.shape(input)[1]
        x = self.embedding(input) # shape = (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :input_seq_len, :]
        x = self.dropout(x)
        attention_weights = {}
        for i in range(self.num_layers):
            x, weight1, weight2 = vars(self)[f'dec_layer{i}'](x, enc_output, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_weight1'] = weight1
            attention_weights[f'decoder_layer{i+1}_weight2'] = weight2
        return x, attention_weights # x.shape = (batch_size, target_seq_len, d_model)


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, output_vocab_size, input_position_encoding, output_position_encoding, rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, input_position_encoding, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, output_vocab_size, output_position_encoding, rate)
        self.fc = tf.keras.layers.Dense(output_vocab_size)
    def call(self, input, target):
        enc_padding_mask, la_mask, dec_padding_mask = self.create_masks(input, target)
        enc_output = self.encoder(input, enc_padding_mask)
        dec_output, attn_weight = self.decoder(target, enc_output, la_mask, dec_padding_mask)
        output = self.fc(dec_output)
        return output, attn_weight
    def create_masks(self, input, target):
        enc_padding_mask = padding_mask(input)
        dec_padding_mask = padding_mask(input)
        la_mask = look_ahead_mask(tf.shape(target)[1])
        target_dec_padding_mask = padding_mask(target)
        la_mask = tf.maximum(la_mask, target_dec_padding_mask)
        return enc_padding_mask, la_mask, dec_padding_mask


sample_transformer = Transformer(
    num_layers=2, d_model=512, num_heads=8, dff=2048,
    input_vocab_size=8500, output_vocab_size=8000,
    input_position_encoding=10000, output_position_encoding=6000)

temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

fn_out, _ = sample_transformer(temp_input, temp_target) # shape = (batch_size, tar_seq_len, target_vocab_size)
