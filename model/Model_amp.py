import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, SimpleRNN, Bidirectional, Flatten, BatchNormalization, AvgPool1D
from tensorflow.keras.layers import Conv2D, MaxPool2D,Reshape
import numpy as np
def Model_amp():
    # left model:       Conv + channel attention + Dense
    # input features:   DCGR
    input_left = tf.keras.Input(shape=(158, 8))
    l = tf.expand_dims(input_left, -1)
    l = Conv2D(16, kernel_size=(1, 8), strides=1, padding='valid')(l)
    ca = channel_attention(l, in_planes=16, ratio=8)  ##通道注意力机制,in_planes is the same as the feature maps of Conv2D
    l = tf.multiply(l, ca)
    l = BatchNormalization()(l)
    l = Flatten()(l)
    l = Dense(350, activation='relu')(l)
    output_left = Dropout(rate=0.434055)(l)
    # output_left=tf.keras.layers.Reshape(-1,1)(output_left)

    # middle model:     BiLSTM
    # input features:   PSSM
    input_middle = tf.keras.Input(shape=(50, 20))  ##pssm
    m = Bidirectional(LSTM(115, return_sequences=False))(input_middle)
    output_middle = BatchNormalization()(m)
    # output_middle =tf.keras.layers.Reshape(-1,1)(output_middle)

    # right model:     Transformer Encoder
    # input features:   position and properties
    input_right = tf.keras.Input(shape=(50, 8))  ##chemiphysical
    r = Encoder2(dff=5, num_heads=2, num_layers=6)(input_right)
    r = tf.transpose(r, (0, 2, 1))
    r = tf.nn.avg_pool(r, ksize=[8], strides=[1], padding='VALID')
    output_right = BatchNormalization()(r)
    output_right = Flatten()(output_right)


    concatenated = keras.layers.concatenate([output_left, output_middle, output_right])
    x = Dense(200, activation='relu')(concatenated)
    x = Dropout(rate=0.5)(x)
    final_output = Dense(1, activation='sigmoid')(x)
    final_model = keras.models.Model(inputs=[input_left, input_middle, input_right], outputs=final_output)
    return final_model





def channel_attention(inputs, in_planes, ratio):
    avgpool = tf.keras.layers.GlobalAveragePooling2D(name='channel_avgpool')(inputs)
    maxpool = tf.keras.layers.GlobalMaxPooling2D(name='channel_maxpool')(inputs)

    # Shared MLP
    Dense_layer1 = tf.keras.layers.Dense(in_planes // ratio, activation='relu', name='channel_fc1')
    Dense_layer2 = tf.keras.layers.Dense(in_planes, activation='relu', name='channel_fc2')
    avg_out = Dense_layer2(Dense_layer1(avgpool))
    max_out = Dense_layer2(Dense_layer1(maxpool))

    channel = tf.keras.layers.add([avg_out, max_out])
    channel = tf.keras.layers.Activation('sigmoid', name='channel_sigmoid')(channel)
    channel_att = tf.keras.layers.Reshape((1, 1, in_planes), name='channel_reshape')(channel)
    return channel_att

'''
encoder part of transformer model
'''
def positional_encoding(pos, d_model):
    def get_angles(position, i):
        # return shape=[position_num, d_model]
        return position / np.power(10000., 2. * (i // 2.) / np.float(d_model))

    angle_rates = get_angles(np.arange(pos)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :])
    pe_sin = np.sin(angle_rates[:, 0::2])
    pe_cos = np.cos(angle_rates[:, 1::2])
    pos_encoding = np.concatenate([pe_sin, pe_cos], axis=-1)
    pos_encoding = tf.cast(pos_encoding[np.newaxis, ...], tf.float32)
    return pos_encoding

'''*************** First part: Scaled dot-product attention ***************'''
def scaled_dot_product_attention(q, k, v, mask):
    '''attention(Q, K, V) = softmax(Q * K^T / sqrt(dk)) * V'''
    # query 和 Key相乘
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(q)[-1], tf.float32)
    scaled_attention =matmul_qk / tf.math.sqrt(dk)
    # 掩码mask
    if mask is not None:
        scaled_attention += mask * -1e-9
    attention_weights = tf.nn.softmax(scaled_attention)  # shape=[batch_size, seq_len_q, seq_len_k]
    outputs = tf.matmul(attention_weights, v)  # shape=[batch_size, seq_len_q, depth]
    return outputs, attention_weights

'''*************** Second part: Multi-Head Attention ***************'''

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # shape=[batch_size, seq_len_q, d_model]
        k = self.wq(k)
        v = self.wq(v)
        q = self.split_heads(q, batch_size)  # shape=[batch_size, num_heads, seq_len_q, depth]
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # scaled_attention shape=[batch_size, num_heads, seq_len_q, depth]
        # attention_weights shape=[batch_size, num_heads, seq_len_q, seq_len_k]
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # shape=[batch_size, seq_len_q, num_heads, depth]

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # shape=[batch_size, seq_len_q, d_model]

        output = self.dense(concat_attention)
        return output, attention_weights

class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(),
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x): # x shape=[batch_size, seq_len, d_model]
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

def point_wise_feed_forward(d_model, diff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff, activation=tf.nn.relu),
        tf.keras.layers.Dense(d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward(d_model, dff)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    def call(self, inputs, training, mask):
        # multi head attention (encoder时Q = K = V)
        att_output, _ = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout1(att_output, training=training)
        output1 = self.layernorm1(inputs + att_output)  # shape=[batch_size, seq_len, d_model]
        # feed forward network
        ffn_output = self.ffn(output1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layernorm2(output1 + ffn_output)  # shape=[batch_size, seq_len, d_model]
        return output2

class Encoder2(tf.keras.layers.Layer):
    def __init__(self,dff=10,num_heads=1, num_layers=6, d_model=8,  max_seq_len=50,  dropout_rate=0.1,**kwargs):
        super(Encoder2, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.pos_encoding = positional_encoding(max_seq_len, d_model)  # shape=[1, max_seq_len, d_model]
        self.encoder_layer = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
                              for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    def call(self, word_embedding, training=False):
        # input part；word_embedding=[batch_size, 50, 40]
        emb= word_embedding + self.pos_encoding[:, :, :]
        x = self.dropout(emb, training=training)
        for i in range(self.num_layers):
            x = self.encoder_layer[i](x, training, None)
        return x  # shape=[batch_size, seq_len, d_model]
    def get_config(self):
        config=super().get_config().copy()
        config.update({

            'num_layers':self.num_layers,
            'd_model':self.d_model,
            # 'max_seq_len':self.max_seq_len,
            # 'dropout_rate':self.dropout_rate

        })
        return config
