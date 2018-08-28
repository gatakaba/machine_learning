# import numpy
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
from keras import layers
from keras.layers import Input, Dense, Lambda, Layer, Flatten, Reshape
from keras.models import Model
from keras import backend, metrics
from keras.datasets import mnist
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import backend as K
import numpy as np


# 2次元正規分布から1点サンプリングする補助関数
def sampling(args):
    z_mean, z_log_var = args
    epsilon = backend.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + backend.exp(z_log_var * 0.5) * epsilon


class CustomVariationalLayer(Layer):
    """Keras の Layer クラスを継承してオリジナルの損失関数を付加するレイヤー"""

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        """オリジナルの損失関数"""
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        # 入力と出力の交差エントロピー
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        # 事前分布と事後分布のKL情報量
        kl_loss = - 0.5 * backend.sum(1 + z_log_var - backend.square(z_mean) - backend.exp(z_log_var), axis=-1)
        return backend.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)  # オリジナルの損失関数を付加
        return x  # この自作レイヤーの出力を一応定義しておきますが、今回この出力は全く使いません


batch_size = 100
latent_dim = 2
epsilon_std = 1.0

# 変分自己符号化器を構築する

# エンコーダ
x = Input(batch_shape=(batch_size, 28, 28))
h = Reshape((28, 28, 1))(x)

h = layers.Conv2D(32, 3, padding="same", activation='relu', strides=(2, 2))(h)
h = layers.Conv2D(32, 3, padding="same", activation='relu')(h)

h = layers.MaxPool2D(pool_size=2, padding="same")(h)
shape_before_flattening = K.int_shape(h)

h = Flatten()(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# デコーダ
h = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(z)
h = Reshape(shape_before_flattening[1:])(h)

h = layers.UpSampling2D(2)(h)

h = layers.Conv2DTranspose(32, 3, padding="same", activation='relu')(h)
h = layers.Conv2DTranspose(32, 3, padding="same", activation='relu', strides=(2, 2))(h)

# decoder_h = Dense(intermediate_dim, activation='relu')
# decoder_mean = Dense(original_dim, activation='sigmoid')
# h_decoded = decoder_h(z)
# x_decoded_mean = decoder_mean(h_decoded)

# h_decoded = Dense(intermediate_dim, activation='relu')(z)
x_decoded_mean = Dense(1, activation='sigmoid')(h)

# x_decoded_mean = layers.Conv2D(1,3,padding ="same", activation='sigmoid')(h)
x_decoded_mean = Reshape([28, 28])(x_decoded_mean)
# カスタマイズした損失関数を付加する訓練用レイヤー
y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)
vae.compile(optimizer='adam', loss=None)
# ============================================================
# モデルを訓練します
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), numpy.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), numpy.prod(x_test.shape[1:])))
epochs = 1000
vae.fit(x_train, shuffle=True, epochs=epochs, batch_size=batch_size)
# ============================================================
# 結果を表示します

# (1) 隠れ変数空間のプロット（エンコードした状態のプロット）
encoder = Model(x, z_mean)  # エンコーダのみ分離
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(15, 12))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test, cmap="jet")
plt.colorbar()
plt.show()
