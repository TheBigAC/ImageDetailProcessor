if __name__ != "__main__":
    import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
    def call(self, inputs):
        self.inputImg = tf.keras.layers.Input((256, 256, 3))(inputs)
        self.noised_input = tf.keras.layers.GaussianNoise(0.05)(self.inputImg)

        #Encoder Block 1
        self.e1c = tf.keras.layers.Conv2D(48, 5, activation="swish", padding="same")(self.noised_input)
        self.e1c = tf.keras.layers.Conv2D(48, 5, activation="swish", padding="same")(self.e1c)
        self.e1p = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(self.e1c)

        #Encoder Block 2
        self.e2c = tf.keras.layers.Conv2D(96, 5, activation="swish", padding="same")(self.e1p)
        self.e2c = tf.keras.layers.Conv2D(96, 5, activation="swish", padding="same")(self.e2c)
        self.e2p = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(self.e2c)

        #Encoder Block 3
        self.e3c = tf.keras.layers.Conv2D(192, 5, activation="swish", padding="same")(self.e2p)
        self.e3c = tf.keras.layers.Conv2D(192, 5, activation="swish", padding="same")(self.e3c)
        self.e3p = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(self.e3c)

        #Encoder Block 4
        self.e4c = tf.keras.layers.Conv2D(384, 5, activation="swish", padding="same")(self.e3p)
        self.e4c = tf.keras.layers.Conv2D(384, 5, activation="swish", padding="same")(self.e4c)
        self.e4p = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(self.e4c)

        #Bottleneck
        self.bc = tf.keras.layers.Conv2D(768, 6, activation="swish", padding="same")(self.e4p)
        self.bc = tf.keras.layers.Conv2D(768, 6, activation="swish", padding="same")(self.bc)
        self.bc = tf.keras.layers.Conv2D(768, 6, activation="swish", padding="same")(self.bc)

        #Decoder Block 4
        self.d4u = tf.keras.layers.UpSampling2D(size=(2, 2))(self.bc)
        self.d4c = tf.keras.layers.Conv2D(384, 5, activation="swish", padding="same")(self.d4u)
        self.d4c = tf.keras.layers.concatenate([self.e4c, self.d4c], axis=3)
        self.d4c = tf.keras.layers.Conv2D(384, 3, activation="swish", padding="same")(self.d4c)
        self.d4c = tf.keras.layers.Conv2D(384, 3, activation="swish", padding="same")(self.d4c)

        #Decoder Block 3
        self.d3u = tf.keras.layers.UpSampling2D(size=(2, 2))(self.d4c)
        self.d3c = tf.keras.layers.Conv2D(192, 5, activation="swish", padding="same")(self.d3u)
        self.d3c = tf.keras.layers.concatenate([self.e3c, self.d3c], axis=3)
        self.d3c = tf.keras.layers.Conv2D(192, 3, activation="swish", padding="same")(self.d3c)
        self.d3c = tf.keras.layers.Conv2D(192, 3, activation="swish", padding="same")(self.d3c)

        #Decoder Block 2
        self.d2u = tf.keras.layers.UpSampling2D(size=(2, 2))(self.d3c)
        self.d2c = tf.keras.layers.Conv2D(96, 5, activation="swish", padding="same")(self.d2u)
        self.d2c = tf.keras.layers.concatenate([self.e2c, self.d2c], axis=3)
        self.d2c = tf.keras.layers.Conv2D(96, 3, activation="swish", padding="same")(self.d2c)
        self.d2c = tf.keras.layers.Conv2D(96, 3, activation="swish", padding="same")(self.d2c)

        #Decoder Block 1
        self.d1u = tf.keras.layers.UpSampling2D(size=(2, 2))(self.d2c)
        self.d1c = tf.keras.layers.Conv2D(48, 5, activation="swish", padding="same")(self.d1u)
        self.d1c = tf.keras.layers.concatenate([self.e1c, self.d1c], axis=3)
        self.d1c = tf.keras.layers.Conv2D(48, 3, activation="swish", padding="same")(self.d1c)
        self.d1c = tf.keras.layers.Conv2D(48, 3, activation="swish", padding="same")(self.d1c)

        #Final Block
        self.f1c = tf.keras.layers.Conv2D(6, 6, activation="swish", padding="same")(self.d1c)
        self.f2c = tf.keras.layers.Conv2D(3, 4, activation="swish", padding="same")(self.f1c)
        self.f3c = tf.keras.layers.Conv2D(3, 4, activation="swish", padding="same")(self.f2c)
        self.out = tf.keras.layers.Conv2D(3, 1, activation="sigmoid")(self.f3c)
        return self.out;