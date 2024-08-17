try:
    import tensorflow as tf
    import numpy as np
except ImportError:
    print("TensorFlow is not installed. Please install it and try again.")

class WGAN(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(WGAN, self).__init__()
        self.generator = self.build_generator()
        generator_weights = kwargs.pop("generator_weights", None)
        if generator_weights is not None:
            self.generator.load_weights(generator_weights)
        self.critic = self.build_critic()
        critic_weights = kwargs.pop("critic_weights", None)
        if critic_weights is not None:
            self.critic.load_weights(critic_weights)

    def compile(self, d_optimizer, g_optimizer):
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        super(WGAN, self).compile()

    def load_data(self, train_images, batch_size=4):
        self.traindata = tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size)
        traindata_list = []
        for i in range(len(train_images) // batch_size):
            batch = []
            for ib in range(batch_size):
                batch.append(train_images[i*batch_size + ib])
                traindata_list.append(batch)

    def build_generator(input_shape=(256, 256, 3)):
        inputs = tf.keras.layers.Input((256, 256, 3))
        noised_input = tf.keras.layers.GaussianNoise(0.05)(inputs)
        initializer = tf.random_normal_initializer(0, 0.02)

        #Encoder Block 1
        e1c = tf.keras.layers.Conv2D(48, 5, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(noised_input)
        elc = tf.keras.layers.BatchNormalization()(e1c)
        e1c = tf.keras.layers.Conv2D(48, 5, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(e1c)
        elc = tf.keras.layers.BatchNormalization()(e1c)
        e1p = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(e1c)

        #Encoder Block 2
        e2c = tf.keras.layers.Conv2D(96, 5, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(e1p)
        e2c = tf.keras.layers.BatchNormalization()(e2c)
        e2c = tf.keras.layers.Conv2D(96, 5, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(e2c)
        e2c = tf.keras.layers.BatchNormalization()(e2c)
        e2p = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(e2c)

        #Encoder Block 3
        e3c = tf.keras.layers.Conv2D(192, 5, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(e2p)
        e3c = tf.keras.layers.BatchNormalization()(e3c)
        e3c = tf.keras.layers.Conv2D(192, 5, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(e3c)
        e3c = tf.keras.layers.BatchNormalization()(e3c)
        e3p = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(e3c)

        #Encoder Block 4
        e4c = tf.keras.layers.Conv2D(384, 5, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(e3p)
        e4c = tf.keras.layers.BatchNormalization()(e4c)
        e4c = tf.keras.layers.Conv2D(384, 5, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(e4c)
        e4c = tf.keras.layers.BatchNormalization()(e4c)
        e4p = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(e4c)

        #Bottleneck
        b1c = tf.keras.layers.Conv2D(768, 6, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(e4p)
        b1c = tf.keras.layers.BatchNormalization()(b1c)
        b1c = tf.keras.layers.Conv2D(768, 6, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(b1c)
        b1c = tf.keras.layers.BatchNormalization()(b1c)
        b1c = tf.keras.layers.Conv2D(768, 6, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(b1c)
        b1c = tf.keras.layers.BatchNormalization()(b1c)

        #Decoder Block 4
        d4u = tf.keras.layers.UpSampling2D(size=(2, 2))(b1c)
        d4c = tf.keras.layers.Conv2D(384, 5, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(d4u)
        d4c = tf.keras.layers.BatchNormalization()(d4c)
        d4c = tf.keras.layers.concatenate([e4c, d4c], axis=3)
        d4c = tf.keras.layers.Conv2D(384, 3, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(d4c)
        d4c = tf.keras.layers.BatchNormalization()(d4c)
        d4c = tf.keras.layers.Conv2D(384, 3, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(d4c)
        d4c = tf.keras.layers.BatchNormalization()(d4c)

        #Decoder Block 3
        d3u = tf.keras.layers.UpSampling2D(size=(2, 2))(d4c)
        d3c = tf.keras.layers.Conv2D(192, 5, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(d3u)
        d3c = tf.keras.layers.BatchNormalization()(d3c)
        d3c = tf.keras.layers.concatenate([e3c, d3c], axis=3)
        d3c = tf.keras.layers.Conv2D(192, 3, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(d3c)
        d3c = tf.keras.layers.BatchNormalization()(d3c)
        d3c = tf.keras.layers.Conv2D(192, 3, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(d3c)
        d3c = tf.keras.layers.BatchNormalization()(d3c)

        #Decoder Block 2
        d2u = tf.keras.layers.UpSampling2D(size=(2, 2))(d3c)
        d2c = tf.keras.layers.Conv2D(96, 5, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(d2u)
        d2c = tf.keras.layers.BatchNormalization()(d2c)
        d2c = tf.keras.layers.concatenate([e2c, d2c], axis=3)
        d2c = tf.keras.layers.Conv2D(96, 3, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(d2c)
        d2c = tf.keras.layers.BatchNormalization()(d2c)
        d2c = tf.keras.layers.Conv2D(96, 3, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(d2c)
        d2c = tf.keras.layers.BatchNormalization()(d2c)

        #Decoder Block 1
        d1u = tf.keras.layers.UpSampling2D(size=(2, 2))(d2c)
        d1c = tf.keras.layers.Conv2D(48, 5, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(d1u)
        d1c = tf.keras.layers.BatchNormalization()(d1c)
        d1c = tf.keras.layers.concatenate([e1c, d1c], axis=3)
        d1c = tf.keras.layers.Conv2D(48, 3, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(d1c)
        d1c = tf.keras.layers.BatchNormalization()(d1c)
        d1c = tf.keras.layers.Conv2D(48, 3, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(d1c)
        d1c = tf.keras.layers.BatchNormalization()(d1c)

        #Final Block
        f1c = tf.keras.layers.Conv2D(6, 6, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(d1c)
        f1c = tf.keras.layers.BatchNormalization()(f1c)
        f2c = tf.keras.layers.Conv2D(3, 4, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(f1c)
        f2c = tf.keras.layers.BatchNormalization()(f2c)
        f3c = tf.keras.layers.Conv2D(3, 4, activation="swish", padding="same", kernel_initializer=initializer, use_bias=False)(f2c)
        f3c = tf.keras.layers.BatchNormalization()(f3c)
        out = tf.keras.layers.Conv2D(3, 1, activation="tanh")(f3c)

        return tf.keras.Model(inputs=inputs, outputs=out, name="generator")
    def build_critic(self, input_shape=(256, 256, 3)):
        input = tf.keras.layers.Input((256,256,3))
        ds1 = tf.keras.layers.Conv2D(filters=48, strides=2, kernel_size=(5, 5), padding='valid', activation='leaky_relu')(input)
        ds2 = tf.keras.layers.Conv2D(filters=96, strides=2, kernel_size=(5, 5), padding='valid', activation='leaky_relu')(ds1)
        ds3 = tf.keras.layers.Conv2D(filters=192, strides=2, kernel_size=(5, 5), padding='valid', activation='leaky_relu')(ds2)
        ds4 = tf.keras.layers.Conv2D(filters=384, strides=2, kernel_size=(5, 5), padding='valid', activation='leaky_relu')(ds3)
        ds5 = tf.keras.layers.Conv2D(filters=768, strides=2, kernel_size=(5, 5), padding='valid', activation='leaky_relu')(ds4)
        ds5 = tf.keras.layers.Flatten()(ds5)
        out = tf.keras.layers.Dense(1)(ds5)
        return tf.keras.models.Model(inputs=input, outputs=out, name="critic")
    
    def process_image(self, image, **kwargs):
        image_from_array = kwargs.get('image_from_array', False)
        image_from_path = kwargs.get('image_from_path', False)
        if image_from_path:
            image = tf.keras.utils.load_img(image, color_mode='rgb')
            image = tf.keras.utils.img_to_array(image)
            image = np.array([image])
            range = (0, 255)
        else:
            
            if image_from_array:
                image = np.array([image])
                range = kwargs.get('range', (0, 1))
            else:
                image = tf.keras.utils.img_to_array(image)
                image = np.array([image])
                range = (0, 255)
        if range is not (-1, -1):
            image = image / ((range[1] - range[0]) / 2) - 1.0
            proscessed_image = (self.generator(image) + 1.0) * ((range[1] - range[0]) / 2)
        else:
            proscessed_image = self.generator(image)
        return proscessed_image
    def shannon_entropy(self, images):
        entropies = []
        for image in images:
            image = (tf.image.rgb_to_grayscale((image  / 2) + 0.5) * 255)
            image = image.numpy()
            p_hist, bin_edges = np.histogram(image, bins=256, range=(0, 255))
            p_hist = p_hist / np.sum(p_hist)
            p_array = np.digitize(image, bins=bin_edges)
            p = p_hist[p_array.flatten() - 1]
            p = tf.convert_to_tensor(p)
            p = tf.cast(p, dtype=tf.float32)
            entropy = -tf.reduce_sum(p * tf.math.log(p))
            entropies.append(entropy)
        loss = tf.reduce_mean(entropies)
        return loss
    def train_step(self, data):
        images_list = self.database[self.index]
        images = data
        real_images = tf.dtypes.cast(images, tf.float32)
        epsilon = tf.random.uniform(shape=[tf.shape(real_images)[0], 1, 1, 1], minval=0, maxval=1)


        with tf.GradientTape(persistent=True) as c_tape:
            with tf.GradientTape() as gp_tape:
                fake_images = self.generator(images, training=True)
                fake_image_mixed = real_images + epsilon * (fake_images - real_images)
                yhat_mixed = self.critic(fake_image_mixed, training=True)

                grads = gp_tape.gradient(yhat_mixed, fake_image_mixed)
            grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean(tf.square(grads_norm - 1))
            yhat_real = self.critic(real_images, training=True)
            yhat_fake = self.critic(fake_images, training=True)
            total_c_loss = tf.reduce_mean(yhat_real) - tf.reduce_mean(yhat_fake) + gradient_penalty

            cgrad = c_tape.gradient(total_c_loss, self.critic.trainable_variables)
            self.c_opt.apply_gradients(zip(cgrad, self.critic.trainable_variables))

        with tf.GradientTape() as g_tape:
            gen_images = self.generator(images, training=True)
            predicted_labels = self.critic(gen_images, training=True)
            critic_g_loss = tf.reduce_mean(predicted_labels)
            detail_images = []
        for i in range(len(images_list)):
            single_image = self.generator(tf.expand_dims(images_list[i], axis=0), training=True)
            detail_images.append(single_image)
        detail_g_loss = self.shannon_entropy(detail_images) * self.detail_weight
        total_g_loss = detail_g_loss + critic_g_loss
        ggrad = g_tape.gradient((total_g_loss), self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"c_loss":total_c_loss, "g_loss": total_g_loss, "critic_g_loss": critic_g_loss, "detail_g_loss": detail_g_loss}
