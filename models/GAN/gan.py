import tensorflow as tf
import numpy as np


"""
Creating a gan for speech generation.
Want a generator model that inputs an indian voice and ouputs an american voice.
Want a discriminator which takes in american voice and the converted american voice.
The generator needs to fool the discriminator. / adversarial training
"""

class GAN(tf.keras.Models.model):
    """Class for the GAN model

    See this tweet for the subclassing method:
    https://twitter.com/fchollet/status/1250622989541838848  

    By doing this, one can call
    >>> gan = GAN(discriminator, generator)
    >>> gan.compile(d_optimizer, g_optimizer, loss_fn)
    >>> gan.fit(dataset, epochs=10)
    with support for callbacks, metrics, etc.
    """
    def __init__(self, discrimnator, generator):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # unpack data
        real_images = data[0]

        # "Forward pass"
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat([tf.ones((batch_size, 1)),
                            tf.zeros((batch_size, 1))], axis=0)
        
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # train the generator (note that we should *not* update the weights of the discriminator)
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}
