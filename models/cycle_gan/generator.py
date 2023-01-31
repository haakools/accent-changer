from tensorflow_examples.models.pix2pix import pix2pix


OUTPUT_CHANNELS = 1
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

print(generator_f.summary())