import tensorflow as tf
import numpy as np


"""
Creating a gan for speech generation.
Want a generator model that inputs an indian voice and ouputs an american voice.
Want a discriminator which takes in american voice and the converted american voice.
The generator needs to fool the discriminator. / adversarial training



Model selection:
    The data will be convoluted images in and out.
    Look for imagewise models, convolutions
    Start with a very basic model and see how it improves. 
    The data coverage should be really good, even without augmentation.
    
"""


