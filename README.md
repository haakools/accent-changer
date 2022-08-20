### Preprocessing 
# 1. Remove the clutter from the labels
# (incl the empty lines)
# 2. Concatenate all the indian voices X and the American voices Y in
# the same order. Pair them as X,Y = batch_data for model



### Feature Engineering
# Convert the data into spectrograms

# Normalize the data

# Check through the repo for other ideas to preprocessing

### Data Augmentation
# Noise injection
# Time shifting (shifting startpoitn a random amount)
# Change the pitch
# Chaning speed
# Besides noise injection, both X and Y needs to be augmentation similarly


### Create model
# Create a GAN model
# Indian accent gets fed through generator which creates an output
# The discriminator takes in english accents and output and tries to differentiate between them

