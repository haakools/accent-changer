### Preprocessing 
 1. Remove the clutter from the labels (incl the empty lines)
 2. Concatenate all the indian voices X and the American voices Y in the same order. Pair them as X,Y = batch_data for model

###  Dataset
Get the dataset accentdb_extender.tar.gz from accent_db and put it into the dataset folder. 
`harvard_senteces.txt` is the file containing what the transcripts contains.

Need to create tf dataset loaded. Tfrecord??

### Feature Engineering
1. Convert the data into spectrograms.

2. Normalize the data

3. Mel spectrograms(?)

4. Split by words, maybe when the total spectrogram has reached a threshold for some amount of time delta t.

### Data Augmentation
* Noise injection
* Time shifting (shifting startpoitn a random amount)
* Change the pitch
* Changing speed


Note : Besides noise injection, both X and Y needs to be augmentation similarly


### Create model
Create a GAN model
Indian accent gets fed through generator which creates an output
The discriminator takes in english accents and output and tries to differentiate between them

