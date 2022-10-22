# Accent Changer Model # 

This repo aims to change different english accents! The project is aimed to better understand various edge models in ML to get a better understanding of model architecure!

Disclaimer: I spent some time trying to setup GPU support for tensorflow, and took the path of least resistance by using anaconda (which I am not the biggest fan of). For further testing, I believe running the model in a docker container would be the better approach.


## Choice of data ##

The choiche of which speakers to use in the dataset was done Ad hoc, based on 

American speaker 7 (male) sounds good -> text to speech ????
American speaker 8 (female) sounds good -> text to speech ????
Bangala speaker 1 (female) sounds good, some noise in the background.
Bangala speaker 2 (female) sounds good, some noise in the background.
Indian speaker 1 & 2 (both female), sounds good but is text to speech
Malayalam speaker 1,2,3 (all female), sounds good, some noise in the background
Odiya speaker 1 (female), sounds good, some noise in the background
Telugu speark 1 &2 (both male), sounds good, some noise in the backgruond


Will use american speaker 7 (male) and Telugu speaker 2 (both male)
Will use american speaker 8 (female) and Bangala 2 (female)













Different languages

* Bangla - It is the official, national, and most widely spoken language of Bangladesh and the second most widely spoken of the 22 scheduled languages of India. [Bengali Language - Wikipedia](https://en.wikipedia.org/wiki/Bengali_language)
* Indian - Hindi, I presume?
* Malayalam - One of the 22 schedules langauges in India, spoken by 34 million people in India [Malayalam Language - Wikiepdia](https://en.wikipedia.org/wiki/Malayalam)
* Telugu - It is one of the few languages that has primary official status in more than one Indian state, alongside Hindi and Bengali. [Telugu Language -  Wikipedia](https://en.wikipedia.org/wiki/Telugu_language)

TODO: make a table of L1 and L2 speakers of the different languages in the dataset, possibly exploring overlap of english speakers in both of them. 








### Preprocessing 
 1. Remove the clutter from the labels (incl the empty lines)
 2. Concatenate all the indian voices X and the American voices Y in the same order. Pair them as X,Y = batch_data for model

###  Dataset
Get the dataset accentdb_extender.tar.gz from https://accentdb.org/#dataset and put it into the dataset folder. 
`harvard_senteces.txt` is the file containing what the transcripts contains.

Need to have TFRecord module which turns npz files into serialized bytelists for faster loading

### Feature Engineering
1. Convert the data into spectrograms.

2. Normalize the data

3. Mel spectrograms

4. Split by words, maybe when the total spectrogram has reached a threshold for some amount of time delta t.

### Data Augmentation
* Noise injection
* Time shifting (shifting startpoitn a random amount)
* Change the pitch
* Changing speed


Note : Besides noise injection, both X and Y needs to be augmentation similarly


### Create model

#### GAN model
Indian accent gets fed through generator which creates an output
The discriminator takes in english accents and output and tries to differentiate between them


#### Diffusion model
Want to understand more about how diffusion models work. 
These are harder to train which may create large difficulties

