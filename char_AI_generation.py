# importing the libraries needed
import random
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop

# loading in our script, using lower case only to improve performance
text = open('script.txt', 'rb').read().decode(encoding='utf-8').lower()

# next we have to convert our words into number, to be read into our network

# we can get all the possible characters in our text
characters = sorted(set(text))

# getting all the characters and index in that order
char_to_index = dict((c, i) for i,c in enumerate(characters))

# getting all the index and characters in that order
index_to_char = dict((i, c) for i,c in enumerate(characters))

# how many of the previous chars to use to predict the next char
seq_len = 40

# how much we shift our chars after every prediction to be used to predict the next char
step_size = 3

sentences = []
next_char = []

# we can comment out our training data processing as we have already used it for training

# loop through the beggining of the text to the end apart from the last sequence and we
# step through our step length
for i in range(0, len(text) - seq_len, step_size):

    # adding in our text from i to i plus sequence length
    # (chunks at a time add)
    sentences.append(text[i: i+seq_len])

    # adding our next char after the sequence to our next char list
    next_char.append(text[i+seq_len])

# 1 dimension for all the possible sentences and 1 for all the characters in our sentence
# then one more for all the possible characters, we do this to get the index of each 
# character in the sentences by indexing them e.g. in sentence 1 (index 0) character 2 
# (index 1) it may contain a 'b' this is represented by 1, and if we do X[0][1][1], it 
# will give us True/1 to show that sentence 1 char 2 contains a 'b'
X = np.zeros((len(sentences), seq_len, len(characters)), dtype = np.bool_)

# here we have a dimension for the number of sentences and 1 for the number of charcters
# we can use the index to get the next character for any given sentence
y = np.zeros((len(sentences), len(characters)), dtype = np.bool_)

# loop for each sentence
for i, sentence in enumerate(sentences):
    # loop for each char in sentence
    for t, char in enumerate(sentence):

        # we load our X values using the index of the sentence and character
        # then the index to character of the character in the sentence
        X[i, t, char_to_index[char]] = 1
    
    # we can now load our y values using our index of sentence
    # and the char to index of the next char
    y[i, char_to_index[next_char[i]]] = 1



# model already trained

# initialising our model
model = Sequential()

# adding our lstm layer to remember the previous characters
model.add(LSTM(units=128, input_shape=(seq_len, len(characters)), return_sequences=True))

# adding our lstm layer to remember the previous characters
model.add(LSTM(units=128))

# we add a dense with an output size of the number of characters
model.add(Dense(units=len(characters)))

# adding our activation layer, softmax
model.add(Activation('softmax'))

# adding our loss function and optimiser
model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(0.01))

# training our model
model.fit(X, y, batch_size = 256, epochs = 200)

# saving our model
model.save('script_gen-5.model')


# loading our pre trained model
model = load_model('script_gen.model')

# creating a function to get a character from our predictions
#  using softmax probabilities and a temperature 
# (higher temp more risky next character, lower the more safe)
def sample(preds, temprature = 1.0):

    # convert our predictions to floats
    preds = np.asarray(preds).astype('float64')

    # using our temprature to be divided by the log of our predictions
    # adjusts how strict the answer has to be
    preds = np.log(preds) / temprature

    # getting the exponential of the predictions
    preds_exp = np.exp(preds)

    # we get the predictions / sum of the predictions (softmax)
    preds = preds_exp / np.sum(preds_exp)

    # we are getting a probability of a multinomial to get the index of the character
    probs = np.random.multinomial(1, preds, 1)

    # we can return the index of the character we have got
    return np.argmax(probs)

def generate_text(length, temperature):

    # we get a random starting point for our generation from our script
    # from the beginning to the end (with space for the sequence length)
    start_index = random.randint(0, len(text) - seq_len-1)

    # we initialise our generated text variable
    generated = ''

    # we create our starting sentence for our generation using our indexes
    sentence = text[start_index: start_index+seq_len]

    # we add our starting sentence to our generated text
    generated += sentence

    # predicting characters for the length inputted
    for i in range(length):

        # x_test array for one sentence of sequence length characters
        # to be used to predict the next character with the index
        x = np.zeros((1, seq_len, len(characters)))

        # we are going to go through each character in our sentence
        for t, char in enumerate(sentence):

            # we can now fill in a one where our character should be in our x_test
            x[0, t, char_to_index[char]] = 1

        # getting our prediction from our model from our x_test input
        # we are turning off verbose (text information printed)
        predictions = model.predict(x, verbose=0)[0]

        # we can use our sample function to get the index of the character we predicted
        next_index = sample(predictions, temperature)

        # we can now add this character to the generated text
        generated += index_to_char[next_index]

        # we now update our sentence to move forward in our generated text
        # by making it the sentence shifted right by 1 and our predicted character
        # added at the end
        sentence = sentence[1:] + index_to_char[next_index]

    return generated

# clearing the terminal (for mac)
import os 
os.system('clear')

# printing our generated text with tempratures
print("-----TEMPERATURE: 0.1-----")
print(generate_text(300, 0.1))
print("-----TEMPERATURE: 0.15-----")
print(generate_text(300, 0.15))
print("-----TEMPERATURE: 0.2-----")
print(generate_text(300, 0.2))
print("-----TEMPERATURE: 0.25-----")
print(generate_text(300, 0.25))
print("-----TEMPERATURE: 0.3-----")
print(generate_text(300, 0.3))
print("-----TEMPERATURE: 0.35-----")
print(generate_text(300, 0.35))
print("-----TEMPERATURE: 0.4-----")
print(generate_text(300, 0.4))
print("-----TEMPERATURE: 0.45-----")
print(generate_text(300, 0.45))
print("-----TEMPERATURE: 0.5-----")
print(generate_text(300, 0.5))
print("-----TEMPERATURE: 0.55-----")
print(generate_text(300, 0.55))
print("-----TEMPERATURE: 0.6-----")
print(generate_text(300, 0.6))
print("-----TEMPERATURE: 0.65-----")
print(generate_text(300, 0.65))
print("-----TEMPERATURE: 0.7-----")
print(generate_text(300, 0.7))
print("-----TEMPERATURE: 0.75-----")
print(generate_text(300, 0.75))
print("-----TEMPERATURE: 0.8-----")
print(generate_text(300, 0.8))
print("-----TEMPERATURE: 0.85-----")
print(generate_text(300, 0.85))
print("-----TEMPERATURE: 0.9-----")
print(generate_text(300, 0.9))
print("-----TEMPERATURE: 0.95-----")
print(generate_text(300, 0.95))
print("-----TEMPERATURE: 1.0-----")
print(generate_text(300, 1.0))


# script_gen.model 1 lstm layer, 1 dense layer 4 epochs
# script_gen-2.model 1 lstm layer 2 dense layers 4 epochs
# script_gen-3.model 1 lstm layer 3 dense layers 100 epochs
# script_gen-4.model 2 lstm layer 1 dense layers 100 epochs
# script_gen-5.model 2 lstm layer 1 dense layers 200 epochs

# considering the model hasnt been trained to understand sentences and word, it does well
# to generate somewhat coherent sentences. We can see the slow progession in the output
# of temperatures as the model starts to try more risky predictions for the next character
# by temperature 1 the model is rambling and makes no sense.
"""
-----model 5 results------

-----TEMPERATURE: 0.1-----
2023-01-11 01:02:59.042168: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
2023-01-11 01:02:59.086869: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
hanging big fish,

you got to be willing the boys.

i was the boys in the back.

what is is is the boy on it.

they were it wish it.

i was the boy now a could a braga me anything anything.

i'm gonna be a brotter.

i was the back of him.

i was the back of here.

i was the boy now and the boys.

i was the back of me are good.

i was the 
-----TEMPERATURE: 0.15-----
 break?

- why should i?

you drive people that good to be of here.

i was the boy now the boys.

i was the broady.

what is a suppet.

i was the boy now in the back.

what is is is the boy now.

i'm gonna be the boys into

it's a copponen of the back.

i was the boy letter the broan.

i was the back it.

- i'm gonna be a bullet of here.

-----TEMPERATURE: 0.2-----
north, 33', 56'...

all clear.

pllot on your him is is the boy now.

i got that is is the boys.

i world you know what is with me of a break.

she was the back of here.

i wish it.

- i'm not the boys.

what is is is the boy on a beer in the paster.

i wish it is the boy now.

they like it is the broad.

i was the back of me a braga him 
-----TEMPERATURE: 0.25-----
obody wanna it with me
oh, i'm so fly

or a seep have is and is is the down.

i was the seep it to be of here.

i wish it with a should and all right.

i will right newd and is off here.

they like it a fice of the back.

i was the boys in the sead.

she was the was the boys

in the say like it.

the way we got the race me of here.

i was
-----TEMPERATURE: 0.3-----
ything looks stable.

uh, guys,
we gotta collar your back of here.

it's a good that down.

i was and the poster.

you know what is are your hands.

(speaklng spanish)

(speaking spanlsh)

(speaking spanlsh)

(speaklng spanish)

(speaklng spanish)

(speaklng spanish)

(speaklng spanlsh)

(speaking spanlsh)

(speaklng spanlsh)

(man seell 
-----TEMPERATURE: 0.35-----
r.

- what makes them think that?

- something like a billers are your him to be in the puttly from your car.

i was that is it out.

i might now are your course.

- and you're going take me shat.

i'm going to get that got to see in the street are here.

broan: this is is what are your him.

i well right now beat them the boy of could a 
-----TEMPERATURE: 0.4-----
gs were different
back then.

once i got a mindon.

i'm gonna be you got to the but ming.

let me in the suttrass into brot.

i was the boys us is is and here in the puttres.

we got a that and that is it the man.

- i think it.

the boy now at your this is
they don't get your this is the boy and and all right now.

you sours off shit is 
-----TEMPERATURE: 0.45-----
tracker.
-pennlng: we don't know that.

and you know what in the way

it's a preal ming.

i'm sere of here in the becker.

i didn't letty we know what is work of your carter, back bight.

she was the car rome men is here

i would he wolld the see.

i'm going to here what is.

she was don't to get to know.

she want the briag into

it's in
-----TEMPERATURE: 0.5-----
yers.

bill myers.

- address?
- el cently.

all right, don't car.

- it's a know what to there.

you know, i the right to be your her.

i wish it?

there your hears we got anything of a car.

connie on of your off-entice.

that's will out.

she was all right.

- i'm sorking in the procees and and craster.

i need be and me and and we got
-----TEMPERATURE: 0.55-----
his up, too.

you won't.

i believe in your the sack unders
how how from to keep that
got to put by for a bull.

hey, man.

i wish man.

what is is here.

any off a ceal.

what is couldn't go that inter.

it's a get here.

she can need that can't call here.

it's were of the but are your hands.

i got it is me allett.

they lown's got a b
-----TEMPERATURE: 0.6-----
utual friend of ours.

- mr. hobbs? - what's us the sure.

but you know what you're want about reess all right?

you're gonna souting
are got a bebing them
got of a boy.

now that's your car.

it's sone real milsion.

let's get in the way.

somery in here.

yeah, we're going to recking.

the bobring for me, man,
we got a stop aman on juil
-----TEMPERATURE: 0.65-----
ng training.

- now, you know you've
never a been the talk peapse of him.

and i a fight?

i'm not don't take for a deep...

- i was are you and read.

(sore heal right surlung)

they think it is.

they're a fill mine.
etcrested

it is i stay sometonet.

intir trould here call me?

(wellsh)

(litter speails plans)

(speaming spanlsh)

(ca
-----TEMPERATURE: 0.7-----
let him get away.

let's end this thing.

what's wilking with a face.

- oh, i don't hele the llause.

cold are right that.

i jist how know.

the don't a little that?

verone the sires if them looking?

coll, yeah.

(speaklng spanish)

(cruiggll cllser]
tolk at hell,

and i'm letty.

i'm here carse.

- what is is?

sound realds to could

-----TEMPERATURE: 0.75-----
he's beautiful, isn't she?

yeah.

she's sere as race what
is wished there's tooge.

this ass good.

i'm torith is with the bight?

him.

thy what of seaw the bedisned,
down.

welr of but in the breaking good
real mali on been polning right.

sory. show, yill the welling on it.

the don't be me like me wantan man?

what is his on screald,
-----TEMPERATURE: 0.8-----
blacktop?

each car does a down-and-back, you hey, man.

- what is is.

we might it?
- fla mam?

(glun thisculupenceaplng)

(halling righing)

sheak olt me like that?

all right it clafer, down.

i'm wart.

i wasck it pulking.

(supten spanish)

(elling grllt himpeam ridlas.

and you'll be theur ent.

i'm reeplngar.

letty was about now,

-----TEMPERATURE: 0.85-----


park: god !

and anyone else who gets man!

let's got a cwayce don't me to retuant
so mact.

bleess buckss please doy.

(speaslng spanish)

lanasels funting in the sideth.

damn! you're going to read.

letty, cracke of ayuro.

i'm best?

therere airms, man.

whon's now give.
- i got sour back!

yo, what is it.

his wells.

- what is lea
-----TEMPERATURE: 0.9-----
and i think i'll prove it.

good luck.

hey, don't these sownificon, that guys.

(sbreaims)

(speaking spanish)

are with the good racear?

i sufking with me.

just we streake right?

hey you just
he got hard firno the lawnooord is the couldn't us oze.

year!

- if a sty, well, cacer ight of place.
and a fasta-

in the would doing to pith
-----TEMPERATURE: 0.95-----
 wanted to tell you
your taillight's out mys.

she was thing, ond veronet-ing tonbors.

velove it working, enter.

myersling of a mille
on themens cars..

- chay,

i was are your fentedsing of some of, sicked.

they're plo.

your sosting making.

they need bett verong us fis then, wayn.

in this sightan
miding.

lettyo to chaw.

shit! man
-----TEMPERATURE: 1.0-----
ough a steel drainpipe?

you see, you have here a

and exicedmolk.

we's chech.

(sllttening)

(crup! i've dreck, imnow thes susing misnicel.

i'll telled.

this is gittie.
what's intormer.

looks [truntlnga torrrpone tick hisclions arey.

polfos!

defim ofe's beal juches,

staliosing we, guys, i makeies.

wask.

the reachtun: is know .
"""