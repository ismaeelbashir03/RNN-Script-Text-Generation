# importing the libraries needed
import random
import string
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop

# loading in our script, using lower case only to improve performance
text = open('script.txt', 'rb').read().decode(encoding='utf-8').lower().replace(' ', ' SPACEBAR ').replace('\n', ' NEWLINE ')

# removing punctuation
exclude = set(string.punctuation)
text = ''.join(ch for ch in text if ch not in exclude)

# we split our text to a list of words
text = text.split(' ')

# removing empty strings
text = [i for i in text if i != '']

# removing newlines one after the other, this helped tremendously for trianing our loss after
# reached 0.19 at epoch 30 when before with 200 it only could go down to 0.23 at the end.
# this must have been because the model had to just predict newlines most of the time since
# they occured too much
x = text[-1] # because we remove the last item in the lext line
text = [x for i,x in enumerate(text[:-1]) if not(text[i] == 'NEWLINE' and text[i+1] == 'NEWLINE')] +[x]

# we can get all the possible words in our text
words = sorted(set(text))

# getting all the words and index in that order
word_to_index = dict((w, i) for i,w in enumerate(words))

# getting all the index and words in that order
index_to_word = dict((i, w) for i,w in enumerate(words))

# how many of the previous words to use to predict the next char
seq_len = 7

# how much we shift our words after every prediction to be used to predict the next char
step_size = 1

sentences = []
next_word = []

# we can comment out our training data processing as we have already used it for training
'''
# loop through the beggining of the text to the end apart from the last sequence and we
# step through our step length
for i in range(0, len(text) - seq_len, step_size):

    # adding in our text from i to i plus sequence length
    # (chunks at a time add)
    sentences.append(text[i: i+seq_len])

    # adding our next word after the sequence to our next word list
    next_word.append(text[i+seq_len])

# 1 dimension for all the possible sentences and 1 for all the words in our sentence
# then one more for all the possible words, we do this to get the index of each 
# word in the sentences by indexing them e.g. in sentence 1 (index 0) word 2 
# (index 1) it may contain a 'the' this is represented by an index, and if we do 
# X[0][1][index], it will give us True/1 to show that sentence 1 char 2 contains 'the'
X = np.zeros((len(sentences), seq_len, len(words)), dtype = np.bool_)

# here we have a dimension for the number of sentences and 1 for the number of words
# we can use the index to get the next word for any given sentence
y = np.zeros((len(sentences), len(words)), dtype = np.bool_)

# loop for each sentence
for i, sentence in enumerate(sentences):
    # loop for each word in sentence
    for t, word in enumerate(sentence):

        # we load our X values using the index of the sentence and word
        # then the index to word of the word in the sentence
        X[i, t, word_to_index[word]] = 1
    
    # we can now load our y values using our index of sentence
    # and the word to index of the next word
    y[i, word_to_index[next_word[i]]] = 1

'''

# model already trained
'''
# initialising our model
model = Sequential()

# adding our lstm layer to remember the previous words
model.add(LSTM(units=256, input_shape=(seq_len, len(words)), return_sequences=True))

# adding our lstm layer to remember context
model.add(LSTM(units=128))

# we add a dense with an output size of the number of words
model.add(Dense(units=len(words)))

# adding our activation layer, softmax
model.add(Activation('softmax'))

# adding our loss function and optimiser
model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(0.01))

# training our model
model.fit(X, y, batch_size = 256, epochs = 200)

# saving our model
model.save('word_script_gen-3.model')
'''
 
# loading our pre trained model
model = load_model('word_script_gen-3.model')

# creating a function to get a word index from our predictions
#  using softmax probabilities and a temperature 
# (higher temp more risky next word, lower the more safe)
def sample(preds, temprature = 1.0):

    # convert our predictions to floats
    preds = np.asarray(preds).astype('float64')

    # using our temprature to be divided by the log of our predictions
    # adjusts how strict the answer has to be
    preds = np.log(preds+0.000000001) / temprature

    # getting the exponential of the predictions
    preds_exp = np.exp(preds)

    # we get the predictions / sum of the predictions (softmax)
    preds = preds_exp / np.sum(preds_exp)

    # we are getting a probability of a multinomial to get the index of the character
    probs = np.random.multinomial(1, preds, 1)

    # we can return the index of the word we have got
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
    generated += ' '.join(sentence)

    # predicting characters for the length inputted
    for i in range(length):

        # x_test array for one sentence of sequence length characters
        # to be used to predict the next character with the index
        x = np.zeros((1, seq_len, len(words)))

        # we are going to go through each character in our sentence
        for t, char in enumerate(sentence):

            # we can now fill in a one where our character should be in our x_test
            x[0, t, word_to_index[char]] = 1

        # getting our prediction from our model from our x_test input
        # we are turning off verbose (text information printed)
        predictions = model.predict(x, verbose=0)[0]

        # we can use our sample function to get the index of the word we predicted
        next_index = sample(predictions, temperature)

        # we can now add this word to the generated text
        generated += ' '+index_to_word[next_index]

        # we now update our sentence to move forward in our generated text
        # by making it the sentence shifted right by 1 and our predicted word
        # added at the end
        sentence = sentence[1:] + [index_to_word[next_index]]
    
    # reoving our spacebar and newline keywords
    generated = generated.replace(' SPACEBAR', '').replace('NEWLINE', '\n').replace('SPACEBAR', '')

    return generated

def generate_fast_7(length, temperatures):

    # list for all the generated temperatures
    final = []

    # for each temprature inputted
    for temperature in temperatures:
        # we initialise our generated text variable
        generated = ''

        # we create our starting sentence to be from the index the fast 7 movie starts
        sentence = text[text.index('england')-2:text.index('england')+5]

        # we add our starting sentence to our generated text
        generated += ' '.join(sentence)

        # predicting characters for the length inputted
        for i in range(length):

            # x_test array for one sentence of sequence length characters
            # to be used to predict the next character with the index
            x = np.zeros((1, seq_len, len(words)))

            # we are going to go through each character in our sentence
            for t, char in enumerate(sentence):

                # we can now fill in a one where our character should be in our x_test
                x[0, t, word_to_index[char]] = 1

            # getting our prediction from our model from our x_test input
            # we are turning off verbose (text information printed)
            predictions = model.predict(x, verbose=0)[0]

            # we can use our sample function to get the index of the word we predicted
            next_index = sample(predictions, temperature)

            # we can now add this word to the generated text
            generated += ' '+index_to_word[next_index]

            # we now update our sentence to move forward in our generated text
            # by making it the sentence shifted right by 1 and our predicted word
            # added at the end
            sentence = sentence[1:] + [index_to_word[next_index]]
        
        # reoving our spacebar and newline keywords
        generated = generated.replace(' SPACEBAR', '').replace('NEWLINE', '\n').replace('SPACEBAR', '')

        # printing our prediction
        print(f'-----TEMPERATURE {temperature}-----')
        print(generated)

        final.append(generated)

    return final

def generate_custom_input(input, temperatures = [0.4], length = 300, title = None):

    # first we format our input to be used to predict the next text
    input = input.lower().replace(' ', ' SPACEBAR ').replace('\n', ' NEWLINE ')

    # removing punctuation
    exclude = set(string.punctuation)
    input = ''.join(ch for ch in input if ch not in exclude)

    # we split our text to a list of words
    input = input.split(' ')

    # removing empty strings
    input = [i for i in input if i != '']

    # removing newlines one after the other, this helped tremendously for trianing our loss after
    # reached 0.19 at epoch 30 when before with 200 it only could go down to 0.23 at the end.
    # this must have been because the model had to just predict newlines most of the time since
    # they occured too much
    x = input[-1] # because we remove the last item in the lext line
    input = [x for i,x in enumerate(input[:-1]) if not(input[i] == 'NEWLINE' and input[i+1] == 'NEWLINE')] +[x]


    # list for all the generated temperatures
    final = []

    # for each temprature inputted
    for temperature in temperatures:
        # we initialise our generated text variable
        generated = ''

        # we create our starting sentence to be from the index the fast 7 movie starts
        sentence = input[len(input)-seq_len:]

        # we add our starting sentence to our generated text
        generated += ' '.join(input)

        # predicting characters for the length inputted
        for i in range(length):

            # x_test array for one sentence of sequence length characters
            # to be used to predict the next character with the index
            x = np.zeros((1, seq_len, len(words)))

            # we are going to go through each character in our sentence
            for t, char in enumerate(sentence):

                # we can now fill in a one where our character should be in our x_test
                x[0, t, word_to_index[char]] = 1

            # getting our prediction from our model from our x_test input
            # we are turning off verbose (text information printed)
            predictions = model.predict(x, verbose=0)[0]

            # we can use our sample function to get the index of the word we predicted
            next_index = sample(predictions, temperature)

            # we can now add this word to the generated text
            generated += ' '+index_to_word[next_index]

            # we now update our sentence to move forward in our generated text
            # by making it the sentence shifted right by 1 and our predicted word
            # added at the end
            sentence = sentence[1:] + [index_to_word[next_index]]
        
        # reoving our spacebar and newline keywords (double spacing for newline)
        generated = generated.replace(' SPACEBAR', '').replace('NEWLINE', '\n').replace('SPACEBAR', '')

        # printing our prediction
        print(f'-----TEMPERATURE {temperature}-----')
        # printing the title
        if title != None:
            print(f'FAST AND FURIOUS: {title}')
        else:
            print('FAST AND FURIOUS')
        print("WRITTEN BY ME\n\n")
        print(generated)

        final.append(generated)

    return final

# clearing the terminal (for mac)
import os 
os.system('clear')

# printing our generated text with tempratures (already done, see below)
'''
print("-----TEMPERATURE: 0.1-----")
print(generate_text(100, 0.1))
print("-----TEMPERATURE: 0.15-----")
print(generate_text(100, 0.15))
print("-----TEMPERATURE: 0.2-----")
print(generate_text(100, 0.2))
print("-----TEMPERATURE: 0.25-----")
print(generate_text(100, 0.25))
print("-----TEMPERATURE: 0.3-----")
print(generate_text(100, 0.3))
print("-----TEMPERATURE: 0.35-----")
print(generate_text(100, 0.35))
print("-----TEMPERATURE: 0.4-----")
print(generate_text(100, 0.4))
print("-----TEMPERATURE: 0.45-----")
print(generate_text(100, 0.45))
print("-----TEMPERATURE: 0.5-----")
print(generate_text(100, 0.5))
print("-----TEMPERATURE: 0.55-----")
print(generate_text(100, 0.55))
print("-----TEMPERATURE: 0.6-----")
print(generate_text(100, 0.6))
print("-----TEMPERATURE: 0.65-----")
print(generate_text(100, 0.65))
print("-----TEMPERATURE: 0.7-----")
print(generate_text(100, 0.7))
print("-----TEMPERATURE: 0.75-----")
print(generate_text(100, 0.75))
print("-----TEMPERATURE: 0.8-----")
print(generate_text(100, 0.8))
print("-----TEMPERATURE: 0.85-----")
print(generate_text(100, 0.85))
print("-----TEMPERATURE: 0.9-----")
print(generate_text(100, 0.9))
print("-----TEMPERATURE: 0.95-----")
print(generate_text(100, 0.95))
print("-----TEMPERATURE: 1.0-----")
print(generate_text(100, 1.0))
'''

# generating faast 7 movie script with different temps
'''
result = generate_fast_7(500,temperatures=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
with open('generated.txt', 'w') as f:
    f.writelines(result)
'''

# custom input script
result = generate_custom_input(
    "(Dom jumps)\n I hate driving and being so cool.",
    [0.4,0.5,0.6,0.7],
    300
)

# we just save the first temperature to the txt file (you can change this)
with open('output.txt', 'w') as f:
    f.writelines('FAST AND FURIOUS\n'+"WRITTEN BY ME\n\n"+result[0])

# script_gen-1.model 1 lstm layer, 1 dense layer 4 epochs
# script_gen-2.model 2 lstm layer 1 dense layers 4 epochs
# script_gen-3.model 2 lstm layer 1 dense layers 200 epochs

# model 3 has some very good results, that may be due to its overfitting of the data given
# i will run the model on the beggining of the fast 7 movie to see if it copies it exactly
# output will be in generated.txt. We can see it take parts of the movie and stitch them 
# with other parts, it does this by stitching the parts where the subject area aligns, 
# e.g. in temp 7 it starts with the fast 7 opening but once dom starts talking about his 
# childhood it talks about something different in his childhood in another part. I decided next 
# to make a function where you can start the movie script with 7 words, INCLUDING spaces. the ouput
# is in output.txt
"""
-----model 2 funny results------

-----TEMPERATURE: 0.5-----
you racing 
  
  i racing
 that how you are that of the man they might even never gonna ask you   
  
 hey again 
  
 its in a cop from the girls bro 
  
 whats in the car. the plan 
  
  im sorry we dont got that 

-----model 3 results------

-----TEMPERATURE: 0.1-----
  
 i wouldnt miss it for the world 
  
  you dont even trust me do you 
  
  you know what they say 
 where were from 
  
 show me how you drive 
 ill show you who you are 
  
  cheers 
  
 nice 
  
 im a policeman 
  
 can you give me
-----TEMPERATURE: 0.2-----
doing 
  im good how you doing 
  
 l didnt know there were any rules 
  
 speaklng spanlsh 
  
 you all right 
  yeah 
  
  well i dont 
  
  thats what i want to introduce you to somebody 
  
 hes a uhh 
  
 hes an amazing guy 
  
 yeah he is 
 
-----TEMPERATURE: 0.3-----

 just what l always wanted 
  
 brlan l got a question for you 
  
  you know what they say 
 where were from 
  
 show me how you drive 
 ill show you who you are 
  
  cheers 
  
 nice 
  
 im a policeman 
  
 im trying to help you 
  
 because were 

-----TEMPERATURE: 0.4-----
miami 
  
 but theyve had a hard time 
 getting the cash out 
  
 weve been surveilling him for a year 
  
 but weve never been able to put him 
 and the money together 
  
 youre gonna be late 
  
 sean 
  
 what the hell is he doing 
  the feds are in the wrong

-----TEMPERATURE: 0.5-----
 solitary 
  
 i only need two 
  
 i want you 
 to rip that ebrake 
  
 all right 
  
 my life be like 
 ooh aah ooh ooh 
  
 my life be like 
  
 its times like these 
 that make me say 
  
 lord if you see me then you mean 
 it then you know

-----TEMPERATURE: 0.6-----
 my money i told you man 
  
 im glad you like it 
  
 darling will you hold that 
  
 we should find two so we just got here 
  and now were leaving 
  
 lets go 
  
 come on baby you got this rome 
 this dude aint serious 
  
 yeah you did 
  

-----TEMPERATURE: 0.7-----
that component personally 
  
 toretto ill arrange transportation for you 
 and your cars 
  
 you brush up on your spanish boys 
 ill see you that low 
  
 you know what they say 
 where were from 
  
 show me how you drive 
 ill show you who you are 
  
  cheers 
  
 nice 
  
 im

-----TEMPERATURE: 0.8-----
 you son of a bitch 
  
 just do it 
  
 your funeral 
  
 ride or die remember 
  
 listen up 
  
 out here in the mountains 
 by the king 
  
 police 
  
 listen very carefully 
  
 the driver of car 55 in the 
 international is frank webster 
  
  come on bill 
  

-----TEMPERATURE: 0.9-----
 hit it baby hit it 
  
 all right were going to get in there 
  go 
  
 go now 
  
 come on man can i lost your mouth 
  
 now we got a match 
  
 the fingerprints bragas 1 00 
  
 just waiting on facial confirmation via fax 
  
 brlan l come on

-----TEMPERATURE: 1.0-----
 the next four days 
  
 they got a tank 
  
 im sorry did somebody just say a tank 
  
 we got him man 
  
 we got a problem with authority 
  
 i have that same problem 
  
 for me its cops in particular 
  
 chuckling 
  
 lets take a walk 
  
 if it


 --- after newline fix ----
 example only, full output in generated.txt
 
-----TEMPERATURE 0.4-----
london england 
 they say if you want to do 
 were gonna take them off your hands 
 you know why dont we settle this now 
 wait wait wait how about we settle 
 this on the blacktop 
 each car does a downandback 
 tagteam style for slips 
 loser walks home 
 we came to race 
 load them up then 
 come on bro lets get these cars 
 all right check it out theres no way 
 were gonna beat these guys straight up 
 that hemis putting out about 425 
 and that yenko will snap 
 a speedo in about five seconds flat 
 were gonna have to pull something 
 out of our ass 
 the only thing i can think of is save 
 the spray for the way back 
 the return trip 
 done deal 
 all right lets do this bro 
 im getting that orange one 
 rome laughs 
 you aint ready i run these streets 
 whooping 
 romey rome you aint heard of me 
 oh shit 
 cheering 
 im gonna get you im gonna get you 
 wheres your big mouth now 
 that cars going home with me homey 
 nitrous oxide hisses 
 exclaiming 
 got to go dawg 
 yelling in spanish 
 put down the gun now 
 put down the gun now 
 put down the gun 
 enough enough 
 shut up shut up 
 its over come on 
 its over lets go 
 its our gun come on lets go 
 mallk hey dont touch me 
 hey dont touch that 
 thats nothing to play with it 
 dwlght hey 
 what are you doing man 
 gas hlsslng 
 come on 
 speaklng spanlsh 
 get out
"""