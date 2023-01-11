# RNN-Script-Text-Generation
Using an RNN to be trained to generate movie scripts. This model was trained on fast and furious scripts

i origionally started with trying to predict the next character (char_AI_generation.py) but then i switched to words and the results were alot more
clearer. Even though the next charcter model is more impressive as it correctly outputs whoel words. it wasnt good for a whole script.

word_model_3 in (word_AI_generation.py) has some very good results, that may be due to its overfitting of the data given
i will run the model on the beggining of the fast 7 movie to see if it copies it exactly
output will be in generated.txt. 
We can see it take parts of the movie and stitch them 
with other parts, it does this by stitching the parts where the subject area aligns, 
e.g. in temp 7 it starts with the fast 7 opening but once dom starts talking about his 
childhood it talks about something different in his childhood in another part. 

I decided next to make a function where you can start the movie script. the ouput
is in output.txt.
