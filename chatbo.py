# -*- coding: utf-8 -*-
"""
Created on Wed May 13 03:09:09 2020

@author: abhis
"""


from newspaper import Article
import random
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
import warnings
from nltk import punkt

nltk.download('punkt')
warnings.filterwarnings('ignore')

article = Article('https://en.wikipedia.org/wiki/Coronavirus')
article.download() #Download the article
article.parse() #Parse the article
article.nlp() #Apply Natural Language Processing (NLP)
corpus = article.text #Store the article text into corpus
print(corpus)
text = corpus

sent_tokens = nltk.sent_tokenize(text)# txt to a list of sentences
print(sent_tokens)

remove_punct_dict = dict(  (ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))

GREETING_INPUTS = ["hi", "hello",  "hola", "greetings",  "wassup","hey"] 
#Greeting responses back to the user
GREETING_RESPONSES = ["hi ","hey ","hey","Hello ","Hello Human"]
#Function to return a random greeting response to a users greeting
def greeting(sentence):
   #If user's input is a greeting, return a randomly chosen greeting response
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
def response(user_response):
    robo_response='' #Create an empty response for the bot
    sent_tokens.append(user_response) #Append the users response to the list of sentence tokens
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english') 
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    score = flat[-2]
    if(score==0):
        robo_response=robo_response+"I apologize, I don't understand."
    else:
        robo_response = robo_response+sent_tokens[idx]
    sent_tokens.remove(user_response) 
       
    return robo_response

flag=True
print("ChatBot: I am Electronic BOT for short. I will answer your queries about Electronics. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Bot: You're welcome!, Chat with you later !")
        else:
            if(greeting(user_response)!=None):
                print("Bot: "+greeting(user_response))
            else:
                print("Bot: "+response(user_response))
    else:
        flag=False
        print("DOCBot: Chat with you later !")