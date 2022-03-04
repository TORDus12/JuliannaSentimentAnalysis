import speech_recognition as sr
import numpy as np
import pyaudio
import time
from Sentiment_Analysis import Sentiment, tokenizer, model

dict = {
        0 : "very negative",
        1 : "negative",
        2 : "neutral",
        3 : "positive",
        4 : "very positve"
    }


def callback(r, audio):
    try:
        speech = r.recognize_google(audio)
        return speech
    except:
        return "Illegible"

def main():
    #To Clear the data from the previous session
    file = open("Statement.txt", "w")
    file.close()


    r = sr.Recognizer()
    mic = sr.Microphone(1)


    print("Play Audio")



    while True:
        with mic as source:
            r.adjust_for_ambient_noise
            r.pause_threshold = 0.5
            audio = r.listen(source)
        phrase = callback(r, audio)

        sent_val = Sentiment(tokenizer, model, phrase)

        print("You said: " + phrase + " with a sentiment value of " + "("+ str(sent_val) + ") " + dict[sent_val])


if __name__ == '__main__':
    main()
