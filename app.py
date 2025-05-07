### 1. Imports and class names setup ### 
import gradio as gr
#import os
#from transformers import BertForSequenceClassification
from transformers import pipeline
#import torch
#import numpy as np
#from transformers import BertTokenizer, BertModel, BertConfig
# from model import create_effnetb2_model
from timeit import default_timer as timer


def predict1(text):

    # Start the timer
    start_time = timer()

    # Pipeline d’analyse de sentiments
    classifier = pipeline("sentiment-analysis")


    result = classifier(text)
    #for text, result in zip(texts, results):
        #print(f"Texte : {text} Résultat : {result}\n")

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    return result[0]["label"], 100*result[0]["score"], pred_time

def predict(text):
    # Start the timer
    start_time = timer()

    # Pipeline d’analyse de sentiments
    classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

    label_map = {
        'LABEL_0': 'Negative',
        'LABEL_1': 'Neutral',
        'LABEL_2': 'Positive'
    }

    result = classifier(text)

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    return label_map[result[0]["label"]], 100*result[0]["score"], pred_time
### 4. Gradio app ###

# Create title, description and article strings
title = "US AIRLINE SENTIMENT ANALYSIS"
description = "Using Bert to predict the us airline tweels sntiment. Either positive, neutral or negative."

# Create examples list from "examples/" directory
# example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Textbox(lines=10, max_lines=10, label="Input Text"),
                    outputs=[gr.Text(label="Result"), 
                             gr.Number(label="Prediction score (%)"),
                             gr.Number(label="Prediction time (s)")],
                    # Create examples list from "examples/" directory
                    #examples=examples, 
                    title=title,
                    description=description)

# Launch the demo!
demo.launch()

