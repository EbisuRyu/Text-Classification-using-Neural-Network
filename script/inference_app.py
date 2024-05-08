# Handle importing from src and models
import sys
import os
import torch
import streamlit as st
import plotly.graph_objects as go

from pathlib import Path
from glob import glob
from processing import *
from model import *
from config import *
from model import TextClassificationModel
from torchtext.data.utils import get_tokenizer

file_dir_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory path of the current file
models_dir_path = os.path.join(os.path.dirname(file_dir_path), 'model')  # Set the directory path for models
vocabs_dir_path = os.path.join(os.path.dirname(file_dir_path), 'vocab')  # Set the directory path for tokenizers
script_dir_path = os.path.join(os.path.dirname(file_dir_path), 'script')  # Set the directory path for sources

def get_last_directory(path):
    parent_dir = os.path.dirname(path).split('\\')[-1]
    return os.path.join(parent_dir, os.path.basename(path))

def list_model_vocab_paths(model_directory, vocab_directory):
    model_paths = []
    vocab_paths = []
    for dirpath, dirnames, filenames in os.walk(model_directory):
        for filename in filenames:
            if filename.endswith('.pt'):
                model_paths.append(os.path.join(dirpath, filename))
                
    for dirpath, dirnames, filenames in os.walk(vocab_directory):
        for filename in filenames:
            if filename.endswith('.pt'):
                vocab_paths.append(os.path.join(dirpath, filename))
    return model_paths, vocab_paths

def load_vocab_model(model_pth, vocab_pth):
    # Load the model from checkpoint with specified configurations
    model = TextClassificationModel()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load model weights based on the selected device
    if device == 'cuda':
        state = torch.load(model_pth, map_location = torch.device('cuda'))
        model.load_state_dict(state['model_state_dict'])
    else:
        state = torch.load(model_pth, map_location = torch.device('cpu'))
        model.load_state_dict(state['model_state_dict'])
    model.eval()
    vocab = torch.load(vocab_pth)
    return model, vocab

# Initialize dictionaries to hold model and tokenizer paths
model_dict = {}
vocab_dict = {}

# Get lists of model and tokenizer paths
model_paths, vocab_paths = list_model_vocab_paths(models_dir_path, vocabs_dir_path)
# For each model path, add an entry to the model dictionary
# The key is the last directory in the path, and the value is the path itself
for model_path in model_paths:
    model_dict[get_last_directory(model_path)] = model_path
for vocab_path in vocab_paths:
    vocab_dict[get_last_directory(vocab_path)] = vocab_path

# Get lists of the keys (i.e., the last directories in the paths) in the model and tokenizer dictionaries
model_lst = list(model_dict.keys())
vocab_lst = list(vocab_dict.keys())
#---------------------------------Streamlit App---------------------------------#
# Configure streamlit layout
st.title("Sentiment Analysis Inference App")  # Set the title of the Streamlit app

# Set up the the sidebar for model and tokenizer selection
st.sidebar.subheader("Model Selection")  # Add a subheader to the sidebar
selected_model = st.sidebar.selectbox("Model", model_lst)  # Create a dropdown for selecting the model
selected_vocab = st.sidebar.selectbox("Vocabulary", vocab_lst)  # Create a dropdown for selecting the model
text_input = st.text_input("Input text", "")  # Add a text input field for user input
run_button = st.button("Run model")  # Add a button for running the model

if run_button:
    # Load model and tokenizer
    model_pth = model_dict[selected_model]  # Get the path of the selected model
    vocab_pth = vocab_dict[selected_vocab]  # Get the path of the selected tokenizer
    tokenizer = get_tokenizer('basic_english')
    vocab = torch.load(vocab_pth)
    model = TextClassificationModel(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_CLASS)
    model.load_state_dict(torch.load(model_pth))
    
    text_input = clean_text(text_input)
    encoded_text = vocab(tokenizer(text_input))
    encoded_sentence = torch.tensor(encoded_text, dtype=torch.int64)
    offsets = torch.tensor([0])
    probabilities, labels = model.predict(encoded_sentence, offsets)
    if labels[0] == 0:
        class_name = 'Negative'
    else:
        class_name = 'Positive'

    # Data for the donut chart
    labels = ['Negative', 'Positive']
    probabilities = probabilities[0] # Example probabilities, make sure they sum up to 1

    # Calculate colors based on probabilities
    colors = ['rgb(255, 0, 71)', 'rgb(0, 204, 102)']

    # Create the donut chart figure
    fig = go.Figure(data=[go.Pie(labels=labels, values=probabilities, hole=.3, marker=dict(colors=colors))])

    # Update layout for better aesthetics
    fig.update_layout(
        title="Donut Chart",
        title_font_size=20,
        title_font_family="Arial",
        title_font_color="black",
        title_x=0.5,
        title_y=0.95,
        font=dict(size=14),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # Update the font size of the numbers representing probabilities
    fig.update_traces(textfont=dict(color='white', size=20, family='Arial', weight='bold'))

    # Display the chart in Streamlit
    st.plotly_chart(fig)