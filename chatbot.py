import random
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
import gradio as gr
from model_training.model import NeuralNet
from model_training.nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("data_preparation/intents.json", "r") as f:
    intents = json.load(f)

systems = ['australian', 'car', 'diabetes', 'fertilityDiagnosis', 'german-credit', 'heartdisease', 'hepatitis-filled',
           'house-votes-84', 'mushroom-modified-filled', 'nursery', 'soybean-large-all-filled', 'SPECT-all-modif',
           'SPECTF-all-modif']


FILE = 'model_training/data.pth'
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


last = ""


def decision_system_import(name, index):
    system = np.loadtxt('data_preparation/dane/'+str(name)+".txt", dtype=str)
    system_types = np.loadtxt('data_preparation/dane/'+str(name)+"-type.txt", dtype=str)
    global last
    last = name
    obiekty = system.shape[0]
    counts = Counter(system[:, -1])
    classes = "  ".join([f"{key}: (Obiekty: {value})" for key, value in counts.items()])

    attributes = system_types[:, 0]
    attributes_types = system_types[:, 1]
    diff_value_count = {attr: len(set(system[:, i])) for i, attr in enumerate(attributes)}
    diff_value = {attr: set(system[:, i]) for i, attr in enumerate(attributes)}
    common_value = {attr: Counter(system[:, i]).most_common()[0][0] for i, attr in enumerate(attributes)}
    min_value = {attr: "Atrybut symboliczny" if attr_type == "s" else sorted(Counter(system[:, i]).values())[0] for i, (attr, attr_type) in enumerate(zip(attributes, attributes_types))}
    max_value = {attr: "Atrybut symboliczny" if attr_type == "s" else sorted(Counter(system[:, i]).values())[-1] for i, (attr, attr_type) in enumerate(zip(attributes, attributes_types))}

    diff = "  ".join([f"{attr}: {', '.join(value)}" for attr, value in diff_value.items()])
    diff_count = "  ".join([f"{attr}: {value}" for attr, value in diff_value_count.items()])
    common = "  ".join([f"{attr}: {value}" for attr, value in common_value.items()])
    min_max = "  ".join([f"{attr}: (Min) {min_value[attr]} (Max) {max_value[attr]}" for attr in min_value])

    txt = f"System {name}:\n\nObiekty - {obiekty}\n\nKlasy - {classes}\n\nUnikatowe wartości - {diff}\n\nIlość unikatowych wartości - {diff_count}\n\nNajczęstsze wartości - {common}\n\nMinimum i maksimum - {min_max}"
    file = pd.DataFrame(system)
    return file.head(index).values.tolist(), txt


def predict(inp, history=None):
    history = history or []
    inp2 = inp + ' ' + last
    sentence = tokenize(inp2)
    predict = bag_of_words(sentence, all_words)
    predict = torch.from_numpy(predict).to(device)
    output = model(predict.unsqueeze(0))
    predicted = torch.argmax(output)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        responses = [tg['responses'] for tg in intents["intents"] if tg["tag"] == tag]
        response = random.choice(responses[0])
    else:
        response = "Nie wiem o co Ci chodzi. Zadaj inne pytanie."
    history.append((inp, response))
    return history, history


with gr.Blocks() as demo:
    with gr.Tab("System decyzyjny"):
        with gr.Row():
            name = gr.Dropdown(choices=systems, label="Nazwa systemu", interactive=True)
            index = gr.Slider(minimum=1, maximum=9000, label="Suwak")
            greet_btn = gr.Button("Generuj")
        with gr.Row():
            output_txt = gr.Textbox(label="Szczegółowy opis systemu decyzyjnego")
        with gr.Row():
            output = gr.Dataframe(headers=["Column 1", "Column 2", "Column 3"], label="Tabela systemu decyzyjnego")
        greet_btn.click(fn=decision_system_import, inputs=[name, index], outputs=[output, output_txt])

    chatbot = gr.Chatbot()
    state = gr.State([])
    with gr.Tab("Chat"):
        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="Wprowadź tekst i naciśnij Enter")
            txt.submit(predict, [txt, state], [chatbot, state])

demo.launch()
