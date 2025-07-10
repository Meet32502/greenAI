import streamlit as st
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from PIL import Image
import io

# Load models
st.set_page_config(page_title="Environmental NLP Dashboard", layout="wide")

st.title("🌍 Environmental Intelligence Dashboard")

# Load NER
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# Load Classification Pipeline
@st.cache_resource
def load_classifier():
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

classifier = load_classifier()

# Load Fill-Mask Pipeline
@st.cache_resource
def load_fill_mask():
    return pipeline("fill-mask", model="bert-base-uncased")

fill_mask = load_fill_mask()

# Load Stable Diffusion
@st.cache_resource
def load_image_gen():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
    pipe.to("cpu")
    return pipe

image_gen = load_image_gen()

# Tabs for each tool
tab1, tab2, tab3, tab4 = st.tabs(["📘 Sentence Classification", "🖼️ Image Generation", "🧠 NER + Graph", "🔤 Fill in the Blank"])

# --- Tab 1: Sentence Classification ---
with tab1:
    st.subheader("Classify Environmental Sentences")
    input_text = st.text_area("Enter a sentence to classify:", "Air pollution levels are increasing in Delhi.")
    if st.button("Classify"):
        result = classifier(input_text)[0]
        st.write(f"**Label:** {result['label']}\n**Confidence:** {round(result['score'] * 100, 2)}%")

# --- Tab 2: Image Generation ---
with tab2:
    st.subheader("Generate Environmental Image")
    prompt = st.text_input("Enter a prompt (e.g. 'forest fire in Himalayas'):", "deforestation in Amazon rainforest")
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            image = image_gen(prompt).images[0]
            st.image(image, caption=prompt)

# --- Tab 3: NER + Graph ---
with tab3:
    st.subheader("Named Entity Recognition & Relationship Graph")
    ner_text = st.text_area("Enter text for NER:", "The Central Pollution Control Board monitors air quality in Mumbai.")
    if st.button("Extract Entities & Visualize Graph"):
        doc = nlp(ner_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        st.write("### Extracted Entities")
        for ent, label in entities:
            st.write(f"- {ent} ({label})")

        G = nx.Graph()
        for ent, label in entities:
            G.add_node(ent, label=label)
        for i in range(len(entities)-1):
            G.add_edge(entities[i][0], entities[i+1][0])

        plt.figure(figsize=(8,5))
        nx.draw(G, with_labels=True, node_color='lightgreen', font_size=10, edge_color='gray')
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        st.image(buf)

# --- Tab 4: Fill in the Blank ---
with tab4:
    st.subheader("Predict Masked Words in Environmental Sentences")
    masked_text = st.text_input("Enter a sentence with [MASK] token:", "The main cause of climate change is [MASK].")
    if st.button("Predict Mask"):
        outputs = fill_mask(masked_text)
        for out in outputs:
            st.write(f"- {out['sequence']} ({round(out['score']*100, 2)}%)")
