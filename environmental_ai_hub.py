import streamlit as st
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, pipeline,
    BertTokenizer, BertForMaskedLM
)
from diffusers import StableDiffusionPipeline
import pandas as pd
import json
from PIL import Image
import io
import base64
import re
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌿 Green AI Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Green AI theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .task-container {
        background: linear-gradient(135deg, #d0f0c0 0%, #b2d8b2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .result-container {
        background: linear-gradient(135deg, #e6ffe6 0%, #ccffcc 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #388e3c;
    }

    .stButton > button {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem 0;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f0f9f0 0%, #e0f2e0 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌿 Green AI Assistant</h1>
    <p>Machine Learning Introduction to Environmental Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Session State Initialization
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'results_history' not in st.session_state:
    st.session_state.results_history = []

@st.cache_resource
def load_models():
    try:
        models = {}

        models['classifier'] = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        models['ner'] = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
        models['mask_tokenizer'] = BertTokenizer.from_pretrained('bert-base-uncased')
        models['mask_model'] = BertForMaskedLM.from_pretrained('bert-base-uncased')
        models['text2img'] = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )

        if torch.cuda.is_available():
            models['text2img'] = models['text2img'].to("cuda")

        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def classify_environmental_sentence(text, classifier):
    environmental_keywords = {
        'renewable_energy': ['solar', 'wind', 'renewable', 'clean energy', 'hydroelectric', 'geothermal'],
        'pollution_control': ['pollution', 'emissions', 'waste', 'contamination', 'toxic', 'air quality'],
        'climate_change': ['climate', 'global warming', 'greenhouse', 'carbon', 'temperature'],
        'conservation': ['conservation', 'biodiversity', 'ecosystem', 'wildlife', 'forest', 'ocean'],
        'sustainability': ['sustainable', 'green', 'eco-friendly', 'recycling', 'circular economy']
    }
    text_lower = text.lower()
    for category, keywords in environmental_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return category.replace('_', ' ').title()
    result = classifier(text)
    sentiment_to_env = {
        'POSITIVE': 'Environmental Benefit',
        'NEGATIVE': 'Environmental Concern',
        'NEUTRAL': 'Environmental Topic'
    }
    return sentiment_to_env.get(result[0]['label'], 'Environmental Topic')

def extract_entities_and_map(text, ner_model):
    entities = ner_model(text)
    env_mapping = {
        'PERSON': 'Person', 'ORG': 'Organization', 'GPE': 'Country/Location',
        'LOC': 'Location', 'DATE': 'Date/Time', 'MONEY': 'Financial',
        'PERCENT': 'Percentage', 'CARDINAL': 'Number'
    }
    env_terms = {
        'emissions': 'Climate Policy', 'carbon': 'Climate Policy', 'net-zero': 'Climate Policy',
        'renewable': 'Energy Policy', 'solar': 'Energy Technology', 'wind': 'Energy Technology',
        'pollution': 'Environmental Issue', 'climate': 'Climate Topic'
    }
    mapped_entities = []
    for entity in entities:
        entity_text = entity['word'].replace('##', '')
        entity_label = entity['entity_group']
        mapped_label = env_mapping.get(entity_label, entity_label)
        for term, env_type in env_terms.items():
            if term in entity_text.lower():
                mapped_label = env_type
                break
        if re.match(r'\d{4}', entity_text):
            mapped_label = 'Year'
        mapped_entities.append({entity_text: mapped_label})
    return mapped_entities

def fill_mask_bert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    if len(mask_token_index) == 0:
        return "No [MASK] token found"
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    mask_token_logits = predictions[0, mask_token_index, :]
    top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    predicted_tokens = [tokenizer.decode([token]) for token in top_tokens]
    best_token = predicted_tokens[0].strip()
    filled_text = text.replace("[MASK]", best_token)
    return {
        "predicted_word": best_token,
        "completed_sentence": filled_text,
        "alternatives": predicted_tokens[1:4]
    }

def generate_environmental_image(prompt, text2img_model):
    enhanced_prompt = f"beautiful, high-quality, detailed environmental scene: {prompt}, green technology, sustainable future, clean environment, photorealistic, 4k"
    with torch.no_grad():
        image = text2img_model(
            enhanced_prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            width=512,
            height=512
        ).images[0]
    return image

def main():
    with st.sidebar:
        st.markdown("### 🎯 Available Tasks")
        st.markdown("""
        - 🔤 **Sentence Classification**
        - 🖼️ **Image Generation**
        - 🏷️ **Entity Recognition & Mapping**
        - 🔍 **Masked Word Prediction**
        """)
        st.markdown("### 📊 Model Status")
        if st.button("🔄 Load Models"):
            with st.spinner("Loading AI models..."):
                models = load_models()
                if models:
                    st.session_state.models_loaded = True
                    st.success("✅ Models loaded successfully!")
                else:
                    st.error("❌ Failed to load models")
        if st.session_state.models_loaded:
            st.success("🤖 Models Ready")
        else:
            st.warning("⚠️ Models not loaded")

    if not st.session_state.models_loaded:
        st.info("👆 Please load the models from the sidebar to get started!")
        return

    models = load_models()
    if not models:
        st.error("Failed to load models. Please check your internet connection and try again.")
        return

    st.markdown("### 🎯 Select Your Task")
    task = st.selectbox("Choose an AI task:", [
        "🔤 Sentence Classification",
        "🖼️ Image Generation",
        "🏷️ Entity Recognition & Mapping",
        "🔍 Masked Word Prediction"
    ])

    if task == "🔤 Sentence Classification":
        st.markdown('<div class="task-container">', unsafe_allow_html=True)
        sentence_input = st.text_area("Enter your sentence:", value="Solar energy reduces air pollution.", height=100)
        if st.button("🔍 Classify Sentence"):
            if sentence_input.strip():
                with st.spinner("Classifying..."):
                    result = classify_environmental_sentence(sentence_input, models['classifier'])
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown(f"**🎯 Classification Result:** {result}")
                    st.markdown(f"**📝 Input:** {sentence_input}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a sentence to classify.")
        st.markdown('</div>', unsafe_allow_html=True)

    elif task == "🖼️ Image Generation":
        st.markdown('<div class="task-container">', unsafe_allow_html=True)
        image_prompt = st.text_area("Enter your image description:", value="A city filled with electric cars and solar panels", height=100)
        if st.button("🎨 Generate Image"):
            if image_prompt.strip():
                with st.spinner("Generating image..."):
                    image = generate_environmental_image(image_prompt, models['text2img'])
                    if image:
                        st.markdown('<div class="result-container">', unsafe_allow_html=True)
                        st.markdown(f"**🎨 Generated Image for:** {image_prompt}")
                        st.image(image, caption="Generated Environmental Scene", use_column_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter an image description.")
        st.markdown('</div>', unsafe_allow_html=True)

    elif task == "🏷️ Entity Recognition & Mapping":
        st.markdown('<div class="task-container">', unsafe_allow_html=True)
        ner_input = st.text_area("Enter your text:", value="India aims for net-zero emissions by 2070.", height=100)
        if st.button("🔍 Extract & Map Entities"):
            if ner_input.strip():
                with st.spinner("Extracting entities..."):
                    entities = extract_entities_and_map(ner_input, models['ner'])
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown(f"**🎯 Extracted Entities for:** {ner_input}")
                    st.json(entities)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter text for entity extraction.")
        st.markdown('</div>', unsafe_allow_html=True)

    elif task == "🔍 Masked Word Prediction":
        st.markdown('<div class="task-container">', unsafe_allow_html=True)
        mask_input = st.text_area("Enter your sentence with [MASK]:", value="Trees absorb [MASK] from the air.", height=100)
        if st.button("🔍 Predict Masked Word"):
            if mask_input.strip() and "[MASK]" in mask_input:
                with st.spinner("Predicting masked word..."):
                    result = fill_mask_bert(mask_input, models['mask_tokenizer'], models['mask_model'])
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown(f"**🎯 Predicted Word:** `{result['predicted_word']}`")
                    st.markdown(f"**📝 Complete Sentence:** {result['completed_sentence']}")
                    st.markdown(f"**🔄 Alternatives:** {', '.join(result['alternatives'])}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a sentence with [MASK] token.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
