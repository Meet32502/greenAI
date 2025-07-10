import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
import plotly.figure_factory as ff
from datetime import datetime
import base64
import io
from PIL import Image
import random
import re

# Set page config
st.set_page_config(
    page_title="Environmental AI Hub",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    
    .stTextArea > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    
    .category-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        color: white;
    }
    
    .env-pollution { background-color: #e74c3c; }
    .climate-change { background-color: #3498db; }
    .renewable-energy { background-color: #2ecc71; }
    .conservation { background-color: #f39c12; }
    .sustainability { background-color: #9b59b6; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌍 Environmental AI Analysis Hub</h1>
    <p>Advanced AI-powered environmental analysis and insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🔧 Navigation")
page = st.sidebar.selectbox(
    "Choose Analysis Tool",
    ["🏠 Home", "📊 Sentence Classification", "🎨 Image Generation", "🔍 NER & Graph Analysis", "📝 Fill in the Blanks"]
)

# Sample data for demonstrations
SAMPLE_SENTENCES = [
    "Carbon emissions from factories are causing severe air pollution in urban areas.",
    "Solar panels are becoming more efficient and affordable for residential use.",
    "Deforestation in the Amazon rainforest threatens biodiversity and climate stability.",
    "Ocean plastic pollution is harming marine life and ecosystems worldwide.",
    "Wind energy installations are growing rapidly across coastal regions.",
    "Recycling programs help reduce waste and conserve natural resources.",
    "Global warming is melting polar ice caps at an alarming rate.",
    "Sustainable agriculture practices can help preserve soil health and water quality.",
    "Electric vehicles are reducing transportation-related greenhouse gas emissions.",
    "Wetland restoration projects are improving water quality and wildlife habitats."
]

ENVIRONMENTAL_CATEGORIES = {
    "Environmental Pollution": {
        "color": "#e74c3c",
        "keywords": ["pollution", "contamination", "toxic", "waste", "emissions", "smog", "acid rain"],
        "description": "Issues related to air, water, and soil contamination"
    },
    "Climate Change": {
        "color": "#3498db", 
        "keywords": ["climate", "global warming", "greenhouse", "temperature", "melting", "sea level"],
        "description": "Climate patterns and global warming effects"
    },
    "Renewable Energy": {
        "color": "#2ecc71",
        "keywords": ["solar", "wind", "renewable", "clean energy", "hydroelectric", "geothermal"],
        "description": "Sustainable energy sources and technologies"
    },
    "Conservation": {
        "color": "#f39c12",
        "keywords": ["conservation", "wildlife", "biodiversity", "habitat", "ecosystem", "endangered"],
        "description": "Protection of natural resources and wildlife"
    },
    "Sustainability": {
        "color": "#9b59b6",
        "keywords": ["sustainable", "recycling", "green", "eco-friendly", "renewable", "circular economy"],
        "description": "Long-term environmental responsibility practices"
    }
}

FILL_BLANK_TEMPLATES = [
    "The increase in _____ emissions is a major contributor to global warming.",
    "_____ energy sources like solar and wind are becoming more cost-effective.",
    "Deforestation leads to loss of _____ and contributes to climate change.",
    "Ocean _____ is threatening marine ecosystems worldwide.",
    "_____ practices in agriculture can help preserve soil health.",
    "The melting of _____ ice caps is causing sea levels to rise.",
    "_____ programs help reduce waste and conserve resources.",
    "Electric vehicles produce zero _____ emissions during operation.",
    "Wetland _____ projects improve water quality and provide wildlife habitats.",
    "The use of _____ packaging materials helps reduce environmental impact."
]

FILL_BLANK_ANSWERS = [
    ["carbon", "greenhouse gas", "CO2"],
    ["renewable", "clean", "sustainable"],
    ["biodiversity", "habitat", "wildlife"],
    ["pollution", "plastic pollution", "contamination"],
    ["sustainable", "eco-friendly", "organic"],
    ["polar", "Arctic", "Antarctic"],
    ["recycling", "waste management", "circular economy"],
    ["tailpipe", "direct", "local"],
    ["restoration", "conservation", "protection"],
    ["biodegradable", "sustainable", "eco-friendly"]
]

def classify_sentence(sentence):
    """Simple rule-based sentence classification"""
    sentence_lower = sentence.lower()
    scores = {}
    
    for category, info in ENVIRONMENTAL_CATEGORIES.items():
        score = 0
        for keyword in info["keywords"]:
            if keyword in sentence_lower:
                score += 1
        scores[category] = score
    
    # Find the category with highest score
    if max(scores.values()) == 0:
        return "General Environmental", 0.5
    
    best_category = max(scores, key=scores.get)
    confidence = min(scores[best_category] / 3, 1.0)  # Normalize confidence
    return best_category, confidence

def extract_entities(text):
    """Simple NER extraction for environmental entities"""
    entities = []
    
    # Environmental pollutants
    pollutants = ["CO2", "carbon dioxide", "methane", "nitrogen oxides", "sulfur dioxide", 
                  "particulate matter", "ozone", "mercury", "lead", "pesticides"]
    
    # Environmental locations
    locations = ["Amazon", "Arctic", "Antarctica", "Pacific Ocean", "Atlantic Ocean", 
                "Sahara Desert", "Great Barrier Reef", "Yellowstone", "rainforest"]
    
    # Environmental concepts
    concepts = ["climate change", "global warming", "deforestation", "biodiversity", 
               "ecosystem", "renewable energy", "sustainability", "conservation"]
    
    # Organizations
    organizations = ["EPA", "NASA", "NOAA", "WWF", "Greenpeace", "IPCC", "UNEP"]
    
    text_lower = text.lower()
    
    for pollutant in pollutants:
        if pollutant.lower() in text_lower:
            entities.append({"text": pollutant, "label": "POLLUTANT"})
    
    for location in locations:
        if location.lower() in text_lower:
            entities.append({"text": location, "label": "LOCATION"})
    
    for concept in concepts:
        if concept.lower() in text_lower:
            entities.append({"text": concept, "label": "CONCEPT"})
    
    for org in organizations:
        if org.lower() in text_lower:
            entities.append({"text": org, "label": "ORGANIZATION"})
    
    return entities

def create_network_graph(entities):
    """Create a network graph from extracted entities"""
    G = nx.Graph()
    
    # Add nodes
    for entity in entities:
        G.add_node(entity["text"], label=entity["label"])
    
    # Add edges based on entity relationships
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities[i+1:], i+1):
            if entity1["label"] != entity2["label"]:
                G.add_edge(entity1["text"], entity2["text"])
    
    return G

def generate_environmental_image_placeholder(prompt):
    """Generate placeholder environmental image description"""
    themes = {
        "forest": "🌲 Lush green forest with diverse wildlife",
        "ocean": "🌊 Crystal clear ocean with coral reefs",
        "renewable": "⚡ Modern solar panels and wind turbines",
        "pollution": "🏭 Industrial area with visible air pollution",
        "climate": "🌡️ Climate change visualization with melting ice",
        "conservation": "🦎 Wildlife conservation area with protected species"
    }
    
    prompt_lower = prompt.lower()
    selected_theme = "forest"  # default
    
    for theme, description in themes.items():
        if theme in prompt_lower:
            selected_theme = theme
            break
    
    return themes[selected_theme]

# Main content based on selected page
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🎯 Project Overview")
        st.markdown("""
        This Environmental AI Hub provides comprehensive analysis tools for environmental data and content:
        
        **Key Features:**
        - **Sentence Classification**: Categorize environmental text into 5 specialized departments
        - **Image Generation**: Create environmental imagery based on text descriptions
        - **NER & Graph Analysis**: Extract and visualize environmental entities and relationships
        - **Fill in the Blanks**: Interactive environmental knowledge testing
        """)
        
        st.markdown("## 📊 Quick Stats")
        col1a, col1b, col1c, col1d = st.columns(4)
        
        with col1a:
            st.markdown("""
            <div class="metric-card">
                <h3>5</h3>
                <p>Categories</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col1b:
            st.markdown("""
            <div class="metric-card">
                <h3>50+</h3>
                <p>Keywords</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col1c:
            st.markdown("""
            <div class="metric-card">
                <h3>10</h3>
                <p>Sample Texts</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col1d:
            st.markdown("""
            <div class="metric-card">
                <h3>4</h3>
                <p>AI Tools</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("## 🔧 Categories")
        for category, info in ENVIRONMENTAL_CATEGORIES.items():
            st.markdown(f"""
            <div class="feature-card">
                <h4 style="color: {info['color']};">{category}</h4>
                <p>{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "📊 Sentence Classification":
    st.markdown("## 📊 Environmental Sentence Classification")
    st.markdown("Classify sentences into environmental categories using AI analysis.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Input Text")
        text_input = st.text_area(
            "Enter text to classify:",
            value="Carbon emissions from factories are causing severe air pollution in urban areas.",
            height=100
        )
        
        use_sample = st.checkbox("Use sample sentences")
        if use_sample:
            selected_sample = st.selectbox("Choose a sample:", SAMPLE_SENTENCES)
            if st.button("Classify Sample"):
                text_input = selected_sample
        
        if st.button("🔍 Classify Text", type="primary"):
            if text_input:
                category, confidence = classify_sentence(text_input)
                
                st.markdown("### 📈 Classification Results")
                
                # Display result with styling
                category_info = ENVIRONMENTAL_CATEGORIES.get(category, {"color": "#95a5a6"})
                st.markdown(f"""
                <div class="feature-card">
                    <h4>Predicted Category: <span style="color: {category_info['color']};">{category}</span></h4>
                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create confidence visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Confidence Score"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': category_info['color']},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 100], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75,
                                       'value': 90}}))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Classification Categories")
        for category, info in ENVIRONMENTAL_CATEGORIES.items():
            st.markdown(f"""
            <div style="background-color: {info['color']}15; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid {info['color']};">
                <h5 style="color: {info['color']}; margin: 0;">{category}</h5>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;">{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Batch classification
    st.markdown("### 📊 Batch Analysis")
    if st.button("Analyze All Samples"):
        results = []
        for sentence in SAMPLE_SENTENCES:
            category, confidence = classify_sentence(sentence)
            results.append({
                "Sentence": sentence[:60] + "..." if len(sentence) > 60 else sentence,
                "Category": category,
                "Confidence": confidence
            })
        
        df = pd.DataFrame(results)
        
        # Category distribution chart
        category_counts = df['Category'].value_counts()
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Distribution of Environmental Categories",
            color_discrete_map={cat: info['color'] for cat, info in ENVIRONMENTAL_CATEGORIES.items()}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Results table
        st.markdown("### 📋 Detailed Results")
        st.dataframe(df, use_container_width=True)

elif page == "🎨 Image Generation":
    st.markdown("## 🎨 Environmental Image Generation")
    st.markdown("Generate environmental images based on text descriptions.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🖼️ Image Generation")
        
        prompt = st.text_area(
            "Enter image description:",
            value="A beautiful forest with diverse wildlife and clean streams",
            height=100
        )
        
        # Style options
        style = st.selectbox(
            "Choose style:",
            ["Realistic", "Artistic", "Satellite View", "Infographic", "Documentary"]
        )
        
        # Quick prompts
        st.markdown("### 🚀 Quick Prompts")
        quick_prompts = [
            "Solar panels on rooftops in a modern city",
            "Plastic pollution in ocean waters",
            "Deforestation in tropical rainforest",
            "Wind turbines in rolling hills",
            "Coral reef ecosystem underwater",
            "Electric vehicle charging station"
        ]
        
        selected_prompt = st.selectbox("Or choose a quick prompt:", [""] + quick_prompts)
        if selected_prompt:
            prompt = selected_prompt
        
        if st.button("🎨 Generate Image", type="primary"):
            with st.spinner("Generating image..."):
                # Simulate image generation
                image_description = generate_environmental_image_placeholder(prompt)
                
                st.markdown("### 🖼️ Generated Image")
                st.info(f"**Generated Image Description:** {image_description}")
                
                # Create a placeholder visualization
                fig = go.Figure()
                fig.add_annotation(
                    text=f"🖼️<br>{image_description}<br><br>Style: {style}",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=2
                )
                fig.update_layout(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=400,
                    title=f"Generated: {prompt[:50]}..."
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Generation Stats")
        
        # Create sample generation statistics
        generation_data = {
            "Theme": ["Forest", "Ocean", "Renewable Energy", "Pollution", "Climate", "Conservation"],
            "Count": [12, 8, 15, 6, 10, 9],
            "Avg Quality": [4.2, 4.5, 4.1, 3.8, 4.0, 4.3]
        }
        
        df_gen = pd.DataFrame(generation_data)
        
        # Bar chart for generation counts
        fig_bar = px.bar(
            df_gen, 
            x="Theme", 
            y="Count",
            title="Images Generated by Theme",
            color="Count",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Quality radar chart
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=df_gen["Avg Quality"],
            theta=df_gen["Theme"],
            fill='toself',
            name='Quality Score'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5])
            ),
            title="Average Quality by Theme",
            height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Recent generations
        st.markdown("### 📈 Recent Generations")
        recent_gens = [
            "Solar farm landscape",
            "Ocean cleanup technology",
            "Urban green spaces",
            "Wildfire prevention",
            "Sustainable agriculture"
        ]
        
        for gen in recent_gens:
            st.markdown(f"- {gen}")

elif page == "🔍 NER & Graph Analysis":
    st.markdown("## 🔍 Named Entity Recognition & Graph Analysis")
    st.markdown("Extract environmental entities and visualize their relationships.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Text Input")
        
        text_input = st.text_area(
            "Enter text for entity extraction:",
            value="The EPA reported that CO2 emissions from factories in the Amazon region are contributing to global warming. NASA satellite data shows deforestation rates increasing, affecting biodiversity in the rainforest ecosystem.",
            height=150
        )
        
        if st.button("🔍 Extract Entities", type="primary"):
            entities = extract_entities(text_input)
            
            if entities:
                st.markdown("### 🏷️ Extracted Entities")
                
                # Group entities by label
                entity_groups = {}
                for entity in entities:
                    label = entity["label"]
                    if label not in entity_groups:
                        entity_groups[label] = []
                    entity_groups[label].append(entity["text"])
                
                # Display entities with color coding
                colors = {
                    "POLLUTANT": "#e74c3c",
                    "LOCATION": "#3498db", 
                    "CONCEPT": "#2ecc71",
                    "ORGANIZATION": "#f39c12"
                }
                
                for label, items in entity_groups.items():
                    st.markdown(f"**{label}s:**")
                    for item in items:
                        color = colors.get(label, "#95a5a6")
                        st.markdown(f"""
                        <span style="background-color: {color}; color: white; padding: 0.3rem 0.8rem; 
                               border-radius: 15px; margin: 0.2rem; display: inline-block;">
                            {item}
                        </span>
                        """, unsafe_allow_html=True)
                
                # Create entity statistics
                entity_stats = pd.DataFrame([
                    {"Type": label, "Count": len(items)} 
                    for label, items in entity_groups.items()
                ])
                
                fig_entities = px.bar(
                    entity_stats,
                    x="Type",
                    y="Count", 
                    title="Entity Distribution",
                    color="Type",
                    color_discrete_map=colors
                )
                st.plotly_chart(fig_entities, use_container_width=True)
                
                # Store entities for graph creation
                st.session_state.entities = entities
    
    with col2:
        st.markdown("### 🕸️ Entity Relationship Graph")
        
        if 'entities' in st.session_state and st.session_state.entities:
            entities = st.session_state.entities
            
            # Create network graph
            G = create_network_graph(entities)
            
            if G.nodes():
                # Get node positions
                pos = nx.spring_layout(G, k=3, iterations=50)
                
                # Create edge traces
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                # Create node traces
                node_x = []
                node_y = []
                node_text = []
                node_color = []
                
                colors = {
                    "POLLUTANT": "#e74c3c",
                    "LOCATION": "#3498db",
                    "CONCEPT": "#2ecc71", 
                    "ORGANIZATION": "#f39c12"
                }
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    
                    # Find node label
                    node_label = "UNKNOWN"
                    for entity in entities:
                        if entity["text"] == node:
                            node_label = entity["label"]
                            break
                    
                    node_color.append(colors.get(node_label, "#95a5a6"))
                
                # Create the plot
                fig = go.Figure()
                
                # Add edges
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                ))
                
                # Add nodes
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition="middle center",
                    textfont=dict(size=10, color="white"),
                    marker=dict(
                        size=30,
                        color=node_color,
                        line=dict(width=2, color="white")
                    )
                ))
                
                fig.update_layout(
                    title="Environmental Entity Relationship Network",
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Node connections show entity relationships",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor="left", yanchor="bottom",
                        font=dict(color="#888", size=12)
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Network statistics
                st.markdown("### 📊 Network Statistics")
                col2a, col2b = st.columns(2)
                
                with col2a:
                    st.metric("Total Entities", len(G.nodes()))
                    st.metric("Connections", len(G.edges()))
                
                with col2b:
                    if G.nodes():
                        avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                        st.metric("Avg Connections", f"{avg_degree:.1f}")
                    st.metric("Entity Types", len(set(e["label"] for e in entities)))
        
        else:
            st.info("Extract entities from text to generate the relationship graph.")

elif page == "📝 Fill in the Blanks":
    st.markdown("## 📝 Environmental Fill in the Blanks")
    st.markdown("Test your environmental knowledge with interactive fill-in-the-blank exercises.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Interactive Exercise")
        
        # Initialize session state
        if 'current_question' not in st.session_state:
            st.session_state.current_question = 0
        if 'score' not in st.session_state:
            st.session_state.score = 0
        if 'answered' not in st.session_state:
            st.session_state.answered = []
        
        # Get current question
        current_idx = st.session_state.current_question
        if current_idx < len(FILL_BLANK_TEMPLATES):
            question = FILL_BLANK_TEMPLATES[current_idx]
            answers = FILL_BLANK_ANSWERS[current_idx]
            
            st.markdown(f"**Question {current_idx + 1} of {len(FILL_BLANK_TEMPLATES)}:**")
            st.markdown(f"*{question}*")
            
            # User input
            user_answer = st.text_input("Your answer:", key=f"answer_{current_idx}")
            
            col1a, col1b, col1c = st.columns([1, 1, 2])
            
            with col1a:
                if st.button("✅ Submit Answer"):
                    if user_answer.strip():
                        is_correct = any(ans.lower() in user_answer.lower() for ans in answers)
                        st.session_state.answered.append({
                            "question": question,
                            "user_answer": user_answer,
                            "correct_answers": answers,
                            "is_correct": is_correct
                        })
                        
                        if is_correct:
                            st.session_state.score += 1
                            st.success(f"✅ Correct! Answer: {answers[0]}")
                        else:
                            st.error(f"❌ Incorrect. Possible answers: {', '.join(answers)}")
                        
                        st.session_state.current_question += 1
                        st.experimental_rerun()
            
            with col1b:
                if st.button("⏭️ Skip"):
                    st.session_state.current_question += 1
                    st.experimental_rerun()
            
            with col1c:
                if st.button("🔄 Reset Quiz"):
                    st.session_state.current_question = 0
                    st.session_state.score = 0
                    st.session_state.answered = []
                    st.experimental_rerun()
            
            # Progress bar
            progress = (current_idx + 1) / len(FILL_BLANK_TEMPLATES)
            st.progress(progress)
            
        else:
            # Quiz completed
            st.markdown("### 🎉 Quiz Completed!")
            final_score = st.session_state.score
            total_questions = len(FILL_BLANK_TEMPLATES)
            percentage = (final_score / total_questions) * 100
            
            st.markdown(f"""
            <div class="feature-card">
                <h3>Final Score: {final_score}/{total_questions} ({percentage:.1f}%)</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=percentage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Quiz Performance"},
                delta={'reference': 80},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 80], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            if st.button("🔄 Restart Quiz"):
                st.session_state.current_question = 0
                st.session_state.score = 0
                st.session_state.answered = []
                st.experimental_rerun()
    
    with col2:
        st.markdown("### 📊 Quiz Statistics")
        
        # Current progress
        current_progress = st.session_state.current_question
        total_questions = len(FILL_BLANK_TEMPLATES)
        current_score = st.session_state.score
        
        st.metric("Progress", f"{current_progress}/{total_questions}")
        st.metric("Current Score", f"{current_score}")
        
        if current_progress > 0:
            accuracy = (current_score / current_progress) * 100
            st.metric("Accuracy", f"{accuracy:.1f}%")
        
        # Topic distribution
        st.markdown("### 🌍 Topics Covered")
        topics = [
            "Climate Change",
            "Renewable Energy", 
            "Pollution Control",
            "Conservation",
            "Sustainability"
        ]
        
        topic_counts = [2, 2, 2, 2, 2]  # Equal distribution
        
        fig_topics = px.pie(
            values=topic_counts,
            names=topics,
            title="Question Topics",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_topics, use_container_width=True)
        
        # Answered questions summary
        if st.session_state.answered:
            st.markdown("### 📋 Review Answers")
            
            for i, ans in enumerate(st.session_state.answered):
                status = "✅" if ans["is_correct"] else "❌"
                st.markdown(f"""
                <div style="background-color: {'#d4edda' if ans['is_correct'] else '#f8d7da'}; 
                           padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;">
                    <strong>{status} Q{i+1}:</strong><br>
                    <small>Your answer: {ans['user_answer']}</small>
                </div>
                """, unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <h4>🌍 Environmental AI Hub</h4>
    <p>Powered by advanced AI for environmental analysis and education</p>
    <p><em>Built with Streamlit & Plotly • Designed for environmental researchers and educators</em></p>
</div>
""", unsafe_allow_html=True)