import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import os

# Set page configuration
st.set_page_config(
    page_title="AI Mental Health Sentiment Monitoring System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load TensorFlow models and preprocessors with caching to optimize performance
@st.cache_resource
def load_assets():
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    
    model_path = "upgraded_RNN_model.h5"
    tokenizer_path = "tokenizer.pkl"
    encoder_path = "label_encoder.pkl"
    
    if not (os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(encoder_path)):
        raise FileNotFoundError("Required model files (upgraded_RNN_model.h5, tokenizer.pkl, label_encoder.pkl) not found in the workspace.")
        
    model = load_model(model_path)
    
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
        
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
        
    return model, tokenizer, label_encoder

# Try loading assets
try:
    model, tokenizer, label_encoder = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    loading_error = str(e)

# Emotional guidance details mapped to predictions
GUIDANCE_DATA = {
    'Anger': {
        'status': 'Elevated Frustration & Anger',
        'motivational_message': "It is completely valid to feel angry. Anger is a natural emotion that signals a boundary has been crossed, but how we channel it determines its outcome.",
        'positive_activity': "Engage in high-energy physical activity (like a brisk run, workout, or cleaning), or practice the 4-7-8 breathing technique to soothe the nervous system.",
        'wellness_tips': [
            "Take a step back from the current environment to allow your heart rate to normalize.",
            "Write down your raw feelings without filter, then safely discard the paper as a symbolic release.",
            "Communicate using constructive 'I feel' statements instead of pointing fingers."
        ],
        'color': '#FF4B4B',
        'gradient': 'linear-gradient(135deg, #FF4B4B, #FF7575)',
        'emoji': '⚡'
    },
    'Anxiety': {
        'status': 'Mild to High Anxiety & Uneasiness',
        'motivational_message': "Anxiety feels like an overwhelming storm, but remember that you are the sky, not the weather. This feeling is uncomfortable, but you are safe and it will pass.",
        'positive_activity': "Practice the 5-4-3-2-1 sensory grounding exercise: find 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
        'wellness_tips': [
            "Slightly prolong your exhalations—breathing out longer than breathing in signals safety to your brain.",
            "Reduce caffeine and sugar intake, as they can amplify or mimic physical symptoms of anxiety.",
            "Acknowledge the anxious thoughts as mental chatter rather than absolute truths."
        ],
        'color': '#FFAA00',
        'gradient': 'linear-gradient(135deg, #FFAA00, #FFC95F)',
        'emoji': '🌪️'
    },
    'Depression': {
        'status': 'Persistent Low Mood & Depressive Feelings',
        'motivational_message': "Even when hope feels distant, your value remains unchanged. Be incredibly gentle with yourself. You do not have to climb the whole mountain today; one tiny step is enough.",
        'positive_activity': "Complete one small micro-task: drink a cold glass of water, make your bed, or sit near an open window for fresh air.",
        'wellness_tips': [
            "Break your day into hour-by-hour intervals rather than worrying about the future.",
            "Reach out to a trusted loved one or counselor, even if it is just a simple greeting to break isolation.",
            "Try to get 10-15 minutes of natural morning sunlight to support your mood and circadian rhythm."
        ],
        'color': '#5A6372',
        'gradient': 'linear-gradient(135deg, #5A6372, #8E9AA6)',
        'emoji': '☁️'
    },
    'Happy': {
        'status': 'Positive, Joyful & Mentally Balanced',
        'motivational_message': "What a beautiful headspace to be in! Celebrating and absorbing joyful moments builds long-term mental resilience and creates a positive neurological loop.",
        'positive_activity': "Savor this feeling by journaling about the specific triggers of your happiness, sending a appreciative message to a friend, or listening to your favorite song.",
        'wellness_tips': [
            "Practice gratitude by noting down three specific things that made today bright.",
            "Channel this positive momentum into a creative project or hobby you enjoy.",
            "Use this energy to check in on a friend who might be going through a tough time."
        ],
        'color': '#00C853',
        'gradient': 'linear-gradient(135deg, #00C853, #5AF158)',
        'emoji': '☀️'
    },
    'Panic': {
        'status': 'Acute Panic or High-Alert State',
        'motivational_message': "Panic can feel terrifying, but it is just your body's alarm system misfiring. The adrenaline rush will naturally peak and subside in a few minutes. You are safe.",
        'positive_activity': "Sit down comfortably, press your feet firmly onto the ground, and wrap your arms tightly around yourself to create a physical sense of grounding.",
        'wellness_tips': [
            "Hold an ice cube or splash cold water on your face to activate the vagus nerve and slow your heart rate.",
            "Focus strictly on breathing: inhale slowly for 4 seconds, hold for 4, exhale for 4, hold for 4 (box breathing).",
            "Take a short break and talk with someone you trust."
        ],
        'color': '#D50000',
        'gradient': 'linear-gradient(135deg, #D50000, #FF3D3D)',
        'emoji': '🚨'
    },
    'Sadness': {
        'status': 'Emotional Sadness & Vulnerability',
        'motivational_message': "Sadness is not a weakness; it is a profound testament to your capacity to feel and care. Allowing yourself to feel sad is the first step toward healing.",
        'positive_activity': "Get cozy under a warm blanket, prepare a cup of warm tea, and allow yourself to rest, listen to calm music, or watch a comforting movie.",
        'wellness_tips': [
            "Give yourself permission to cry if needed—it releases stress hormones and acts as a natural emotional detox.",
            "Express your feelings creatively through writing, painting, or whispering them aloud to yourself.",
            "Treat yourself with the same compassion and care you would offer to a dear friend who is grieving."
        ],
        'color': '#2979FF',
        'gradient': 'linear-gradient(135deg, #2979FF, #7DA8FF)',
        'emoji': '🌧️'
    },
    'Stress': {
        'status': 'Cognitive Overload & Stress Tension',
        'motivational_message': "You are carrying a lot, but you don't have to carry it all at once. Pause, set down your mental baggage for a minute, and breathe. You can only do one thing at a time.",
        'positive_activity': "Do a gentle neck and shoulder stretch, declutter your desk area for 5 minutes, or step outside for a brief walk without your phone.",
        'wellness_tips': [
            "Use the 2-minute rule: if a task takes less than 2 minutes, do it now. For larger tasks, write down the single next immediate action.",
            "Establish a clear boundary between 'work time' and 'rest time', disabling notifications when you log off.",
            "Practice deep, diaphragmatic breathing to switch your body from fight-or-flight into rest-and-digest."
        ],
        'color': '#7C4DFF',
        'gradient': 'linear-gradient(135deg, #7C4DFF, #B388FF)',
        'emoji': '⚖️'
    }
}

# Custom CSS for styling the UI professionally
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1e1b4b, #312e81, #1e3a8a);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
        background: linear-gradient(90deg, #ffffff, #e0e7ff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-subtitle {
        font-size: 1.15rem;
        font-weight: 300;
        opacity: 0.9;
        letter-spacing: 0.05em;
        color: #c7d2fe;
    }
    
    /* About section */
    .about-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .about-title {
        font-size: 1.25rem;
        color: #818cf8;
        margin-top: 0;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    /* Suggestion pills */
    .suggestion-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }
    
    /* Result card */
    .result-container {
        border-radius: 16px;
        padding: 2rem;
        color: white;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15), 0 10px 10px -5px rgba(0, 0, 0, 0.15);
        margin-bottom: 2rem;
        transition: transform 0.3s ease;
    }
    .result-container:hover {
        transform: translateY(-2px);
    }
    .result-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        opacity: 0.8;
        margin-bottom: 0.25rem;
    }
    .result-emotion {
        font-size: 2.75rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .result-score {
        font-size: 1.25rem;
        font-weight: 500;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    .result-status {
        background: rgba(255, 255, 255, 0.15);
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        display: inline-block;
        font-size: 0.85rem;
        margin-top: 1rem;
        font-weight: 600;
        backdrop-filter: blur(4px);
    }
    
    /* Guidance cards */
    .guidance-section {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.75rem;
        margin-top: 1.5rem;
    }
    .guidance-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .guidance-card {
        background-color: rgba(255, 255, 255, 0.02);
        border-left: 4px solid #818cf8;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .guidance-card-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .guidance-card-content {
        color: #f1f5f9;
        font-size: 1.05rem;
        line-height: 1.5;
    }
    .tip-item {
        margin-bottom: 0.5rem;
        display: flex;
        gap: 0.5rem;
        align-items: flex-start;
        color: #e2e8f0;
    }
    .tip-bullet {
        color: #818cf8;
        font-weight: bold;
    }
    
    /* Placeholder info box */
    .placeholder-box {
        border: 2px dashed rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        color: #94a3b8;
    }
    
    /* Custom spacing */
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        background-color: rgba(255, 255, 255, 0.02);
        color: #f8fafc;
        font-size: 1.05rem;
        transition: border-color 0.3s;
    }
    .stTextArea textarea:focus {
        border-color: #818cf8;
        box-shadow: 0 0 0 1px #818cf8;
    }
</style>
""", unsafe_allow_html=True)

# SECTION 1 — Header
st.markdown("""
<div class="header-container">
    <div class="header-title">AI-Based Mental Health Sentiment Monitoring System</div>
    <div class="header-subtitle">Emotion Detection using Simple Recurrent Neural Networks</div>
</div>
""", unsafe_allow_html=True)

# Handle error loading assets
if not assets_loaded:
    st.error("### ⚠️ Model Load Failure")
    st.write("The Streamlit application could not load the pre-trained model files. Please verify that the files exist in the same directory.")
    st.info(f"**Error Details:** {loading_error}")
    st.stop()

# Initialize session state for user input if not present
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

# Define suggestions and their target feelings to showcase
suggestions = [
    {
        "label": "🌱 Serenity / Peace",
        "text": "I walked along the beach and found peace in the sound of the waves. It was a good day."
    },
    {
        "label": "⚡ Stress & Overwhelm",
        "text": "I feel completely overwhelmed, irritated, and stressed with my workload. I cannot sleep."
    },
    {
        "label": "🌧️ Sadness / Empty",
        "text": "Lately, I feel so drained, empty, and just want to sleep all day. Everything feels heavy."
    }
]

# Set up main columns
col_input, col_output = st.columns([1.1, 0.9], gap="large")

with col_input:
    # SECTION 2 — About the Project
    with st.expander("ℹ️ About the Project & Technology Stack", expanded=False):
        st.markdown("""
        <div class="about-card">
            <div class="about-title">Importance of Emotional AI</div>
            <p style="color:#cbd5e1; font-size:0.95rem; line-height:1.5; margin-bottom:1rem;">
                Emotional AI (Affective Computing) enables machines to detect, interpret, and respond to human emotional states. In mental health contexts, monitoring emotional sentiment in written thoughts can act as an early flag for stress, anxiety, or depression, assisting individuals and clinicians in tracking emotional wellbeing.
            </p>
            
            <div class="about-title">NLP Applications</div>
            <p style="color:#cbd5e1; font-size:0.95rem; line-height:1.5; margin-bottom:1rem;">
                Natural Language Processing (NLP) translates unstructured text into machine-readable formats. By mapping words to numerical representations (embeddings) and analyzing context, NLP algorithms capture semantic meaning, enabling fine-grained emotion classification.
            </p>
            
            <div class="about-title">Role of RNNs in Sequence Learning</div>
            <p style="color:#cbd5e1; font-size:0.95rem; line-height:1.5; margin-bottom:0;">
                Recurrent Neural Networks (RNNs) are designed for sequential data like text. Unlike standard feedforward neural networks, RNNs possess "memory" (internal feedback loops) that allows them to process words in relation to previous words, making them highly effective at understanding context and order in sequence sentiment analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("📝 Express Your Thoughts")
    
    # SECTION 3 — User Text Input Area
    # Create the text area linked to session state
    user_input = st.text_area(
        label="Share what's on your mind. Your text will be analyzed locally and private.",
        value=st.session_state.user_text,
        placeholder="Enter your thoughts or feelings here...",
        height=180,
        label_visibility="collapsed"
    )
    
    # Render suggestion chips
    st.markdown('<div class="suggestion-label">Suggested phrases to try:</div>', unsafe_allow_html=True)
    sug_cols = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        with sug_cols[i]:
            if st.button(sug["label"], key=f"sug_btn_{i}", use_container_width=True):
                st.session_state.user_text = sug["text"]
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # SECTION 4 — Prediction Button
    analyze_clicked = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)

with col_output:
    st.subheader("📊 Analysis Results")
    
    # Trigger analysis on click OR if the input is changed via suggestion selection
    # Wait, we want to analyze if either the button is clicked, or if user_input has content and a suggestion was just clicked
    should_analyze = analyze_clicked or (st.session_state.user_text != "" and user_input == st.session_state.user_text)
    
    if should_analyze and user_input.strip() != "":
        with st.spinner("Analyzing text patterns using RNN model..."):
            # Preprocess the input
            # 1. texts_to_sequences
            seq = tokenizer.texts_to_sequences([user_input])
            
            # 2. pad_sequences (maxlen=50, matching model architecture)
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            padded = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
            
            # 3. Predict sentiment probability distribution
            predictions = model.predict(padded, verbose=0)[0]
            
            # 4. Extract labels and confidences
            classes = label_encoder.classes_
            predicted_index = np.argmax(predictions)
            predicted_emotion = classes[predicted_index]
            confidence_score = predictions[predicted_index]
            
            # Get guidance data
            g_data = GUIDANCE_DATA.get(predicted_emotion, {
                'status': 'Unknown Status',
                'motivational_message': 'No guidance available for this emotion.',
                'positive_activity': 'Engage in self-care.',
                'wellness_tips': ['Take a deep breath.'],
                'color': '#818cf8',
                'gradient': 'linear-gradient(135deg, #1e3a8a, #3b82f6)',
                'emoji': '🧠'
            })
            
            # SECTION 5 — Prediction Output
            st.markdown(f"""
            <div class="result-container" style="background: {g_data['gradient']};">
                <div class="result-label">Predominant Emotion Detected</div>
                <div class="result-emotion">{g_data['emoji']} {predicted_emotion}</div>
                <div class="result-score">Confidence Score: {confidence_score * 100:.1f}%</div>
                <div class="result-status">Status: {g_data['status']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a dataframe for visualization
            prob_df = pd.DataFrame({
                'Emotion': classes,
                'Probability': predictions * 100
            }).sort_values(by='Probability', ascending=True)
            
            # SECTION 6 — Visualization Area (horizontal probability distribution graph)
            # Create a beautifully styled Plotly chart
            fig = px.bar(
                prob_df,
                x='Probability',
                y='Emotion',
                orientation='h',
                labels={'Probability': 'Confidence (%)', 'Emotion': 'Emotional State'},
                title='Sentiment Confidence Chart',
                color='Probability',
                color_continuous_scale=[
                    [0, '#1e1b4b'],     # deep dark indigo
                    [0.5, '#4f46e5'],   # medium indigo
                    [1, g_data['color']] # active emotion color accent
                ]
            )
            
            # Adjust styling of the chart
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                height=260,
                coloraxis_showscale=False,
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.05)',
                    title_font=dict(color='#94a3b8', size=11),
                    tickfont=dict(color='#cbd5e1'),
                    range=[0, 100]
                ),
                yaxis=dict(
                    title_font=dict(color='#94a3b8', size=11),
                    tickfont=dict(color='#cbd5e1')
                ),
                title_font=dict(color='#f1f5f9', size=14, family='Outfit')
            )
            fig.update_traces(
                hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
                marker_line_color='rgba(255,255,255,0.15)',
                marker_line_width=1
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # SECTION 7 — Emotional Guidance Area
            st.markdown(f"""
            <div class="guidance-section">
                <div class="guidance-title">🌱 Emotional Guidance & Action Plan</div>
                
                <div class="guidance-card" style="border-left-color: {g_data['color']};">
                    <div class="guidance-card-title">Motivational Message</div>
                    <div class="guidance-card-content">"{g_data['motivational_message']}"</div>
                </div>
                
                <div class="guidance-card" style="border-left-color: {g_data['color']};">
                    <div class="guidance-card-title">Suggested Positive Activity</div>
                    <div class="guidance-card-content">{g_data['positive_activity']}</div>
                </div>
                
                <div style="margin-top: 1rem; padding: 0.5rem 0 0 0;">
                    <div class="guidance-card-title" style="margin-bottom: 0.5rem;">Emotional Wellness Tips</div>
                    {"".join(f'<div class="tip-item"><span class="tip-bullet" style="color: {g_data["color"]};">•</span><div>{tip}</div></div>' for tip in g_data['wellness_tips'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        # Initial placeholder state
        if user_input.strip() == "":
            st.markdown("""
            <div class="placeholder-box">
                <h3 style="margin-top:0; color:#e2e8f0; font-family:'Outfit'; font-weight:500;">Ready for Sentiment Analysis</h3>
                <p style="margin-bottom:0; font-size:0.95rem;">Enter your feelings in the text area on the left and click <b>Analyze Emotion</b>, or click one of the suggested phrases to observe how the Simple RNN model makes predictions.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # User cleared input, or click analyze but no text
            st.info("Please enter some text in the thoughts area to analyze.")
