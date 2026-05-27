import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import os
import re
import nltk

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

# Emotional guidance details mapped to the 7 real trained classes
GUIDANCE_DATA = {
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
    'Bipolar': {
        'status': 'Significant Mood Oscillations',
        'motivational_message': "Navigating extreme emotional shifts is incredibly challenging. Keep anchor points in your day; small, consistent routines can help stabilize your energy.",
        'positive_activity': "Create a daily mood and sleep log, or engage in a calm, structured activity like coloring, building blocks, or sorting.",
        'wellness_tips': [
            "Maintain a regular sleep schedule, as sleep disruptions can trigger mood instability.",
            "Break down goals into tiny, predictable steps to prevent feeling overwhelmed or overstimulated.",
            "Have a pre-arranged action plan with family, friends, or a therapist for intense moods."
        ],
        'color': '#AB47BC',
        'gradient': 'linear-gradient(135deg, #AB47BC, #E1BEE7)',
        'emoji': '🌓'
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
    'Normal': {
        'status': 'Stable, Joyful & Mentally Balanced',
        'motivational_message': "What a beautiful headspace to be in! Celebrating and absorbing stable, balanced moments builds long-term mental resilience and creates a positive neurological loop.",
        'positive_activity': "Savor this feeling by journaling about the specific triggers of your stable mood, sending an appreciative message to a friend, or listening to your favorite song.",
        'wellness_tips': [
            "Practice gratitude by noting down three specific things that made today bright.",
            "Channel this positive momentum into a creative project or hobby you enjoy.",
            "Use this energy to check in on a friend who might be going through a tough time."
        ],
        'color': '#00C853',
        'gradient': 'linear-gradient(135deg, #00C853, #5AF158)',
        'emoji': '☀️'
    },
    'Personality disorder': {
        'status': 'Complex Emotional Intensity',
        'motivational_message': "Your emotions might feel intense and rapidly changing. Remember that emotions are information, not directives. You can observe them without having to react immediately.",
        'positive_activity': "Use temperature therapy: wash your face with ice-cold water or hold a warm cup of tea to calm the physical response.",
        'wellness_tips': [
            "Practice the 'STOP' technique: Stop, Take a step back, Observe your feelings, and Proceed mindfully.",
            "Engage in non-judgmental journaling—let your thoughts flow onto paper without editing them.",
            "Focus on grounding yourself in your physical body through deep breaths or stretching."
        ],
        'color': '#00ACC1',
        'gradient': 'linear-gradient(135deg, #00ACC1, #80DEEA)',
        'emoji': '🎭'
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
    },
    'Suicidal': {
        'status': 'Critical Distress / High Crisis State',
        'motivational_message': "Please know that your pain is real, but there is help and you do not have to walk through this darkness alone. There are people who want to listen and support you.",
        'positive_activity': "Connect immediately with someone you trust, or call/text a mental health support line (like 988 in the US/Canada or your local crisis helpline).",
        'wellness_tips': [
            "Reach out to a professional immediately: call or text the Suicide & Crisis Lifeline at 988.",
            "Remove yourself from any immediate risk or tools, and move to a safe, public, or shared space.",
            "Take a short break and talk with someone you trust."
        ],
        'color': '#D50000',
        'gradient': 'linear-gradient(135deg, #D50000, #FF3D3D)',
        'emoji': '🚨'
    }
}

# Custom CSS for styling the UI professionally
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Base App Custom Overrides */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0d0f18 0%, #05060b 100%) !important;
        color: #f8fafc !important;
        overflow-x: hidden;
    }
    
    /* Background Ambient Glowing Orbs */
    .stApp::before {
        content: '';
        position: absolute;
        top: -150px;
        left: -150px;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, rgba(99, 102, 241, 0) 70%);
        filter: blur(90px);
        pointer-events: none;
        z-index: 0;
        animation: orb-bounce-1 25s infinite alternate ease-in-out;
    }
    .stApp::after {
        content: '';
        position: absolute;
        bottom: -150px;
        right: -150px;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.1) 0%, rgba(168, 85, 247, 0) 70%);
        filter: blur(100px);
        pointer-events: none;
        z-index: 0;
        animation: orb-bounce-2 30s infinite alternate ease-in-out;
    }
    
    @keyframes orb-bounce-1 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(80px, 120px) scale(1.15); }
        100% { transform: translate(-40px, -40px) scale(0.9); }
    }
    @keyframes orb-bounce-2 {
        0% { transform: translate(0, 0) scale(1.1); }
        50% { transform: translate(-100px, -80px) scale(0.85); }
        100% { transform: translate(60px, 40px) scale(1.05); }
    }
    
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Keyframe Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(24px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-6px) rotate(2deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    @keyframes pulseShadow {
        0% { box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3), 0 0 10px var(--emotion-color, rgba(129, 140, 248, 0.15)); }
        50% { box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4), 0 0 25px var(--emotion-color, rgba(129, 140, 248, 0.35)); }
        100% { box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3), 0 0 10px var(--emotion-color, rgba(129, 140, 248, 0.15)); }
    }
    
    .fade-in-section {
        animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .floating-emoji {
        display: inline-block;
        animation: float 4s ease-in-out infinite;
    }
    
    /* Header Container styling (Premium Glassmorphism) */
    .header-container {
        position: relative;
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        padding: 3.5rem 2rem;
        border-radius: 28px;
        color: white;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 50px -15px rgba(0, 0, 0, 0.6), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        overflow: hidden;
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(129, 140, 248, 0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.85rem;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 30%, #c7d2fe 70%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.85rem;
    }
    .header-subtitle {
        font-size: 1.25rem;
        font-weight: 400;
        letter-spacing: 0.05em;
        color: #94a3b8;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.5;
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(129, 140, 248, 0.12);
        border: 1px solid rgba(129, 140, 248, 0.25);
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #a5b4fc;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1.25rem;
    }
    
    /* 3D Glassmorphic Cards (About cards, results, guidance) */
    .card-3d {
        position: relative;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        transform-style: preserve-3d;
        perspective: 1000px;
        transition: transform 0.4s cubic-bezier(0.2, 0.8, 0.2, 1), border-color 0.4s ease, box-shadow 0.4s ease;
    }
    .card-3d:hover {
        transform: translateY(-8px) rotateX(3deg) rotateY(-3deg) scale3d(1.01, 1.01, 1.01);
        border-color: rgba(129, 140, 248, 0.25);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 20px rgba(129, 140, 248, 0.08);
    }
    
    /* About section expander styling */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.01) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 18px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2) !important;
        overflow: hidden !important;
        backdrop-filter: blur(8px);
        margin-bottom: 1.5rem !important;
    }
    div[data-testid="stExpander"] details {
        border: none !important;
    }
    div[data-testid="stExpander"] summary {
        background: rgba(255, 255, 255, 0.02) !important;
        padding: 0.9rem 1.25rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        transition: background 0.3s, color 0.3s;
    }
    div[data-testid="stExpander"] summary:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #818cf8 !important;
    }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 1.25rem !important;
    }
    
    .about-container {
        display: flex;
        flex-direction: column;
        gap: 1.25rem;
    }
    .about-card {
        background: rgba(255, 255, 255, 0.01);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 14px;
        padding: 1.25rem;
        transition: transform 0.3s cubic-bezier(0.2, 0.8, 0.2, 1), border-color 0.3s ease;
    }
    .about-card:hover {
        transform: translateY(-4px) scale(1.005);
        border-color: rgba(129, 140, 248, 0.15);
    }
    .about-card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }
    .about-card-icon {
        background: rgba(129, 140, 248, 0.1);
        border-radius: 10px;
        width: 38px;
        height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        color: #818cf8;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .about-card-title {
        font-size: 1.15rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: #e2e8f0;
        margin: 0;
    }
    .about-card-text {
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Input Area & Forms Overrides */
    .stTextArea textarea {
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: rgba(15, 23, 42, 0.4) !important;
        color: #f8fafc !important;
        font-size: 1.05rem !important;
        padding: 1.25rem !important;
        line-height: 1.6 !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        transition: border-color 0.3s, box-shadow 0.3s, background-color 0.3s !important;
        backdrop-filter: blur(8px);
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2), 0 8px 24px rgba(99, 102, 241, 0.15) !important;
        background-color: rgba(15, 23, 42, 0.6) !important;
    }
    
    /* Primary Action Button (Tactile 3D style) */
    .stButton>button[kind="primary"] {
        border-radius: 14px !important;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: white !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 0.8rem 2rem !important;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stButton>button[kind="primary"]:hover {
        transform: translateY(-3px) scale(1.01) !important;
        box-shadow: 0 12px 30px rgba(79, 70, 229, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    .stButton>button[kind="primary"]:active {
        transform: translateY(1px) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    }
    
    /* Secondary Action Button (Suggestion Tags) */
    .stButton>button[kind="secondary"] {
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        color: #cbd5e1 !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.25s ease-in-out !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
        backdrop-filter: blur(8px);
    }
    .stButton>button[kind="secondary"]:hover {
        transform: translateY(-2px) rotate(1deg) !important;
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(129, 140, 248, 0.3) !important;
        color: #ffffff !important;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.25), 0 0 10px rgba(129, 140, 248, 0.1) !important;
    }
    .stButton>button[kind="secondary"]:active {
        transform: translateY(1px) !important;
    }
    
    .suggestion-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    /* Analysis Results Layout */
    .result-container {
        border-radius: 24px;
        padding: 2.25rem;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 2rem;
        transition: transform 0.4s cubic-bezier(0.2, 0.8, 0.2, 1);
        transform-style: preserve-3d;
        perspective: 1000px;
        animation: pulseShadow 4s infinite ease-in-out;
    }
    .result-container:hover {
        transform: translateY(-6px) rotateX(2deg) rotateY(-2deg);
    }
    .result-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        opacity: 0.85;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }
    .result-emotion {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.1;
        letter-spacing: -0.02em;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));
    }
    .result-score {
        font-size: 1.35rem;
        font-weight: 600;
        margin-top: 0.75rem;
        opacity: 0.95;
    }
    .result-status {
        background: rgba(255, 255, 255, 0.18);
        padding: 0.6rem 1.4rem;
        border-radius: 9999px;
        display: inline-block;
        font-size: 0.85rem;
        margin-top: 1.25rem;
        font-weight: 700;
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Guidance Layout & Section Styling */
    .guidance-section {
        background-color: rgba(255, 255, 255, 0.015);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 24px;
        padding: 2.25rem;
        margin-top: 1.5rem;
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    .guidance-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 1.75rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        letter-spacing: -0.01em;
    }
    .guidance-card {
        background: rgba(255, 255, 255, 0.01);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-left: 5px solid #818cf8;
        padding: 1.5rem;
        border-radius: 8px 20px 20px 8px;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s cubic-bezier(0.2, 0.8, 0.2, 1), background-color 0.3s, border-color 0.3s, box-shadow 0.3s;
    }
    .guidance-card:hover {
        transform: translateX(6px) scale(1.005);
        background: rgba(255, 255, 255, 0.025);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
    }
    .guidance-card-title {
        font-weight: 700;
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .guidance-card-content {
        color: #f8fafc;
        font-size: 1.1rem;
        line-height: 1.6;
        font-weight: 400;
        font-style: italic;
    }
    .activity-card-content {
        color: #f8fafc;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .tip-box {
        background: rgba(15, 23, 42, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.03);
        padding: 1.5rem;
        border-radius: 18px;
        margin-top: 1.5rem;
    }
    .tip-item {
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
        align-items: flex-start;
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.5;
    }
    .tip-item:last-child {
        margin-bottom: 0;
    }
    .tip-icon {
        border-radius: 50%;
        width: 26px;
        height: 26px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        flex-shrink: 0;
        margin-top: 2px;
        font-weight: 700;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    
    /* Placeholder info box */
    .placeholder-box {
        border: 2px dashed rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 4.5rem 2.5rem;
        text-align: center;
        color: #64748b;
        background-color: rgba(255, 255, 255, 0.005);
        backdrop-filter: blur(8px);
        transition: border-color 0.3s;
    }
    .placeholder-box:hover {
        border-color: rgba(129, 140, 248, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# SECTION 1 — Header
st.markdown("""<div class="header-container">
<div class="header-badge">⚡ Interactive AI Platform</div>
<div class="header-title"><span class="floating-emoji">🧠</span> AI-Based Mental Health Sentiment Monitoring System</div>
<div class="header-subtitle">Analyze emotional markers and sequence sentiment in written thoughts using deep learning Recurrent Neural Networks</div>
</div>""", unsafe_allow_html=True)

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
        "label": "🌱 Balanced & Peaceful",
        "text": "I walked along the beach and found peace in the sound of the waves. It was a good day."
    },
    {
        "label": "🌪️ Anxious & Worried",
        "text": "I feel completely anxious and worried about my future. My mind is constantly racing."
    },
    {
        "label": "🌧️ Tired & Hopeless",
        "text": "I feel tired, hopeless, and empty. I do not enjoy anything anymore and just want to sleep."
    }
]

# Set up main columns
col_input, col_output = st.columns([1.1, 0.9], gap="large")

with col_input:
    # SECTION 2 — About the Project
    with st.expander("ℹ️ About the Project & Technology Stack", expanded=False):
        st.markdown("""<div class="about-container">
<div class="about-card card-3d">
<div class="about-card-header">
<div class="about-card-icon">🧠</div>
<h4 class="about-card-title">Importance of Emotional AI</h4>
</div>
<p class="about-card-text">
Emotional AI (Affective Computing) enables machines to detect, interpret, and respond to human emotional states. In mental health contexts, monitoring emotional sentiment in written thoughts can act as an early flag for stress, anxiety, or depression, assisting individuals and clinicians in tracking emotional wellbeing.
</p>
</div>
<div class="about-card card-3d">
<div class="about-card-header">
<div class="about-card-icon">💻</div>
<h4 class="about-card-title">NLP Applications</h4>
</div>
<p class="about-card-text">
Natural Language Processing (NLP) translates unstructured text into machine-readable formats. By mapping words to numerical representations (embeddings) and analyzing context, NLP algorithms capture semantic meaning, enabling fine-grained emotion classification.
</p>
</div>
<div class="about-card card-3d">
<div class="about-card-header">
<div class="about-card-icon">🔄</div>
<h4 class="about-card-title">Role of RNNs in Sequence Learning</h4>
</div>
<p class="about-card-text">
Recurrent Neural Networks (RNNs) are designed for sequential data like text. Unlike standard feedforward neural networks, RNNs possess "memory" (internal feedback loops) that allows them to process words in relation to previous words, making them highly effective at understanding context and order in sequence sentiment analysis.
</p>
</div>
</div>""", unsafe_allow_html=True)
        
    st.subheader("📝 Express Your Thoughts")
    
    # SECTION 3 — User Text Input Area
    # Create the text area linked to session state
    user_input = st.text_area(
        label="Share what's on your mind. Your text will be analyzed locally and privately.",
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
    should_analyze = analyze_clicked or (st.session_state.user_text != "" and user_input == st.session_state.user_text)
    
    if should_analyze and user_input.strip() != "":
        with st.spinner("Analyzing text patterns using RNN model..."):
            # Preprocess the input
            # Load stopwords list
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            
            try:
                stop_words = set(stopwords.words('english'))
            except:
                nltk.download('stopwords')
                nltk.download('punkt')
                stop_words = set(stopwords.words('english'))
                
            clean_text = str(user_input).lower()
            clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)
            try:
                tokens = word_tokenize(clean_text)
            except:
                tokens = clean_text.split()
            tokens = [w for w in tokens if w not in stop_words]
            processed_text = ' '.join(tokens)
            
            # 1. texts_to_sequences
            seq = tokenizer.texts_to_sequences([processed_text])
            
            # 2. pad_sequences (maxlen=50, matching model architecture)
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            padded = pad_sequences(seq, maxlen=50, padding='pre', truncating='pre')
            
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
            <div class="result-container" style="--emotion-color: {g_data['color']}; background: {g_data['gradient']};">
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
            st.markdown(f"""<div class="guidance-section">
<div class="guidance-title">🌱 Emotional Guidance & Action Plan</div>
<div class="guidance-card card-3d" style="border-left-color: {g_data['color']};">
<div class="guidance-card-title" style="color: {g_data['color']};">
<span>✨</span> MOTIVATIONAL MESSAGE
</div>
<div class="guidance-card-content">"{g_data['motivational_message']}"</div>
</div>
<div class="guidance-card card-3d" style="border-left-color: {g_data['color']};">
<div class="guidance-card-title" style="color: {g_data['color']};">
<span>🎯</span> SUGGESTED POSITIVE ACTIVITY
</div>
<div class="activity-card-content">{g_data['positive_activity']}</div>
</div>
<div class="tip-box">
<div class="guidance-card-title" style="margin-bottom: 0.75rem; color: #a5b4fc;">
<span>💡</span> EMOTIONAL WELLNESS TIPS
</div>
{"".join(f'''<div class="tip-item">
<div class="tip-icon" style="color: {g_data['color']}; background: {g_data['color']}20;">✓</div>
<div>{tip}</div>
</div>''' for tip in g_data['wellness_tips'])}
</div>
</div>""", unsafe_allow_html=True)
            
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
