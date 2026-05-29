import streamlit as st
from pathlib import Path
import os

st.title("Iframe Test")

html_path = "graph.html"
if os.path.exists(html_path):
    st.write("Embedding as Path:")
    st.iframe(Path(html_path), height=400)
    
    st.write("Embedding as String:")
    with open(html_path, "r", encoding="utf-8") as f:
        html_str = f.read()
    st.iframe(html_str, height=400)
else:
    st.write("graph.html not found!")
