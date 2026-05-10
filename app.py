import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# ✅ Gunakan path absolute agar tidak ada masalah working directory
@st.cache_resource
def load_model():
    # Ambil direktori tempat app.py berada
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model.joblib')
    
    # Debug: tampilkan path dan cek file
    st.write(f"📂 Looking for model at: `{model_path}`")
    st.write(f"📁 Files in directory: `{os.listdir(base_dir)}`")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file tidak ditemukan di: {model_path}")
        st.stop()
    
    try:
        model = joblib.load(model_path)
        st.success("✅ Model berhasil dimuat!")
        return model
    except AttributeError as e:
        st.error(f"❌ AttributeError - Kemungkinan version mismatch scikit-learn")
        st.error(f"Detail error: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {type(e).__name__}: {str(e)}")
        st.stop()

model = load_model()
