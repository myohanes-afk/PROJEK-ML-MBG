import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

st.set_page_config(page_title="Analisis Sentimen MBG", page_icon="🍽️", layout="centered")

st.title("🍽️ Aplikasi Analisis Sentimen Program Makan Bergizi Gratis")
st.write("Aplikasi ini memprediksi sentimen masyarakat menggunakan model Support Vector Machine (SVM).")

@st.cache_resource
def load_model():
    with open('svm_tuned_mbg.pkl', 'rb') as f:
        loaded_object = pickle.load(f)
    
    # Ambil estimator terbaik jika dibungkus GridSearchCV/RandomizedSearchCV
    if hasattr(loaded_object, 'best_estimator_'):
        return loaded_object.best_estimator_
    return loaded_object

try:
    model_pipeline = load_model()
    
    user_tweet = st.text_area("Masukkan teks tweet/komentar di sini:", placeholder="Tulis pendapat di sini...")

    if st.button("Analisis Sentimen", type="primary"):
        if user_tweet.strip() != "":
            # 1. Membersihkan teks
            cleaned_tweet = re.sub(r'http\S+|www\.\S+', '', user_tweet)
            cleaned_tweet = re.sub(r'@\w+', '', cleaned_tweet)
            cleaned_tweet = cleaned_tweet.lower()
            
            # 2. Mendapatkan probabilitas semua kelas
            proba = model_pipeline.predict_proba([cleaned_tweet])[0]
            classes = model_pipeline.classes_
            
            # 3. Menentukan kelas dengan probabilitas tertinggi (Prediksi Utama)
            import numpy as np
            best_class_idx = np.argmax(proba)
            prediction = classes[best_class_idx]
            
            # 4. Menampilkan Hasil Utama
            st.subheader("Hasil Analisis:")
            if prediction == 'Positif':
                st.success(f"Sentimen Dominan: **{prediction}** 😊🟢")
            elif prediction == 'Negatif':
                st.error(f"Sentimen Dominan: **{prediction}** 😡🔴")
            else:
                st.warning(f"Sentimen Dominan: **{prediction}** 😐🟡")
                
            st.divider() # Garis pembatas
            
            # 5. Menampilkan Detail Persentase Setiap Kelas
            st.markdown("### 📊 Detail Probabilitas Model")
            
            # Membuat perulangan untuk menampilkan setiap kelas dan persentasenya
            for i, class_name in enumerate(classes):
                percentage = proba[i] * 100
                
                # Mengatur warna dan emoji berdasarkan nama kelas
                if class_name == 'Positif':
                    label = f"🟢 Positif: {percentage:.2f}%"
                elif class_name == 'Negatif':
                    label = f"🔴 Negatif: {percentage:.2f}%"
                else:
                    label = f"🟡 Netral: {percentage:.2f}%"
                
                # Menampilkan teks persentase
                st.write(label)
                # Menampilkan visualisasi bar (harus bernilai antara 0.0 sampai 1.0)
                st.progress(float(proba[i]))
                
        else:
            st.error("Silakan masukkan teks terlebih dahulu!")

except Exception as e:
    st.error(f"Error memuat model: {e}")
