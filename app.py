import streamlit as st
import pandas as pd
import re
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import fitz  # PyMuPDF
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords
import nltk
import numpy as np
from wordcloud import WordCloud
import os
import shutil # Untuk menghapus folder sementara jika ada
from streamlit_navigation_bar import st_navbar


# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="CV-ATS Checker Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI & VARIABEL GLOBAL ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

@st.cache_data
def get_stopwords():
    """Menggabungkan stopwords Bahasa Inggris dan Bahasa Indonesia."""
    stop_words_en = set(stopwords.words('english'))
    factory = StopWordRemoverFactory()
    stop_words_id = set(factory.get_stop_words())
    return stop_words_en.union(stop_words_id)

combined_stopwords = get_stopwords()

def preprocess(text):
    """Fungsi preprocessing teks untuk membersihkan data."""
    if pd.isnull(text): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in combined_stopwords]
    return " ".join(tokens)

# --- Pemuatan Data & Preprocessing (Cached) ---
@st.cache_data
def load_and_preprocess_data():
    """Memuat, membersihkan, dan melabeli ulang dataset CV."""
    try:
        df = pd.read_csv('CV-ATS.csv')
    except FileNotFoundError:
        st.error("File 'CV-ATS.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        st.stop()
    
    def extract_label(text):
        try:
            json_data = ast.literal_eval(text)
            return json_data[0]['labels'][0] if json_data and 'labels' in json_data[0] and json_data[0]['labels'] else "Unknown"
        except:
            return "Unknown"
            
    df['label_extracted'] = df['label'].apply(extract_label)
    df['label_binary'] = df['label_extracted'].replace({'ATS_friendly_with_issues': 'ATS_friendly'})
    df_binary = df[df['label_binary'].isin(['ATS_friendly', 'Not_ATS_friendly'])].copy()
    df_binary['clean_text'] = df_binary['text'].apply(preprocess)
    df_binary.dropna(subset=['clean_text'], inplace=True)
    return df_binary

df = load_and_preprocess_data()

# --- Pelatihan Model (Cached) ---
@st.cache_resource
def train_models():
    """Melatih model klasifikasi dan mengembalikan objek yang relevan."""
    if df.empty:
        st.error("Dataset kosong setelah preprocessing. Tidak dapat melatih model.")
        st.stop()

    X = df['clean_text']
    y = df['label_binary']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    
    models = {
        'Multinomial Naive Bayes': MultinomialNB().fit(X_train_tfidf, y_train),
        'SVM Linear': SVC(kernel='linear', probability=True, random_state=42).fit(X_train_tfidf, y_train)
    }
    
    return models, tfidf_vectorizer, X_test, y_test

models, tfidf_vectorizer, X_test, y_test = train_models()
best_model = models['SVM Linear']
best_model_name = 'SVM Linear'

# --- Fungsi Bantuan untuk Visualisasi ---
@st.cache_data
def generate_wordcloud(text_series):
    """Menghasilkan wordcloud dari seri teks."""
    text = " ".join(review for review in text_series)
    wordcloud = WordCloud(stopwords=list(combined_stopwords), background_color="white", colormap='viridis', width=800, height=400).generate(text)
    return wordcloud

@st.cache_data
def generate_tsne_plot(_tfidf_vectorizer, _df_binary):
    """Menghasilkan plot t-SNE dari data yang sudah di-vectorize."""
    tfidf_matrix = _tfidf_vectorizer.transform(_df_binary['clean_text'])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=300)
    tsne_results = tsne.fit_transform(tfidf_matrix.toarray())
    
    df_tsne = pd.DataFrame({
        'x': tsne_results[:,0],
        'y': tsne_results[:,1],
        'label': _df_binary['label_binary']
    })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x="x", y="y", hue="label", palette={"ATS_friendly": "#3498db", "Not_ATS_friendly": "#e74c3c"}, data=df_tsne, legend="full", alpha=0.7, ax=ax)
    ax.set_title('Visualisasi Cluster Dokumen CV dengan t-SNE')
    return fig

def plot_feature_importance(df_coef, title, palette):
    """Fungsi pembantu untuk membuat bar plot fitur terpenting."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Bobot', y='Fitur', data=df_coef, ax=ax, palette=palette)
    ax.set_title(title)
    ax.set_xlabel("Bobot (Pengaruh)")
    ax.set_ylabel("Fitur (Kata)")
    return fig

# --- Tampilan Antarmuka (UI) ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Tentang Proyek ℹ️", "Ringkasan Data 📊", "Performa Model 📈", "Cek CV-ATS 🔍"])

st.title("🤖 Dashboard Analisis CV-ATS")
st.markdown("Analisis mendalam mengenai klasifikasi CV yang ramah *Applicant Tracking System* (ATS) menggunakan machine learning.")

# --- KONTEN HALAMAN ---
if page == "Tentang Proyek ℹ️":
    st.header("ℹ️ Tentang Proyek Ini")
    st.markdown("""
    Proyek ini bertujuan untuk membangun sebuah sistem cerdas yang dapat mengklasifikasikan Curriculum Vitae (CV) sebagai **ATS Friendly** atau **Not ATS Friendly**. Applicant Tracking System (ATS) adalah perangkat lunak yang digunakan oleh banyak perusahaan untuk menyaring CV secara otomatis. CV yang tidak "ramah" terhadap ATS seringkali gagal dibaca dengan benar, sehingga kandidat yang berkualitas bisa terlewatkan.
    
    **Metodologi yang digunakan:**
    1.  **Pengumpulan Data:** Dataset CV dikumpulkan dan dilabeli secara manual.
    2.  **Preprocessing Teks:** Teks dari setiap CV dibersihkan dari noise seperti tanda baca, angka, dan kata-kata umum (stopwords).
    3.  **Ekstraksi Fitur:** Teks yang bersih diubah menjadi representasi numerik menggunakan metode **TF-IDF**.
    4.  **Pelatihan Model:** Dua model machine learning (Naive Bayes & SVM Linear) dilatih untuk mempelajari pola dari data.
    5.  **Evaluasi:** Model dievaluasi untuk menemukan yang memiliki performa terbaik dalam memprediksi status keramahan CV terhadap ATS.
    
    Dashboard ini menyajikan hasil dari setiap tahapan dan menyediakan alat interaktif untuk menguji CV Anda sendiri.
    """)

elif page == "Ringkasan Data 📊":
    st.header("📊 Ringkasan dan Analisis Data Eksploratif")
    
    tab_dist, tab_wc = st.tabs(["Distribusi Data", "Word Clouds"])

    with tab_dist:
        label_counts = df['label_binary'].value_counts()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total CV Dianalisis", len(df), "CV")
        col2.metric("Jumlah ATS Friendly", label_counts.get('ATS_friendly', 0), "CV")
        col3.metric("Jumlah Not ATS Friendly", label_counts.get('Not_ATS_friendly', 0), "CV")
        st.markdown("---")

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Distribusi Label")
            fig, ax = plt.subplots()
            sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax, palette=["#3498db", "#e74c3c"])
            ax.set_ylabel("Jumlah")
            st.pyplot(fig)
        with colB:
            st.subheader("Komposisi Persentase")
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=["#3498db", "#e74c3c"])
            ax_pie.axis('equal')
            st.pyplot(fig_pie)
    
    with tab_wc:
        st.subheader("Visualisasi Kata Paling Umum (Word Cloud)")
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("###### Word Cloud `ATS Friendly`")
            wc_friendly = generate_wordcloud(df[df['label_binary'] == 'ATS_friendly']['clean_text'])
            st.image(wc_friendly.to_array())
        with col6:
            st.markdown("###### Word Cloud `Not ATS Friendly`")
            wc_not_friendly = generate_wordcloud(df[df['label_binary'] == 'Not_ATS_friendly']['clean_text'])
            st.image(wc_not_friendly.to_array())

elif page == "Performa Model 📈":
    st.header("📈 Analisis Performa Model Klasifikasi")

    tab_perf, tab_imp, tab_cluster = st.tabs(["Perbandingan Kinerja", "Fitur Berpengaruh", "Visualisasi Cluster"])

    with tab_perf:
        st.subheader("Perbandingan Akurasi Antar Model")
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        accuracies = {name: model.score(X_test_tfidf, y_test) for name, model in models.items()}
        acc_perf_df = pd.DataFrame(list(accuracies.items()), columns=['Model', 'Akurasi']).sort_values('Akurasi', ascending=False)
        st.dataframe(acc_perf_df, use_container_width=True)
        
        with st.expander(f"Lihat Confusion Matrix untuk Model Terbaik ({best_model_name})"):
            y_pred = best_model.predict(X_test_tfidf)
            cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_, ax=ax)
            ax.set_xlabel('Label Prediksi'); ax.set_ylabel('Label Sebenarnya')
            st.pyplot(fig)

    with tab_imp:
        st.subheader(f"Fitur/Kata Paling Berpengaruh (Model {best_model_name})")
        
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Perbaikan di sini: Menangani model yang berbeda
        if best_model_name == 'SVM Linear':
            # Koefisien untuk SVM
            coefficients = best_model.coef_.toarray().flatten()
        elif best_model_name == 'Multinomial Naive Bayes':
            # Fitur terpenting untuk Naive Bayes (log-prob)
            coefficients = best_model.feature_log_prob_[1, :] - best_model.feature_log_prob_[0, :]
        else:
            st.warning("Model terbaik tidak mendukung analisis koefisien.")
            coefficients = np.zeros(len(feature_names))

        coef_df = pd.DataFrame({'Fitur': feature_names, 'Bobot': coefficients})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### ↗️ Indikator Kuat `Not ATS Friendly`")
            top_pos = coef_df.sort_values(by='Bobot', ascending=False).head(15)
            st.dataframe(top_pos, use_container_width=True)
            st.pyplot(plot_feature_importance(top_pos, "Kata-kata `Not ATS Friendly`", 'Reds_r'))
        
        with col2:
            st.markdown("##### ↘️ Indikator Kuat `ATS Friendly`")
            top_neg = coef_df.sort_values(by='Bobot', ascending=True).head(15)
            st.dataframe(top_neg, use_container_width=True)
            st.pyplot(plot_feature_importance(top_neg, "Kata-kata `ATS Friendly`", 'Greens_r'))

    with tab_cluster:
        st.subheader("Visualisasi Cluster Dokumen (t-SNE)")
        with st.spinner("Membuat plot t-SNE... (mungkin perlu waktu beberapa saat)"):
            tsne_fig = generate_tsne_plot(tfidf_vectorizer, df)
            st.pyplot(tsne_fig)

elif page == "Cek CV-ATS 🔍":
    st.header("🔍 Cek CV-ATS Anda Secara Langsung")
    st.markdown("Unggah CV dalam format PDF untuk memprediksi tingkat keramahannya terhadap sistem ATS.")

    uploaded_file = st.file_uploader("Pilih file PDF", type="pdf")
    temp_file_path = "temp_cv.pdf"

    if uploaded_file:
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Menganalisis CV yang diunggah..."):
                pdf_text = ""
                try:
                    with fitz.open(temp_file_path) as doc:
                        for page in doc:
                            pdf_text += page.get_text()
                except Exception as e:
                    st.error(f"Gagal membaca PDF: {e}")
                    pdf_text = ""

                if pdf_text:
                    cleaned_text = preprocess(pdf_text)
                    pdf_tfidf = tfidf_vectorizer.transform([cleaned_text])
                    st.success("CV berhasil dianalisis!")
                    
                    prediction_result = best_model.predict(pdf_tfidf)[0]

                    if prediction_result == 'ATS_friendly':
                        st.success(f"✅ **Prediksi Utama ({best_model_name}): CV Anda kemungkinan besar ATS Friendly.**")
                    else:
                        st.error(f"❌ **Prediksi Utama ({best_model_name}): CV Anda kemungkinan besar Not ATS Friendly.**")
                    
                    st.markdown("---")
                    st.subheader("Detail Prediksi dari Setiap Model")
                    
                    col1, col2 = st.columns(2)
                    
                    for i, (name, model) in enumerate(models.items()):
                        # Menggunakan kolom yang berbeda untuk setiap model
                        target_col = col1 if i % 2 == 0 else col2
                        with target_col:
                            prediction = model.predict(pdf_tfidf)[0]
                            probability = model.predict_proba(pdf_tfidf)[0]
                            classes = model.classes_
                            prob_ats_friendly_idx = list(classes).index('ATS_friendly')
                            prob_ats_friendly = probability[prob_ats_friendly_idx]
                            
                            st.markdown(f"##### {name}")
                            color = "green" if prediction == "ATS_friendly" else "red"
                            st.markdown(f"**Prediksi:** <span style='color:{color}; font-weight:bold;'>{prediction}</span>", unsafe_allow_html=True)
                            st.progress(prob_ats_friendly)
                            st.markdown(f"**Keyakinan (ATS Friendly):** `{prob_ats_friendly:.2%}`")
                else:
                    st.warning("Tidak ada teks yang dapat diekstrak dari PDF.")

        finally:
            # Hapus file sementara setelah selesai
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)