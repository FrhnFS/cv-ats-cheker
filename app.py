import streamlit as st
import pandas as pd
import re
import ast
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
import fitz  # PyMuPDF
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords
import nltk
import numpy as np
from wordcloud import WordCloud

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="CV-ATS Analysis Dashboard",
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
    stop_words_en = set(stopwords.words('english'))
    factory = StopWordRemoverFactory()
    stop_words_id = set(factory.get_stop_words())
    return stop_words_en.union(stop_words_id)

combined_stopwords = get_stopwords()

def preprocess(text):
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
    df = pd.read_csv('CV-ATS.csv')
    def extract_label(text):
        try:
            json_data = ast.literal_eval(text)
            return json_data[0]['labels'][0] if json_data and 'labels' in json_data[0] and json_data[0]['labels'] else "Unknown"
        except:
            return "Unknown"
    df['label_extracted'] = df['label'].apply(extract_label)
    df['clean_text'] = df['text'].apply(preprocess)
    df.dropna(subset=['clean_text', 'label_extracted'], inplace=True)
    df['word_count'] = df['clean_text'].str.split().str.len()
    return df

df = load_and_preprocess_data()
df_binary = df[df['label_extracted'].isin(['ATS_friendly', 'Not_ATS_friendly'])].copy()

# --- Pelatihan Model (Cached) ---
@st.cache_resource
def train_models():
    X_binary = df_binary['clean_text']
    y_binary = df_binary['label_extracted']
    X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
        X_binary, y_binary, test_size=0.2, stratify=y_binary, random_state=42
    )
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_binary)
    models = {
        'Multinomial Naive Bayes': MultinomialNB().fit(X_train_tfidf, y_train_binary),
        'SVM Linear': SVC(kernel='linear', probability=True, random_state=42).fit(X_train_tfidf, y_train_binary),
        'SVM RBF': SVC(kernel='rbf', probability=True, random_state=42).fit(X_train_tfidf, y_train_binary),
        'SVM Polynomial': SVC(kernel='poly', probability=True, random_state=42).fit(X_train_tfidf, y_train_binary)
    }
    return models, tfidf_vectorizer, X_test_binary, y_test_binary

models, tfidf_vectorizer, X_test_binary, y_test_binary = train_models()
best_model = models['SVM Linear']

# --- Fungsi Bantuan untuk Visualisasi ---
@st.cache_data
def get_top_n_grams(corpus, n_gram_range=(1, 1), n=10):
    vec = CountVectorizer(ngram_range=n_gram_range, stop_words=list(combined_stopwords)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

@st.cache_data
def generate_wordcloud(text_series):
    text = " ".join(review for review in text_series)
    wordcloud = WordCloud(stopwords=list(combined_stopwords), background_color="white", colormap='viridis', width=800, height=400).generate(text)
    return wordcloud

@st.cache_data
def generate_tsne_plot(_tfidf_vectorizer, _df_binary):
    tfidf_matrix = _tfidf_vectorizer.transform(_df_binary['clean_text'])
    
    # === PERBAIKAN DI SINI: ganti n_iter menjadi max_iter ===
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=300)
    # =======================================================

    tsne_results = tsne.fit_transform(tfidf_matrix.toarray())
    
    df_tsne = pd.DataFrame({
        'x': tsne_results[:,0],
        'y': tsne_results[:,1],
        'label': _df_binary['label_extracted']
    })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x="x", y="y", hue="label", palette={"ATS_friendly": "green", "Not_ATS_friendly": "red"}, data=df_tsne, legend="full", alpha=0.7, ax=ax)
    ax.set_title('Visualisasi Cluster Dokumen CV dengan t-SNE')
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
    4.  **Pelatihan Model:** Beberapa model machine learning (Naive Bayes, SVM dengan berbagai kernel) dilatih untuk mempelajari pola dari data.
    5.  **Evaluasi:** Model dievaluasi untuk menemukan yang memiliki performa terbaik dalam memprediksi status keramahan CV terhadap ATS.
    
    Dashboard ini menyajikan hasil dari setiap tahapan dan menyediakan alat interaktif untuk menguji CV Anda sendiri.
    """)

elif page == "Ringkasan Data 📊":
    st.header("📊 Ringkasan dan Analisis Data Eksploratif")
    
    tab_dist, tab_text, tab_wc = st.tabs(["Distribusi Data 📈", "Analisis Teks ✍️", "Word Clouds ☁️"])

    with tab_dist:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribusi Label CV")
            label_counts = df['label_extracted'].value_counts()
            st.bar_chart(label_counts, color="#FFC300")
        with col2:
            st.subheader("Komposisi Persentase")
            fig, ax = plt.subplots()
            ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", len(label_counts)))
            ax.axis('equal')
            st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Distribusi Jumlah Kata per Kategori")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        sns.histplot(df_binary[df_binary['label_extracted'] == 'ATS_friendly']['word_count'], bins=30, ax=ax[0], color='green', kde=True)
        ax[0].set_title('ATS Friendly')
        ax[0].set_xlabel('Jumlah Kata')
        sns.histplot(df_binary[df_binary['label_extracted'] == 'Not_ATS_friendly']['word_count'], bins=30, ax=ax[1], color='red', kde=True)
        ax[1].set_title('Not ATS Friendly')
        ax[1].set_xlabel('Jumlah Kata')
        st.pyplot(fig)

    with tab_text:
        st.subheader("Analisis N-gram Interaktif")
        st.markdown("Pilih parameter untuk melihat kata atau frasa yang paling sering muncul.")
        
        col_select1, col_select2, col_select3 = st.columns(3)
        with col_select1:
            label_choice = st.selectbox("Pilih Kategori Label:", df['label_extracted'].unique())
        with col_select2:
            ngram_choice = st.radio("Pilih Tipe N-gram:", ["Unigram (1 kata)", "Bigram (2 kata)", "Trigram (3 kata)"])
        with col_select3:
            top_n_choice = st.slider("Jumlah Hasil Teratas:", 5, 20, 10)

        ngram_range_map = {"Unigram (1 kata)": (1, 1), "Bigram (2 kata)": (2, 2), "Trigram (3 kata)": (3, 3)}
        corpus_choice = df[df['label_extracted'] == label_choice]['clean_text']
        top_ngrams = get_top_n_grams(corpus_choice, n_gram_range=ngram_range_map[ngram_choice], n=top_n_choice)
        st.dataframe(pd.DataFrame(top_ngrams, columns=['N-gram', 'Frekuensi']), use_container_width=True)

    with tab_wc:
        st.subheader("Visualisasi Kata Paling Umum (Word Cloud)")
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("###### Word Cloud `ATS Friendly`")
            wc_friendly = generate_wordcloud(df_binary[df_binary['label_extracted'] == 'ATS_friendly']['clean_text'])
            st.image(wc_friendly.to_array())
        with col6:
            st.markdown("###### Word Cloud `Not ATS Friendly`")
            wc_not_friendly = generate_wordcloud(df_binary[df_binary['label_extracted'] == 'Not_ATS_friendly']['clean_text'])
            st.image(wc_not_friendly.to_array())
    
    st.markdown("---")
    st.subheader("Unduh Data")
    st.download_button(label="📥 Unduh Data CV yang Sudah Dibersihkan (CSV)", data=df.to_csv(index=False).encode('utf-8'), file_name='cleaned_cv_data.csv', mime='text/csv')

elif page == "Performa Model 📈":
    st.header("📈 Analisis Performa Model Klasifikasi")

    tab_perf, tab_imp, tab_roc, tab_cluster = st.tabs(["📊 Perbandingan Kinerja", "💡 Fitur Berpengaruh", "🎯 Kurva ROC", "🌐 Visualisasi Cluster"])

    with tab_perf:
        st.subheader("Perbandingan Akurasi Antar Model")
        X_test_tfidf = tfidf_vectorizer.transform(X_test_binary)
        accuracies = {name: model.score(X_test_tfidf, y_test_binary) for name, model in models.items()}
        acc_perf_df = pd.DataFrame(list(accuracies.items()), columns=['Model', 'Akurasi']).sort_values('Akurasi', ascending=False)
        st.dataframe(acc_perf_df, use_container_width=True)
        
        with st.expander("Lihat Confusion Matrix untuk Model Terbaik (SVM Linear)"):
            y_pred = best_model.predict(X_test_tfidf)
            cm = confusion_matrix(y_test_binary, y_pred, labels=best_model.classes_)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_, ax=ax)
            ax.set_xlabel('Label Prediksi')
            ax.set_ylabel('Label Sebenarnya')
            ax.set_title('Confusion Matrix untuk SVM Linear')
            st.pyplot(fig)

    with tab_imp:
        st.subheader("Fitur/Kata Paling Berpengaruh (Model SVM Linear)")
        feature_names = tfidf_vectorizer.get_feature_names_out()
        coefficients = best_model.coef_.toarray().flatten()
        coef_df = pd.DataFrame({'Fitur': feature_names, 'Bobot': coefficients})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### ↗️ Indikator Kuat `Not ATS Friendly`")
            top_pos = coef_df.sort_values(by='Bobot', ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x='Bobot', y='Fitur', data=top_pos, ax=ax, palette='Reds_r')
            st.pyplot(fig)
        with col2:
            st.markdown("##### ↘️ Indikator Kuat `ATS Friendly`")
            top_neg = coef_df.sort_values(by='Bobot', ascending=True).head(15)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x='Bobot', y='Fitur', data=top_neg, ax=ax, palette='Greens_r')
            st.pyplot(fig)
        
        st.download_button(label="📥 Unduh Daftar Fitur & Bobotnya (CSV)", data=coef_df.to_csv(index=False).encode('utf-8'), file_name='feature_importance.csv', mime='text/csv')

    with tab_roc:
        st.subheader("Kurva ROC untuk Model Terbaik (SVM Linear)")
        X_test_tfidf = tfidf_vectorizer.transform(X_test_binary)
        y_prob = best_model.predict_proba(X_test_tfidf)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test_binary, y_prob, pos_label='Not_ATS_friendly')
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Kurva ROC (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        st.pyplot(fig)

    with tab_cluster:
        st.subheader("Visualisasi Cluster Dokumen (t-SNE)")
        st.markdown("Plot ini memvisualisasikan kemiripan konten antar CV. Idealnya, CV dengan label yang sama akan saling berdekatan.")
        with st.spinner("Membuat plot t-SNE... (mungkin perlu waktu beberapa saat)"):
            tsne_fig = generate_tsne_plot(tfidf_vectorizer, df_binary)
            st.pyplot(tsne_fig)

elif page == "Cek CV-ATS 🔍":
    st.header("🔍 Cek CV-ATS Anda Secara Langsung")
    st.markdown("Unggah CV dalam format PDF untuk memprediksi tingkat keramahannya terhadap sistem ATS.")

    uploaded_file = st.file_uploader("Pilih file PDF", type="pdf")

    if uploaded_file:
        with open("temp_cv.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Menganalisis CV yang diunggah..."):
            pdf_text = ""
            try:
                with fitz.open("temp_cv.pdf") as doc:
                    for page in doc:
                        pdf_text += page.get_text()
            except Exception as e:
                st.error(f"Gagal membaca PDF: {e}")

            if pdf_text:
                cleaned_text = preprocess(pdf_text)
                pdf_tfidf = tfidf_vectorizer.transform([cleaned_text])
                st.success("CV berhasil dianalisis! Berikut adalah hasil prediksi:")
                
                prediction_result = best_model.predict(pdf_tfidf)[0]

                if prediction_result == 'ATS_friendly':
                    st.success(f"✅ **Prediksi Utama: CV Anda kemungkinan besar ATS Friendly.**")
                else:
                    st.error(f"❌ **Prediksi Utama: CV Anda kemungkinan besar Not ATS Friendly.**")
                
                with st.expander("💡 Klik di sini untuk Umpan Balik dan Kata Kunci"):
                    user_words = set(cleaned_text.split())
                    feature_names = tfidf_vectorizer.get_feature_names_out()
                    coefficients = best_model.coef_.toarray().flatten()
                    coef_df = pd.DataFrame({'Fitur': feature_names, 'Bobot': coefficients})
                    
                    top_pos_features = set(coef_df.sort_values(by='Bobot', ascending=False).head(30)['Fitur'])
                    top_neg_features = set(coef_df.sort_values(by='Bobot', ascending=True).head(30)['Fitur'])
                    
                    found_pos = list(user_words.intersection(top_pos_features))
                    found_neg = list(user_words.intersection(top_neg_features))

                    st.markdown("**Kata Kunci Terdeteksi yang Memengaruhi Penilaian:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("###### Cenderung `Not ATS Friendly`")
                        if found_pos:
                            st.warning(" ".join(f"`{word}`" for word in found_pos))
                        else:
                            st.info("Tidak ada kata kunci negatif yang menonjol.")
                    with col2:
                        st.markdown("###### Cenderung `ATS Friendly`")
                        if found_neg:
                            st.success(" ".join(f"`{word}`" for word in found_neg))
                        else:
                            st.info("Tidak ada kata kunci positif yang menonjol.")

                st.markdown("---")
                st.subheader("Detail Prediksi dari Setiap Model")
                
                col1, col2 = st.columns(2)
                col_list = [col1, col2, col1, col2]
                for i, (name, model) in enumerate(models.items()):
                    with col_list[i]:
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