import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="loremipsum - Sentimen Analisis", page_icon="✨")
st.html(
    '''
    <h1 style='text-align: center;'>loremipsum</h1>
    <p style='text-align: center'>
        loremipsum.com adalah alat yang dirancang untuk menganalisis dan memahami emosi di balik teks. 
        Dengan dukungan teknologi linguistik komputasional dan text mining, alat ini secara otomatis mengidentifikasi 
        sentimen positif, negatif, atau netral dari teks yang Anda masukkan.
    </p>
    '''
)

# Memuat model dan vectorizer
loaded_model = joblib.load('naive_bayes_model.joblib')
loaded_vectorizer = joblib.load('tf_vectorizer.joblib')

# Fungsi analisis sentimen
def analyze_sentiment(text):
    new_texts_tf = loaded_vectorizer.transform([text])
    prediction = loaded_model.predict(new_texts_tf)
    if prediction > 0:
        return 'Positif', '✅', st.success
    elif prediction < 0:
        return 'Negatif', '🚨', st.error
    else:
        return 'Netral', 'ℹ️', st.info

tab1, tab2 = st.tabs(['**Analisis Teks Langsung**', '**Analisis dari File**'])

# Tab 1: Analisis Teks Langsung
with tab1:
    text = st.text_area(
        '',
        placeholder=(
            'Masukkan ulasan, feedback, atau teks apapun di sini. Gunakan 2 enter untuk memisahkan setiap teks, seperti contoh berikut:\n\n'
            'Produk ini sangat bagus, saya suka! 😊\n\n'
            'Layanan sangat lambat, saya kecewa. 😞\n\n'
            'Biasa saja, tidak ada yang istimewa. 😑'
        ),
        height=210,
        max_chars=1000,
        label_visibility='collapsed'
    )

    if st.button('Jalankan Analisis') and text:
        lines = [line.strip() for line in text.split('\n\n') if line.strip()]
        
        if len(lines) == 1:
            # Analisis satu teks
            sentiment, icon, display = analyze_sentiment(lines[0])
            display(f'**Teks ini bersentimen ({sentiment}):**\n\n{lines[0]}', icon=icon)
        else:
            # Analisis banyak teks
            for idx, line in enumerate(lines):
                sentiment, icon, display = analyze_sentiment(line)
                display(f'**Teks {idx + 1} bersentimen ({sentiment}):**\n\n{line}', icon=icon)

# Tab 2: Analisis dari File
with tab2:
    upl = st.file_uploader('', type='csv', accept_multiple_files=False, label_visibility='collapsed')

    if upl:
        df = pd.read_csv(upl)
        selected_column = st.selectbox('Pilih kolom yang berisi teks untuk analisis:', df.columns)

        if selected_column:
            # Analisis sentimen pada kolom yang dipilih
            df['hasil_analisis'] = df[selected_column].apply(lambda x: analyze_sentiment(x.strip())[0])
            
            # Hitung jumlah total dan distribusi sentimen
            total_teks = len(df)
            sentiment_counts = df['hasil_analisis'].value_counts()
            positif = sentiment_counts.get('Positif', 0)
            negatif = sentiment_counts.get('Negatif', 0)
            netral = sentiment_counts.get('Netral', 0)
            
            # Pilih jumlah baris untuk ditampilkan
            rows = st.selectbox('Pilih jumlah baris yang ingin ditampilkan:', [5, 10, 50, 100, len(df)], 
                                format_func=lambda x: f'Semua ({x})' if x == len(df) else f'{x} baris')

            # Highlight kolom yang dipilih
            def highlight_column(s):
                return ['background-color: yellow' if col == selected_column else '' for col in s.index]
            
            # Tampilkan ringkasan hasil analisis
            st.success(
                f"File berhasil dianalisis. Dari total **{total_teks}** teks, terdapat **{positif}** positif, "
                f"**{negatif}** negatif, dan **{netral}** netral."
            )

            st.write(df.head(rows).style.apply(highlight_column, axis=1))

            # Pie Chart
            fig = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.3,
                textinfo='label+percent+value',
                insidetextorientation='radial'
            )])
            st.plotly_chart(fig)

            # Unduh hasil analisis
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='Unduh Hasil Analisis',
                data=csv,
                file_name='hasil_sentimen_analisis.csv',
                mime='text/csv'
            )