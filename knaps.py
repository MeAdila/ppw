import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib



st.title("UTS PENCARIAN DAN PENAMBANGAN WEB")
st.write("-------------------------------------------------------------------------------------------------------------------------")
st.write("**Nama  : Fadilatul Jannah**")
st.write("**NIM   : 200411100148**")
st.write("**Kelas : PPW A**")
st.write("-------------------------------------------------------------------------------------------------------------------------")
upload_data, preporcessing, modeling = st.tabs(["Upload Data", "Preprocessing", "Modeling"])


with upload_data:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

with modeling:
    st.write("""# Modeling""")
    
    with st.form("modeling"):
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        # NB
        
        # Memisahkan fitur dan label kelas target
        X = df[['Topik 1', 'Topik 2', 'Topik 3']]
        y = df['Cluster']  # Gantilah 'Kelas_Target' dengan nama kolom yang sesuai untuk label kelas target
        
        # Memisahkan data menjadi data latih dan data uji
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        GaussianNB(priors=None)
        # Membuat model Naive Bayes
        naive_bayes = GaussianNB()
        
        # Melatih model menggunakan data latih
        naive_bayes.fit(X_train, y_train)
        
        # Membuat prediksi menggunakan data uji
        predictions = naive_bayes.predict(X_test)
        
        # Mengukur akurasi model
        gaussian_akurasi = accuracy_score(y_test, predictions)
        print("Akurasi Naive Bayes:", gaussian_akurasi)
        # akurasi = 10

        #Gaussian Naive Bayes
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # probas = gaussian.predict_proba(test)
        # probas = probas[:,1]
        # probas = probas.round()

        # gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        #KNN
        neigh = KNeighborsClassifier(n_neighbors=3)
        knn = neigh.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)

        knn_akurasi = accuracy_score(y_test, y_pred_knn)
        print("Akurasi:", knn_akurasi)

        #Decission Tree
        
        clf = tree.DecisionTreeClassifier()
        decision_tree = clf.fit(X_train, y_train)

        y_pred_clf = decision_tree.predict(X_test)

        dt_akurasi = accuracy_score(y_test, y_pred_clf)
        print("Akurasi:", dt_akurasi)

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
