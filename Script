# Kaggle API anahtar dosyasını yükleme
from google.colab import files
files.upload()  

# Kaggle API anahtar dosyasını doğru yere taşıma
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Kaggle Titanic veri setini indirme
!kaggle datasets download -d heptapod/titanic

!unzip titanic.zip

# Gerekli kütüphaneleri yükleme
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Veri setini okuma
df = pd.read_csv('/content/train_and_test2.csv') 

print(df.info())
print(df.describe())
print(df.head())

# Özniteliklerin tanımları
# - PassengerId: Yolcu ID'si
# - Survived: Hayatta kalma (0 = Hayır, 1 = Evet)
# - Pclass: Yolcu sınıfı (1 = 1. sınıf, 2 = 2. sınıf, 3 = 3. sınıf)
# - Name: Yolcunun adı
# - Sex: Cinsiyet
# - Age: Yaş
# - SibSp: Gemideki kardeş/eş sayısı
# - Parch: Gemideki ebeveyn/çocuk sayısı
# - Ticket: Bilet numarası
# - Fare: Bilet ücreti
# - Cabin: Kabin numarası
# - Embarked: Biniş limanı (C = Cherbourg, Q = Queenstown, S = Southampton)

# Kategorik özniteliklerin belirlenmesi
categorical_features = ['Sex', 'Embarked'] 

# Kategorik özniteliklerin sayısal hale dönüştürülmesi
le = LabelEncoder()
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature].astype(str))

# Özniteliklerin gözden geçirilmesi
print(df.head())

# Veri setindeki sütun isimlerini kontrol etme
print(df.columns)

# Hedef değişkenin ve bağımsız değişkenlerin ayrılması
X = df.drop(columns=['2urvived'])  # Bağımsız değişkenler
y = df['2urvived']  # Hedef değişken

# Verilerin eğitim ve test olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizasyon
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Normalize edilmiş verilerin gözden geçirilmesi
print(X_train[:5])

# KNN için komşuluk değerleri
knn_neighbors = [3, 7, 11]

for k in knn_neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f'KNN (k={k}) - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

# MLP için gizli katman konfigürasyonları
mlp_layers = [(32,), (32, 32), (32, 32, 32)]

for layers in mlp_layers:
    mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f'MLP (layers={layers}) - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

# Naive Bayes için varsayılan parametreler 
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Naive Bayes - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
