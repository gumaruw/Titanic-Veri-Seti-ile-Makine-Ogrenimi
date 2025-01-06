# Titanic Veri Seti ile Makine Öğrenimi

Bu proje, Titanic veri kümesini kullanarak yolcuların hayatta kalma durumlarını tahmin etmeyi amaçlamaktadır. Projede K-Nearest Neighbors (KNN), Multi Layer Perceptron (MLP) ve Naive Bayes (NB) sınıflandırma algoritmaları kullanılmış ve performansları karşılaştırılmıştır.

## İçindekiler
- [Proje Hakkında](#proje-hakkında)
- [Veri Kümesi](#veri-kümesi)
- [Özniteliklerin Tanıtımı](#özniteliklerin-tanıtımı)
- [Veri Ön İşleme](#veri-ön-işleme)
- [Modelleme ve Değerlendirme](#modelleme-ve-değerlendirme)
- [Sonuçlar](#sonuçlar)
- [Nasıl Çalıştırılır](#nasıl-çalıştırılır)
- [Lisans](#lisans)

## Proje Hakkında
Bu proje, Titanic yolcu veri kümesini kullanarak yolcuların hayatta kalma durumlarını tahmin etmeyi amaçlamaktadır. Projede üç farklı sınıflandırma algoritması kullanılmış ve bu algoritmaların performansları accuracy, precision, recall ve F1-score metrikleri ile değerlendirilmiştir.

## Veri Kümesi
Proje için Kaggle'dan Titanic veri kümesi kullanılmıştır. Veri kümesini [buradan](https://www.kaggle.com/datasets/heptapod/titanic) indirebilirsiniz.

## Özniteliklerin Tanıtımı
Veri kümesi aşağıdaki öznitelikleri içermektedir:

- **PassengerId**: Yolcu ID'si
- **Survived**: Hayatta kalma (0 = Hayır, 1 = Evet)
- **Pclass**: Yolcu sınıfı (1 = 1. sınıf, 2 = 2. sınıf, 3 = 3. sınıf)
- **Name**: Yolcunun adı
- **Sex**: Cinsiyet
- **Age**: Yaş
- **SibSp**: Gemideki kardeş/eş sayısı
- **Parch**: Gemideki ebeveyn/çocuk sayısı
- **Ticket**: Bilet numarası
- **Fare**: Bilet ücreti
- **Cabin**: Kabin numarası
- **Embarked**: Biniş limanı (C = Cherbourg, Q = Queenstown, S = Southampton)

## Veri Ön İşleme
- Kategorik öznitelikler sayısal değerlere dönüştürülmüştür.
- Veriler `StandardScaler` kullanılarak normalize edilmiştir.

## Modelleme ve Değerlendirme
Projede kullanılan algoritmalar ve konfigurasyonları:

- **KNN (K-Nearest Neighbors)**: K=3, 7 ve 11 komşuluk değerleri
- **MLP (Multi Layer Perceptron)**: 1 gizli katman (32 nöron), 2 gizli katman (32’şer nöron) ve 3 gizli katman (32’şer nöron)
- **Naive Bayes**: Varsayılan parametreler

Her algoritmanın performansı accuracy, precision, recall ve F1-score metrikleri kullanılarak değerlendirilmiştir.

## Sonuçlar
- En yüksek accuracy değerine sahip algoritma: **KNN (k=11)** - 0.8817
- En yüksek precision değerine sahip algoritma: **KNN (k=11)** - 0.8750
- En yüksek recall değerine sahip algoritma: **MLP (32, 32)** - 0.7123
- En yüksek F1 Score değerine sahip algoritma: **KNN (k=11)** - 0.7597

Genel olarak en iyi performansı gösteren algoritma **KNN (k=11)** olmuştur.

## Nasıl Çalıştırılır
1. Proje dosyalarını indirin ve gerekli kütüphaneleri yükleyin.
2. Kaggle API anahtarınızı `kaggle.json` dosyası olarak proje dizinine ekleyin.
3. Titanic veri kümesini indirin ve proje dizinine çıkarın.
4. `titanic_classification.py` dosyasını çalıştırarak modelleri eğitin ve değerlendirin.

```python
# Kaggle API anahtar dosyasını yükleme
from google.colab import files
files.upload()  # Bu hücreyi çalıştırdıktan sonra 'kaggle.json' dosyasını yükleyin

# Kaggle API anahtar dosyasını doğru yere taşıma
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Kaggle Titanic veri setini indirme
!kaggle datasets download -d heptapod/titanic

# İndirilen zip dosyasını açma
!unzip titanic.zip

# Gerekli kütüphaneleri yükleme ve model kodları
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Veri setini okuma ve ön işleme adımları
df = pd.read_csv('/content/train_and_test2.csv')
