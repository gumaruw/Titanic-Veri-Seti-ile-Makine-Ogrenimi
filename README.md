# TitanicDatasetML
Bu proje, Titanic yolcularının hayatta kalma oranlarını tahmin etmek amacıyla gerçekleştirilmiştir.

KNN (K-Nearest Neighbors):
K=3, K=7 ve K=11 komşuluk değerleri için uygulanarak başarı metrikleri hesaplandı.
MLP (Multi Layer Perceptron):
1, 2 ve 3 gizli katman konfigürasyonları ile uygulanarak başarı metrikleri hesaplandı.
Naive Bayes:
Varsayılan parametrelerle uygulanarak başarı metrikleri hesaplandı.

Titanic Yolcu Hayatta Kalma Tahmini Projesi
Bu proje, Titanic yolcularının hayatta kalma oranlarını tahmin etmek amacıyla çeşitli makine öğrenimi algoritmalarını kullanarak gerçekleştirildi. Projenin adımları ve kullanılan parametreler aşağıdaki gibidir:

Veri Kümesinin Seçilmesi ve Yüklenmesi:

Titanic veri seti, Kaggle platformundan indirildi ve train_and_test2.csv dosyası kullanılarak analiz edildi.
Veri Kümesinin İncelenmesi ve Özniteliklerin Tanıtılması:

Veri setindeki öznitelikler incelendi ve özniteliklerin veri tipleri, özet istatistikleri ve ilk birkaç satırı gözden geçirildi.
Kategorik Değerlerin Sayısal Hale Dönüştürülmesi:

Kategorik öznitelikler (Sex ve Embarked) Label Encoder kullanılarak sayısal değerlere dönüştürüldü.
Veri Setinin Hazırlanması ve Normalize Edilmesi:

Hedef değişken (2urvived) ve bağımsız değişkenler ayrıldı.
Veriler eğitim ve test setlerine bölündü.
StandardScaler kullanılarak öznitelikler normalize edildi.
Makine Öğrenimi Algoritmalarının Uygulanması:

K-Nearest Neighbors (KNN):
K=3, K=7 ve K=11 komşuluk değerleri için KNN algoritması uygulandı.
Her bir model için başarı metrikleri (accuracy, precision, recall, F1-score) hesaplandı.
Multi Layer Perceptron (MLP):
1 gizli katman (32 nöron), 2 gizli katman (32’şer nöron) ve 3 gizli katman (32’şer nöron) konfigürasyonları ile MLP algoritması uygulandı.
Her bir model için başarı metrikleri hesaplandı.
Naive Bayes (NB):
Varsayılan parametrelerle Naive Bayes algoritması uygulandı.
Başarı metrikleri hesaplandı.
Algoritmaların Karşılaştırılması:

Tüm algoritmaların başarı metrikleri (accuracy, precision, recall, F1-score) karşılaştırıldı.
En iyi performansı gösteren algoritma belirlendi ve sonuçlar detaylandırıldı.


a. K-Nearest Neighbors (KNN)
KNN algoritması, farklı komşuluk değerleri (k=3, k=7, k=11) kullanılarak uygulanmıştır.

KNN (k=3):

Accuracy: 0.8397
Precision: 0.7246
Recall: 0.6849
F1 Score: 0.7042
KNN (k=7):

Accuracy: 0.8702
Precision: 0.8305
Recall: 0.6712
F1 Score: 0.7424
KNN (k=11):

Accuracy: 0.8817
Precision: 0.8750
Recall: 0.6712
F1 Score: 0.7597

b. Multi Layer Perceptron (MLP)
MLP algoritması, farklı katman konfigürasyonları (32, 32-32, 32-32-32) kullanılarak uygulanmıştır.

MLP (32):

Accuracy: 0.8588
Precision: 0.8000
Recall: 0.6575
F1 Score: 0.7218
MLP (32, 32):

Accuracy: 0.8588
Precision: 0.7647
Recall: 0.7123
F1 Score: 0.7376
MLP (32, 32, 32):

Accuracy: 0.8397
Precision: 0.7246
Recall: 0.6849
F1 Score: 0.7042
Not: MLP modelleri, maksimum iterasyon (500) ulaşmasına rağmen yakınsama sağlamamıştır.

c. Naive Bayes
Naive Bayes algoritması varsayılan parametrelerle uygulanmıştır.

Naive Bayes:
Accuracy: 0.7939
Precision: 0.6462
Recall: 0.5753
F1 Score: 0.6087

Analiz ve Yorum
En Yüksek Accuracy Değerine Sahip Algoritma: KNN (k=11) - 0.8817
En Yüksek Precision Değerine Sahip Algoritma: KNN (k=11) - 0.8750
En Yüksek Recall Değerine Sahip Algoritma: MLP (32, 32) - 0.7123
En Yüksek F1 Score Değerine Sahip Algoritma: KNN (k=11) - 0.7597
Bu sonuçlara göre, genel olarak en iyi performansı gösteren algoritma KNN (k=11) olmuştur. MLP ve Naive Bayes algoritmaları da iyi performans göstermiştir, ancak KNN (k=11) en yüksek değerlere ulaşmıştır.


