# Özge Çinko

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import missingno as msno

from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("KaggleCovidDataset.csv")

# Veri Ön-İşleme (Data Preprocssesing)
""" 
df.head()
df.shape #Satır ve sütun sayısını inceleriz.
df.columns
df.values
df.info() #Değişken değerlerini inceleriz.
msno.bar(df)  #Eksik değer varsa gösteren grafik.
plt.show()
df.isnull().sum() # Eksik değer var mı diye kontrol ederiz.

"""


# Apriori Algoritmasının Uygulanması (Implementing Apriori Algorithm)
"""
Apriori algoritması True/False ya da 1/0 değerleriyle çalışmaktadır. (One Hot Encoding)
Veri setimizdeki değerler Yes ve No'dur.
Yes = 1, No = 0 olarak çeviririz.
"""

df = df.applymap(lambda x : 1 if x == "Yes" else 0)
df = apriori(df, min_support = 0.2, use_colnames = True, verbose = 1)

# Birliktelik Kurallarının Elde Edilmesi (Association Rules)
df_ar = association_rules(df, metric = "confidence", min_threshold = 0.6)

# antecedents ve consequents değerleri frozenset olduğu için bu değerleri daha iyi anlayabilmek adına string'e çevirmek gerekir.
df_ar["antecedents"] = df_ar["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
df_ar["consequents"] = df_ar["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
df_ar.head()
df_ar.sort_values('confidence', ascending=False)

# Kesin olan sonuçlar için confidence = 1 olan değerlere bakmamız gerekir.
df_confidence = df_ar.loc[df_ar['confidence'] == 1]
df_confidence

# Verileri excel dosyasına aktararak daha detaylı inceleyebiliriz.
writer = pd.ExcelWriter('CovidSymptoms.xlsx')
df_confidence.to_excel(writer)
writer.save()
print('DataFrame is written successfully to Excel File.')


# Support, lift, confidence değerlerini görselleştirerek inceleyebiliriz.
# Support vs Confidence
plt.scatter(df_ar['support'], df_ar['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

# Support vs Lift
plt.scatter(df_ar['support'], df_ar['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()

# Lift vs Confidence
fit = np.polyfit(df_ar['lift'], df_ar['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(df_ar['lift'], df_ar['confidence'], 'yo', df_ar['lift'], 
fit_fn(df_ar['lift']))



