import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from tabulate import tabulate

# 1	Veriyi ve Projeyi Anlama ve Yorumlama

# Veri Setinin Hikayesi

# Veriseti 7.11.2019 - 7.11.2024 tarihleri arasındaki 5 yıllık TEFAS fonlarının verilerini içermektedir.

# Türkiye Elektronik Fon Alım Satım Platformu (TEFAS) Türkiye'deki yatırım fonlarının yatırımcılar
# tarafından tek bir platform üzerinden alınıp satılabilmesini sağlayan bir sistemdir.
# Fon, birçok yatırımcının bir araya gelerek topladığı paraların, profesyonel portföy yöneticileri
# tarafından çeşitli yatırım araçlarına yatırılmasıyla oluşan bir yatırım aracıdır.
# Fonlar, bireysel yatırımcıların tek başlarına ulaşmakta zorlanacakları yatırım araçlarına
# erişim sağlar ve yatırım riskini dağıtarak yatırımcılara avantaj sağlar.

# FON KODU --> Herbir yatırım fonuna ait kısaltılmış benzersiz kodlardır.Bu kod, her fonun kolayca tanımlanması ve ayrıştırılması için kullanılır.
# FON ADI --> Yatırım fonunun adını ifade eder ve genellikle fonun yatırım stratejisi, türü veya içerdiği varlıklar hakkında bilgi verir.
# KATEGORİ --> Yatırım fonlarının hangi tür yatırım stratejisine veya varlık sınıfına odaklandığını gösteren bir sınıflandırmadır.
# ALT KATEGORİ --> Alt kategoriler, bir fonun hangi tür varlıklara daha ağırlıklı yatırım yaptığını veya hangi stratejiyi izlediğini daha ayrıntılı olarak belirtir.
# FİYAT --> Yatırım fonu fiyatı, bir yatırımcının belirli bir günde bir fon payı için ödemesi gereken değeri gösterir.Fon fiyatı, fon portföyündeki varlıkların toplam değerinin,
# fonun çıkardığı toplam pay sayısına bölünmesiyle hesaplanır.
# RİSK DEĞERİ --> Risk Değeri, bir yatırım fonunun taşıdığı risk seviyesini ölçen bir metriktir. Risk değeri arttıkça getiri olasılığı bu yönde artar.
# GÜNLÜK,HAFTALIK,1AY,3AY,YBB(Yıl Başından Beri),6AY,1YIL,3YIL,5YIL sütunlarındaki %'lik değişimler ilgili fonun 7.11.2024 ile belirtilen
# zaman aralığı arasında fiyattaki yüzdelik değişimi ifade etmektedir.
# PORTFÖY BÜYÜKLÜĞÜ --> Bir yatırım fonunun toplam varlık değerini ifade eder. Yani, fonun sahip olduğu tüm yatırım araçlarının (hisse senetleri, tahviller, nakit varlıklar, emtialar vb.)
# toplam piyasa değeri, portföy büyüklüğünü oluşturur.
# YATIRIMCI SAYISI --> Bir yatırım fonuna yatırım yapan kişi veya kurumların toplam sayısını ifade eder. Bu sayı, fonun popülaritesini, likiditesini ve yatırımcı güvenini gösteren önemli bir metriktir.
# Yatırımcı sayısı yüksek olan fonlar, genellikle daha güvenilir ve daha fazla ilgi gören fonlar olarak değerlendirilir.
# TEDAVÜLDEKİ PAY ADETİ --> Bir yatırım fonunun piyasada dolaşımda olan toplam pay sayısını ifade eder.

#Keşifçi Veri Analizi

df = pd.read_excel(r"D:\Miuul\Data Scientist Bootcamp-16\TEFAS_Fund_Comparison_Machine_Learning_Project\datasets\Fon_Data.xlsx")
#df_copy = df.copy()
print(tabulate(df.head(50), headers='keys', tablefmt='psql'))

df.shape
df.head(10)
df.info
df.dtypes
df.columns
df.describe().T
df.isnull().sum()

def check_df(dataframe, head=20):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    # print("##################### Info #####################")
    # print(dataframe.info)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.select_dtypes(include=['float', 'int']).quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

#2	Veri Ön İşleme

# Eksik veri olan kolonların tespit edilmesi
missing_values = df.isnull().mean() * 100
print(missing_values[missing_values > 0])  # Eksik veri yüzdesi 0'dan büyük olanları göster

# Belirtilen sütunlar veri seti ve fonlar açısında finansal açıdan incelenmeyecektir.
columns_to_drop = [
    "STANDART SAPMA", "DEĞİŞKENLİK KATSAYISI", "SHARPE", "AŞAĞI YÖNLÜ RİSK", "NEGATİF GETİRİLİ GÜN SAYISI",
    "NEGATİF GETİRİLİ GÜN ORANI", "MAKSİMUM KAYIP", "YÖNETİM ÜCRETİ", "AZAMİ FON İŞLETİM GİDERİ",
    "ALIŞ VALÖRÜ", "SATIŞ VALÖRÜ", "MİNİMUM ALIŞ MİKTARI", "MAXİMUM ALIŞ MİKTARI",
    "MİNİMUM SATIŞ MİKTARI", "MAXİMUM SATIŞ MİKTARI"
]

df = df.drop(columns=columns_to_drop)


# Fon_Bilgileri_Columns'da yer alan Fiyat,Yatırımcı Sayısı ve Risk Değeri gibi değerler null ya da 0 olamaz !!

# FİYAT, YATIRIMCI SAYISI ve RİSK DEĞERİ sütunlarında NaN değer içeren satırları kaldırma
df = df.dropna(subset=['FİYAT', 'YATIRIMCI SAYISI', 'RİSK DEĞERİ'])


# Yatirim_Paylari_Columns'da yer alan Portföy araçları  her bir fon için hepsini içeremeyeceği için bazı satırlar Null değerdedir.
# Fakat burdaki Null değerler 0 ile doldurulacaktır.

# Yatirim_Paylari_Columns Boş olan değerleri 0 ile doldurma
df[Yatirim_Paylari_Columns] = df[Yatirim_Paylari_Columns].fillna(0)

# "Fon_Bilgileri" adında yeni bir alt veri çerçevesi oluştur

Fon_Bilgileri_Columns = [
    'FON KODU', 'FON ADI', 'KATEGORİ', 'ALT KATEGORİ', 'FİYAT',
       'RİSK DEĞERİ', 'GÜNLÜK', 'HAFTALIK', '1AY', '3AY', '6AY', '1YIL',
       '3YIL', '5YIL', 'FONUN YAŞI', 'PORTFÖY BÜYÜKLÜĞÜ', 'YATIRIMCI SAYISI',
       'TEDAVÜLDEKİ PAY ADETİ', 'DOLULUK ORANI'
]
len(Fon_Bilgileri_Columns)

Yatirim_Paylari_Columns = [
    'BANKA BONOSU',
       'BYF KATILMA PAYI', 'DEVLET TAHVİLİ', 'DİĞER',
       'DÖVİZ KAMU İÇ BORÇLANMA ARACI', 'DÖVİZ ÖDEMELİ BONO',
       'DÖVİZ ÖDEMELİ TAHVİL', 'EUROBONDS', 'FİNANSMAN BONOSU',
       'FON KATILMA BELGESİ', 'GAYRİMENKUL SERTİFİKASI',
       'GAYRİMENKUL YATIRIM FON KATILMA PAYI',
       'GİRİŞİM YATIRIM FON KATILMA PAYI', 'HAZİNE BONOSU', 'HİSSE SENEDİ',
       'KAMU DIŞ BORÇLANMA ARACI', 'KAMU KİRA SERTİFİKASI',
       'KAMU KİRA SERTİFİKASI DÖVİZ', 'KAMU KİRA SERTİFİKASI TL',
       'KAMU YURT DIŞI KİRA SERTİFİKASI', 'KATILIM HESABI',
       'KATILIM HESABI ALTIN', 'KATILIM HESABI DÖVİZ', 'KATILIM HESABI TL',
       'KIYMETLİ MADEN', 'KIYMETLİ MADEN CİNSİNDEN BYF',
       'KIYMETLİ MADEN KAMU BORÇLANMA ARACI',
       'KIYMETLİ MADEN KAMU KİRA SERTİFİKASI', 'MEVDUAT ALTIN',
       'MEVDUAT DÖVİZ', 'MEVDUAT TL', 'ÖZEL SEKTÖR YURT DIŞI KİRA SERTİFİKASI',
       'ÖZEL SEKTÖR DIŞ BORÇ ARACI', 'ÖZEL SEKTÖR KİRA SERTİFİKASI',
       'ÖZEL SEKTÖR TAHVİLİ', 'REPO', 'TAKASBANK PARA PİYASASI', 'TERS REPO',
       'TÜREV ARACI', 'VADELİ İŞLEMLER NAKİT TEMİNATI', 'VADELİ MEVDUAT',
       'VARLIĞA DAYALI MENKUL KIYMET', 'YABANCI BORÇLANMA ARACI',
       'YABANCI BORSA YATIRIM FONU', 'YABANCI HİSSE SENEDİ',
       'YABANCI KAMU BORÇLANMA ARACI', 'YABANCI MENKUL KIYMET',
       'YABANCI ÖZEL SEKTÖR BORÇLANMA ARACI', 'YATIRIM FONLARI KATILMA PAYI'
]
len(Yatirim_Paylari_Columns)


df[Fon_Bilgileri_Columns].isnull().sum()
df[Yatirim_Paylari_Columns].isnull().sum()

#Toplam kolundaki benzersiz değerlerin listesi

df['Toplam'].unique().tolist()

# 98 ile 102 arasındaki değerleri 100'e yuvarlayalım (loc kullanarak)
df.loc[(df['Toplam'] >= 98) & (df['Toplam'] <= 102), 'Toplam'] = 100

# Toplam sütununda 100 değeri dışındaki satırları silme
df = df[df['Toplam'] == 100]

# Kategori ve Alt Kategori'ye göre gruplama ve her gruba ait veri sayısını gösterme
grouped_df = df.groupby(['KATEGORİ', 'ALT KATEGORİ']).size().reset_index(name='Summer')
print(grouped_df)

grouped_df['Summer'].sum()

# Yatırım paylarında 0'dan farklı olan fon yüzdelilkli saılarını hesaplama
df['Yatirim_Paylari_Dolu_Sayisi'] = (df[Yatirim_Paylari_Columns] != 0).sum(axis=1)

# Display the first few rows to verify the new column
print(df[['Yatirim_Paylari_Dolu_Sayisi']].head())

# Sonucu kontrol edelim
print(df[['FON ADI', 'Yatirim_Paylari_Dolu_Sayisi']].head())


# 'FON ADI' ve 'Yatirim_Paylari_Dolu_Sayisi' sütunlarını seçip, 'Yatirim_Paylari_Dolu_Sayisi'na göre azalan sırada sıralayalım
fon_yatirim_dolu_sayisi_sorted_df = df[['FON ADI', 'Yatirim_Paylari_Dolu_Sayisi']].sort_values(by='Yatirim_Paylari_Dolu_Sayisi', ascending=False)

# İlk birkaç satırı görüntüleyelim
print(fon_yatirim_dolu_sayisi_sorted_df.head(10))

# GÜNLÜK, HAFTALIK, 1AY, 3AY, YBB, 6AY, 1YIL, 3YIL ve 5YIL sütunlarını içeren bir liste oluşturma
zaman_serileri = ['GÜNLÜK', 'HAFTALIK', '1AY', '3AY', '6AY', '1YIL', '3YIL', '5YIL']

# Hem 'FON ADI', 'Yatirim_Paylari_Dolu_Sayisi' hem de belirtilen diğer sütunları seçip sıralayalım
tum_sutunlar = ['FON ADI', 'Yatirim_Paylari_Dolu_Sayisi'] + [col for col in zaman_serileri if col in df.columns]
fon_yatirim_dolu_sayisi_sorted_df = df[tum_sutunlar].sort_values(by='Yatirim_Paylari_Dolu_Sayisi', ascending=False)

fon_yatirim_dolu_sayisi_sorted_df.head(10)

# Yatirim_Paylari_Dolu_Sayisi sütununda 0 değeri olan satırları sayma
zero_values_count = (fon_yatirim_dolu_sayisi_sorted_df['Yatirim_Paylari_Dolu_Sayisi'] == 0).sum()
print("0 değeri olan satır sayısı:", zero_values_count)

# 'Yatirim_Paylari_Dolu_Sayisi' sütununu 'Toplam' sütunundan önceki konuma taşıyalım
columns = df.columns.tolist()

if 'Yatirim_Paylari_Dolu_Sayisi' in columns and 'Toplam' in columns:
    columns.remove('Yatirim_Paylari_Dolu_Sayisi')
    toplam_index = columns.index('Toplam')
    columns.insert(toplam_index, 'Yatirim_Paylari_Dolu_Sayisi')
    df = df[columns]

#Veri Ön İşlemi sonrası verinin excelde incelenmesi
df.to_excel("Fon_Veri_Rev0.xlsx", index=False)

#3	Feature Engineering

# Zaman Temelli Özellikler

df.isnull().sum()


# Günlük ve haftalık getiriler arasındaki fark
df['Gunluk_Haftalik_Fark'] = df['HAFTALIK'] - df['GÜNLÜK']

# 1 ay ve 3 ay arasındaki fark
df['1Ay_3Ay_Fark'] = df['3AY'] - df['1AY']

# 1 yıl ve 5 yıl arasındaki fark
df['1Yil_5Yil_Fark'] = df['5YIL'] - df['1YIL']


#Örnek olarak yatırımcı yoğunluğu hesaplama
df.loc[:, 'YATIRIMCI YOĞUNLUĞU'] = df['YATIRIMCI SAYISI'] / df['PORTFÖY BÜYÜKLÜĞÜ']
print(df.columns)

# Fon_Yasi_Featured adı altında yeni bir kategorik değişken oluşturulması

# Fon yaş gruplarını tanımlayın
bebek_fonlar = ['GÜNLÜK', 'HAFTALIK', '1AY', '3AY']
genc_fonlar = ['GÜNLÜK', 'HAFTALIK', '1AY', '3AY', '6AY', '1YIL']
yetiskin_fonlar = ['GÜNLÜK', 'HAFTALIK', '1AY', '3AY', '6AY', '1YIL', '3YIL', '5YIL']

# Fon grubu atama fonksiyonu (belirtilen sütunların tamamında değer olanlar için)
def fon_yasi_featured(row):
    if all(pd.notnull(row[col]) for col in yetiskin_fonlar):
        return 'Yetişkin Fonlar'
    elif all(pd.notnull(row[col]) for col in genc_fonlar) and all(pd.isnull(row[col]) for col in set(yetiskin_fonlar) - set(genc_fonlar)):
        return 'Genç Fonlar'
    elif all(pd.notnull(row[col]) for col in bebek_fonlar) and all(pd.isnull(row[col]) for col in set(genc_fonlar) - set(bebek_fonlar)):
        return 'Bebek Fonlar'
    return 'Bilinmeyen'

# Yeni 'Fon_Yasi_Featured' sütununu ekleyin
df['Fon_Yasi_Featured'] = df.apply(fon_yasi_featured, axis=1)

# 'Fon_Yasi_Featured' sütununda 'Bilinmeyen' yazan satırları veri setinden silme
df = df[df['Fon_Yasi_Featured'] != 'Bilinmeyen']

# Sonucu kontrol edin
print(df[['FON ADI', 'Fon_Yasi_Featured']].head())


# Zaman serisi sütunlarını ve 'FON ADI', 'Fon_Yasi_Featured' sütunlarını seçip görüntüleyelim
zaman_serisi_sutunlar = ['GÜNLÜK', 'HAFTALIK', '1AY', '3AY', '6AY', '1YIL', '3YIL', '5YIL']
df_zaman_serisi = df[['FON ADI', 'Fon_Yasi_Featured'] + zaman_serisi_sutunlar]

# İlk birkaç satırı görüntüleyelim
print(df_zaman_serisi.head(20))

# Trend Özellikleri

# 3 aylık getirinin 1 aylık getiriden büyük olup olmadığına göre kısa vadeli trend belirleme
df['Kisa_Vadeli_Trend'] = df.apply(lambda x: 'Pozitif' if x['3AY'] > x['1AY'] else 'Negatif', axis=1)

# 5 yıllık getirinin 1 yıllık getiriden büyük olup olmadığına göre uzun vadeli trend belirleme
df['Uzun_Vadeli_Trend'] = df.apply(lambda x: 'Pozitif' if x['5YIL'] > x['1YIL'] else 'Negatif', axis=1)

# İlk birkaç satırı görüntüleyerek sonuçları kontrol edelim
print(df[['FON ADI', 'Kisa_Vadeli_Trend', 'Uzun_Vadeli_Trend']].head())

# Yeni eklenen özellik sütunlarını tanımla
featured_columns = [
    "Gunluk_Haftalik_Fark", "1Ay_3Ay_Fark", "1Yil_5Yil_Fark", "Fon_Yasi_Featured",
    "Kisa_Vadeli_Trend", "Uzun_Vadeli_Trend", "YATIRIMCI YOĞUNLUĞU"
]

# featured_column kolonlarını taşıma

#############################

# Taşınacak sütunları tanımla
featured_columns = [
    "Gunluk_Haftalik_Fark", "1Ay_3Ay_Fark", "1Yil_5Yil_Fark", "Fon_Yasi_Featured",
    "Kisa_Vadeli_Trend", "Uzun_Vadeli_Trend", "YATIRIMCI YOĞUNLUĞU"
]

# Mevcut tüm sütunları liste olarak al
all_columns = df.columns.tolist()

# "BANKA BONOSU" sütununun konumunu bul
bank_bonosu_index = all_columns.index("BANKA BONOSU")

# Önce featured_columns'u tüm sütun listesinden çıkaralım, böylece sona eklenmezler
all_columns = [col for col in all_columns if col not in featured_columns]

# Yeni sütun sırasını oluştur: 'featured_columns' sütunları BANKA BONOSU'ndan önceki konuma yerleştir
new_column_order = all_columns[:bank_bonosu_index] + featured_columns + all_columns[bank_bonosu_index:]

# Veri setini yeni sütun sırasına göre yeniden düzenle
df1 = df[new_column_order]

df1.head()

# Yeni özellik sütunlarını mevcut listeye ekle
Fon_Bilgileri_Columns = [
    'FON KODU', 'FON ADI', 'KATEGORİ', 'ALT KATEGORİ', 'FİYAT',
       'RİSK DEĞERİ', 'GÜNLÜK', 'HAFTALIK', '1AY', '3AY', '6AY', '1YIL',
       '3YIL', '5YIL', 'FONUN YAŞI', 'PORTFÖY BÜYÜKLÜĞÜ', 'YATIRIMCI SAYISI',
       'TEDAVÜLDEKİ PAY ADETİ', 'DOLULUK ORANI', 'Gunluk_Haftalik_Fark',
 '1Ay_3Ay_Fark',
 '1Yil_5Yil_Fark',
 'Fon_Yasi_Featured',
 'Kisa_Vadeli_Trend',
 'Uzun_Vadeli_Trend',
 'YATIRIMCI YOĞUNLUĞU'
]
Yatirim_Paylari_Columns

#Feature engineering sonrası verinin excelde incelenmesi
df1.to_excel("Fon_Veri_Rev1.xlsx", index=False)

# 4	Projenin Gerçek Hayatta Bir Probleme Çözüm Sunması

# Fonların Performans Tahmini
# Problem Tanımı: Yatırımcılar, bir fonun gelecekte nasıl performans göstereceği konusunda bilgi sahibi olmak isterler.
# Bu nedenle, bir fonun kısa ve uzun vadeli performansını tahmin etmek, yatırım kararlarını destekleyebilir.
# Çözüm Yaklaşımı:
# Geçmiş performans, risk değeri, yatırımcı yoğunluğu ve portföy dağılımı gibi gibi özellikler kullanılarak gelecekteki fiyat hareketlerinin veya getirilerin tahmini yapılabilir.
# Bir zaman serisi tahmin modeli veya regresyon modelini kullanarak belirli bir fonun 1 aylık veya 1 yıllık getirisini tahmin edilerek yatırım yapılabilir.


##########################################################
#########################################################
#5	Model Kurma Ve Parametre Optimizasyonu

#Lasso Regresyon
#Bebek Fonlar için Yeni

target_column_bebek = '3AY'
feature_columns_bebek = [
    'FİYAT', 'RİSK DEĞERİ', 'FONUN YAŞI', 'PORTFÖY BÜYÜKLÜĞÜ', 'YATIRIMCI SAYISI', 'TEDAVÜLDEKİ PAY ADETİ',
    'DOLULUK ORANI', 'Gunluk_Haftalik_Fark', '1Ay_3Ay_Fark', 'YATIRIMCI YOĞUNLUĞU',
    'BANKA BONOSU', 'BYF KATILMA PAYI', 'DEVLET TAHVİLİ', 'DİĞER',
    'DÖVİZ KAMU İÇ BORÇLANMA ARACI', 'DÖVİZ ÖDEMELİ BONO', 'DÖVİZ ÖDEMELİ TAHVİL', 'EUROBONDS',
    'FİNANSMAN BONOSU', 'FON KATILMA BELGESİ', 'GAYRİMENKUL SERTİFİKASI', 'GAYRİMENKUL YATIRIM FON KATILMA PAYI',
    'GİRİŞİM YATIRIM FON KATILMA PAYI', 'HAZİNE BONOSU', 'HİSSE SENEDİ', 'KAMU DIŞ BORÇLANMA ARACI',
    'KAMU KİRA SERTİFİKASI', 'KAMU KİRA SERTİFİKASI DÖVİZ', 'KAMU KİRA SERTİFİKASI TL',
    'KAMU YURT DIŞI KİRA SERTİFİKASI', 'KATILIM HESABI', 'KATILIM HESABI ALTIN', 'KATILIM HESABI DÖVİZ',
    'KATILIM HESABI TL', 'KIYMETLİ MADEN', 'KIYMETLİ MADEN CİNSİNDEN BYF', 'KIYMETLİ MADEN KAMU BORÇLANMA ARACI',
    'KIYMETLİ MADEN KAMU KİRA SERTİFİKASI', 'MEVDUAT ALTIN', 'MEVDUAT DÖVİZ', 'MEVDUAT TL',
    'ÖZEL SEKTÖR YURT DIŞI KİRA SERTİFİKASI', 'ÖZEL SEKTÖR DIŞ BORÇ ARACI', 'ÖZEL SEKTÖR KİRA SERTİFİKASI',
    'ÖZEL SEKTÖR TAHVİLİ', 'REPO', 'TAKASBANK PARA PİYASASI', 'TERS REPO', 'TÜREV ARACI',
    'VADELİ İŞLEMLER NAKİT TEMİNATI', 'VADELİ MEVDUAT', 'VARLIĞA DAYALI MENKUL KIYMET',
    'YABANCI BORÇLANMA ARACI', 'YABANCI BORSA YATIRIM FONU', 'YABANCI HİSSE SENEDİ',
    'YABANCI KAMU BORÇLANMA ARACI', 'YABANCI MENKUL KIYMET', 'YABANCI ÖZEL SEKTÖR BORÇLANMA ARACI',
    'YATIRIM FONLARI KATILMA PAYI', 'Yatirim_Paylari_Dolu_Sayisi'
]

# "Bebek Fonlar" için veri setini filtreleyip gerekli sütunları seçme
bebek_fonlar_data = df1[df1['Fon_Yasi_Featured'] == 'Bebek Fonlar'][[target_column_bebek] + feature_columns_bebek]
bebek_fonlar_data = bebek_fonlar_data.dropna()  # Eksik değerleri çıkarma

# X ve y olarak bağımlı ve bağımsız değişkenleri ayırma
X_bebek = bebek_fonlar_data[feature_columns_bebek]
y_bebek = bebek_fonlar_data[target_column_bebek]

# Eğitim ve test veri setlerine ayırma
X_train_bebek, X_test_bebek, y_train_bebek, y_test_bebek = train_test_split(X_bebek, y_bebek, test_size=0.2, random_state=42)

# Özellik ölçeklendirme
scaler = StandardScaler()
X_train_bebek = scaler.fit_transform(X_train_bebek)
X_test_bebek = scaler.transform(X_test_bebek)

# Lasso model için en iyi alpha değerini belirlemek için GridSearchCV kullanma
lasso = Lasso(random_state=42)
param_grid = {'alpha': np.logspace(-4, 0, 50)}
grid_search = GridSearchCV(lasso, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train_bebek, y_train_bebek)

# En iyi model ve en iyi alpha değeri
best_lasso = grid_search.best_estimator_
print("Best alpha:", grid_search.best_params_['alpha'])

# Test seti ile tahmin yapma
y_pred_bebek = best_lasso.predict(X_test_bebek)

# Başarı metriklerini hesaplama
mse_bebek = mean_squared_error(y_test_bebek, y_pred_bebek)
rmse_bebek = np.sqrt(mse_bebek)
mae_bebek = mean_absolute_error(y_test_bebek, y_pred_bebek)
r2_bebek = r2_score(y_test_bebek, y_pred_bebek)

# Sonuçları yazdırma
print("Lasso Regression Model Performance for Bebek Fonlar:")
print("Mean Squared Error (MSE):", mse_bebek)
print("Root Mean Squared Error (RMSE):", rmse_bebek)
print("Mean Absolute Error (MAE):", mae_bebek)
print("R-squared (R²):", r2_bebek)

# Özellik ağırlıklarını gösterme
feature_weights_bebek = pd.DataFrame({
    'Feature': feature_columns_bebek,
    'Weight': best_lasso.coef_
}).sort_values(by='Weight', ascending=False)

print("\nFeature Weights:")
print(feature_weights_bebek)

# Örnek ağırlık verilerini içeriyorsa bunu tanımlayın. (Önceden tanımlanmışsa gerek yok)
feature_weights_bebek = pd.DataFrame({
    'Feature': feature_columns_bebek,
    'Weight': best_lasso.coef_
}).sort_values(by='Weight', ascending=False)

# Ağırlıkları çubuk grafik olarak gösterme
plt.figure(figsize=(12, 8))
plt.barh(feature_weights_bebek['Feature'], feature_weights_bebek['Weight'], color='skyblue')
plt.xlabel('Weight')
plt.title('Feature Weights from Optimized Lasso Regression Model for Bebek Fonlar')
plt.gca().invert_yaxis()  # En yüksek ağırlıkların en üstte görünmesi için y eksenini ters çevirme
plt.tight_layout()
plt.show()


####################
# Genç Fonlar İçin Yeni
target_column_genc = '1YIL'
feature_columns_genc = [
    'FİYAT', 'RİSK DEĞERİ', 'FONUN YAŞI', 'PORTFÖY BÜYÜKLÜĞÜ', 'YATIRIMCI SAYISI', 'TEDAVÜLDEKİ PAY ADETİ',
    'DOLULUK ORANI', 'Gunluk_Haftalik_Fark', '1Ay_3Ay_Fark', 'YATIRIMCI YOĞUNLUĞU',
    'BANKA BONOSU', 'BYF KATILMA PAYI', 'DEVLET TAHVİLİ', 'DİĞER',
    'DÖVİZ KAMU İÇ BORÇLANMA ARACI', 'DÖVİZ ÖDEMELİ BONO', 'DÖVİZ ÖDEMELİ TAHVİL', 'EUROBONDS',
    'FİNANSMAN BONOSU', 'FON KATILMA BELGESİ', 'GAYRİMENKUL SERTİFİKASI', 'GAYRİMENKUL YATIRIM FON KATILMA PAYI',
    'GİRİŞİM YATIRIM FON KATILMA PAYI', 'HAZİNE BONOSU', 'HİSSE SENEDİ', 'KAMU DIŞ BORÇLANMA ARACI',
    'KAMU KİRA SERTİFİKASI', 'KAMU KİRA SERTİFİKASI DÖVİZ', 'KAMU KİRA SERTİFİKASI TL',
    'KAMU YURT DIŞI KİRA SERTİFİKASI', 'KATILIM HESABI', 'KATILIM HESABI ALTIN', 'KATILIM HESABI DÖVİZ',
    'KATILIM HESABI TL', 'KIYMETLİ MADEN', 'KIYMETLİ MADEN CİNSİNDEN BYF', 'KIYMETLİ MADEN KAMU BORÇLANMA ARACI',
    'KIYMETLİ MADEN KAMU KİRA SERTİFİKASI', 'MEVDUAT ALTIN', 'MEVDUAT DÖVİZ', 'MEVDUAT TL',
    'ÖZEL SEKTÖR YURT DIŞI KİRA SERTİFİKASI', 'ÖZEL SEKTÖR DIŞ BORÇ ARACI', 'ÖZEL SEKTÖR KİRA SERTİFİKASI',
    'ÖZEL SEKTÖR TAHVİLİ', 'REPO', 'TAKASBANK PARA PİYASASI', 'TERS REPO', 'TÜREV ARACI',
    'VADELİ İŞLEMLER NAKİT TEMİNATI', 'VADELİ MEVDUAT', 'VARLIĞA DAYALI MENKUL KIYMET',
    'YABANCI BORÇLANMA ARACI', 'YABANCI BORSA YATIRIM FONU', 'YABANCI HİSSE SENEDİ',
    'YABANCI KAMU BORÇLANMA ARACI', 'YABANCI MENKUL KIYMET', 'YABANCI ÖZEL SEKTÖR BORÇLANMA ARACI',
    'YATIRIM FONLARI KATILMA PAYI', 'Yatirim_Paylari_Dolu_Sayisi'
]

# Genç Fonlar için veri hazırlığı
genc_fonlar_data = df1[df1['Fon_Yasi_Featured'] == 'Genç Fonlar'][[target_column_genc] + feature_columns_genc].dropna()
X_genc = genc_fonlar_data[feature_columns_genc]
y_genc = genc_fonlar_data[target_column_genc]

# Eğitim ve test setlerine ayırma
X_train_genc, X_test_genc, y_train_genc, y_test_genc = train_test_split(X_genc, y_genc, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler_genc = StandardScaler()
X_train_genc = scaler_genc.fit_transform(X_train_genc)
X_test_genc = scaler_genc.transform(X_test_genc)

# En iyi alpha değeri için GridSearchCV ile Lasso modelini optimize etme
lasso = Lasso(random_state=42)
param_grid = {'alpha': np.logspace(-4, 0, 50)}
grid_search_genc = GridSearchCV(lasso, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search_genc.fit(X_train_genc, y_train_genc)

# En iyi model ve alpha değeri
best_lasso_genc = grid_search_genc.best_estimator_

# Test seti ile tahmin yapma
y_pred_genc = best_lasso_genc.predict(X_test_genc)

# Başarı metriklerini hesaplama
mse_genc = mean_squared_error(y_test_genc, y_pred_genc)
rmse_genc = np.sqrt(mse_genc)
mae_genc = mean_absolute_error(y_test_genc, y_pred_genc)
r2_genc = r2_score(y_test_genc, y_pred_genc)

# Sonuçları yazdırma
print("Lasso Regression Model Performance for Genç Fonlar:")
print("Mean Squared Error (MSE):", mse_genc)
print("Root Mean Squared Error (RMSE):", rmse_genc)
print("Mean Absolute Error (MAE):", mae_genc)
print("R-squared (R²):", r2_genc)

# Ağırlıkları görselleştirme
feature_weights_genc = pd.DataFrame({
    'Feature': feature_columns_genc,
    'Weight': best_lasso_genc.coef_
}).sort_values(by='Weight', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(feature_weights_genc['Feature'], feature_weights_genc['Weight'], color='skyblue')
plt.xlabel('Weight')
plt.title('Feature Weights from Optimized Lasso Regression Model for Genç Fonlar')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


#Yetiskin Fonlar için Yeni

target_column_yetiskin = '5YIL'
feature_columns_yetiskin = [
    'FİYAT', 'RİSK DEĞERİ', 'FONUN YAŞI', 'PORTFÖY BÜYÜKLÜĞÜ', 'YATIRIMCI SAYISI', 'TEDAVÜLDEKİ PAY ADETİ',
    'DOLULUK ORANI', 'Gunluk_Haftalik_Fark', '1Ay_3Ay_Fark', 'YATIRIMCI YOĞUNLUĞU',
    'BANKA BONOSU', 'BYF KATILMA PAYI', 'DEVLET TAHVİLİ', 'DİĞER',
    'DÖVİZ KAMU İÇ BORÇLANMA ARACI', 'DÖVİZ ÖDEMELİ BONO', 'DÖVİZ ÖDEMELİ TAHVİL', 'EUROBONDS',
    'FİNANSMAN BONOSU', 'FON KATILMA BELGESİ', 'GAYRİMENKUL SERTİFİKASI', 'GAYRİMENKUL YATIRIM FON KATILMA PAYI',
    'GİRİŞİM YATIRIM FON KATILMA PAYI', 'HAZİNE BONOSU', 'HİSSE SENEDİ', 'KAMU DIŞ BORÇLANMA ARACI',
    'KAMU KİRA SERTİFİKASI', 'KAMU KİRA SERTİFİKASI DÖVİZ', 'KAMU KİRA SERTİFİKASI TL',
    'KAMU YURT DIŞI KİRA SERTİFİKASI', 'KATILIM HESABI', 'KATILIM HESABI ALTIN', 'KATILIM HESABI DÖVİZ',
    'KATILIM HESABI TL', 'KIYMETLİ MADEN', 'KIYMETLİ MADEN CİNSİNDEN BYF', 'KIYMETLİ MADEN KAMU BORÇLANMA ARACI',
    'KIYMETLİ MADEN KAMU KİRA SERTİFİKASI', 'MEVDUAT ALTIN', 'MEVDUAT DÖVİZ', 'MEVDUAT TL',
    'ÖZEL SEKTÖR YURT DIŞI KİRA SERTİFİKASI', 'ÖZEL SEKTÖR DIŞ BORÇ ARACI', 'ÖZEL SEKTÖR KİRA SERTİFİKASI',
    'ÖZEL SEKTÖR TAHVİLİ', 'REPO', 'TAKASBANK PARA PİYASASI', 'TERS REPO', 'TÜREV ARACI',
    'VADELİ İŞLEMLER NAKİT TEMİNATI', 'VADELİ MEVDUAT', 'VARLIĞA DAYALI MENKUL KIYMET',
    'YABANCI BORÇLANMA ARACI', 'YABANCI BORSA YATIRIM FONU', 'YABANCI HİSSE SENEDİ',
    'YABANCI KAMU BORÇLANMA ARACI', 'YABANCI MENKUL KIYMET', 'YABANCI ÖZEL SEKTÖR BORÇLANMA ARACI',
    'YATIRIM FONLARI KATILMA PAYI', 'Yatirim_Paylari_Dolu_Sayisi'
]

# Yetişkin Fonlar için veri hazırlığı
yetiskin_fonlar_data = df1[df1['Fon_Yasi_Featured'] == 'Yetişkin Fonlar'][[target_column_yetiskin] + feature_columns_yetiskin].dropna()
X_yetiskin = yetiskin_fonlar_data[feature_columns_yetiskin]
y_yetiskin = yetiskin_fonlar_data[target_column_yetiskin]

# Eğitim ve test setlerine ayırma
X_train_yetiskin, X_test_yetiskin, y_train_yetiskin, y_test_yetiskin = train_test_split(X_yetiskin, y_yetiskin, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler_yetiskin = StandardScaler()
X_train_yetiskin = scaler_yetiskin.fit_transform(X_train_yetiskin)
X_test_yetiskin = scaler_yetiskin.transform(X_test_yetiskin)

# En iyi alpha değeri için GridSearchCV ile Lasso modelini optimize etme
lasso = Lasso(random_state=42)
param_grid = {'alpha': np.logspace(-4, 0, 50)}
grid_search_yetiskin = GridSearchCV(lasso, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search_yetiskin.fit(X_train_yetiskin, y_train_yetiskin)

# En iyi model ve alpha değeri
best_lasso_yetiskin = grid_search_yetiskin.best_estimator_

# Test seti ile tahmin yapma
y_pred_yetiskin = best_lasso_yetiskin.predict(X_test_yetiskin)

# Başarı metriklerini hesaplama
mse_yetiskin = mean_squared_error(y_test_yetiskin, y_pred_yetiskin)
rmse_yetiskin = np.sqrt(mse_yetiskin)
mae_yetiskin = mean_absolute_error(y_test_yetiskin, y_pred_yetiskin)
r2_yetiskin = r2_score(y_test_yetiskin, y_pred_yetiskin)

# Sonuçları yazdırma
print("Lasso Regression Model Performance for Yetişkin Fonlar:")
print("Mean Squared Error (MSE):", mse_yetiskin)
print("Root Mean Squared Error (RMSE):", rmse_yetiskin)
print("Mean Absolute Error (MAE):", mae_yetiskin)
print("R-squared (R²):", r2_yetiskin)

# Ağırlıkları görselleştirme
feature_weights_yetiskin = pd.DataFrame({
    'Feature': feature_columns_yetiskin,
    'Weight': best_lasso_yetiskin.coef_
}).sort_values(by='Weight', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(feature_weights_yetiskin['Feature'], feature_weights_yetiskin['Weight'], color='skyblue')
plt.xlabel('Weight')
plt.title('Feature Weights from Optimized Lasso Regression Model for Yetişkin Fonlar')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

##########################################################
#########################################################

#5	Model Kurma Ve Parametre Optimizasyonu Son

# Metrikleri hesaplayan bir fonksiyon
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}


# Ağırlıkları görselleştirmek için bir fonksiyon
def plot_feature_weights(model, feature_columns, title):
    weights = model.named_steps['model'].coef_
    feature_weights = pd.DataFrame({"Feature": feature_columns, "Weight": weights})
    feature_weights = feature_weights.sort_values(by="Weight", ascending=False)

    plt.figure(figsize=(12, 8))
    plt.barh(feature_weights["Feature"], feature_weights["Weight"], color='skyblue')
    plt.xlabel("Weight")
    plt.title(f"{title} - Feature Weights")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# Her fon grubu için en iyi model ve metrikleri saklayacağımız bir sözlük
best_models = {}

# Her fon grubu için ayarlamalar
for group_name, target_column, feature_columns in [
    ("Bebek Fonlar", '3AY', feature_columns_bebek),
    ("Genç Fonlar", '1YIL', feature_columns_genc),
    ("Yetişkin Fonlar", '5YIL', feature_columns_yetiskin),
]:
    print(f"\n== {group_name} İçin Sonuçlar ==")

    # Veri Hazırlığı
    fonlar_data = df1[df1['Fon_Yasi_Featured'] == group_name][[target_column] + feature_columns].dropna()
    X = fonlar_data[feature_columns]
    y = fonlar_data[target_column]

    # Eğitim ve test veri setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modeller ve parametreler
    models = {
        'Lasso': Lasso(random_state=42, max_iter=10000),
        'Ridge': Ridge(random_state=42, max_iter=10000),
        'ElasticNet': ElasticNet(random_state=42, max_iter=10000)
    }
    param_grids = {
        'Lasso': {'model__alpha': np.logspace(-4, 2, 100)},
        'Ridge': {'model__alpha': np.logspace(-4, 2, 100)},
        'ElasticNet': {'model__alpha': np.logspace(-4, 2, 100), 'model__l1_ratio': [0.1, 0.5, 0.7, 0.9]}
    }

    # Her model için GridSearchCV ile en iyi parametreleri bul ve modeli değerlendir
    group_results = {}  # Her grup için sonuçları saklayacağımız sözlük
    for model_name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        grid_search = GridSearchCV(pipe, param_grids[model_name], cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Modeli değerlendir
        metrics = evaluate_model(best_model, X_test, y_test)
        print(f"{model_name} - Best Params: {grid_search.best_params_} - Metrics: {metrics}")

        # Ağırlıkları çubuk grafik ile görselleştirme
        plot_feature_weights(best_model, feature_columns, f"{group_name} - {model_name}")

        # Grup sonuçlarına ekle
        group_results[model_name] = metrics

    # Her fon grubu için en iyi modeli seç ve sözlüğe ekle
    best_model_for_group = min(group_results.items(), key=lambda x: x[1]['MSE'])
    best_models[group_name] = {
        "Model": best_model_for_group[0],
        "Metrics": best_model_for_group[1]
    }

# Tüm fon grupları için en iyi modelleri ve başarı metriklerini görüntüleme
print("\nHer Fon Grubu İçin En İyi Modeller:")
for group, info in best_models.items():
    print(f"Fon Grubu: {group}, Model: {info['Model']}, Metrikler: {info['Metrics']}")