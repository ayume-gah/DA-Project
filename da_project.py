import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(
    layout="wide",
    page_icon=':basketball:',
    page_title="DA Project",
)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown ("<h1 style='margin-bottom:30px;text-align:center;'> Market Research and Recommendation and Visualization Technique for Business Decision Making </h1>", unsafe_allow_html=True)

#st.header('Market Research and Recommendation and Visualization Technique for Business Decision Making')
st.write('''
**DQLab sport center** adalah toko yang menjual berbagai kebutuhan olahraga seperti Jaket, Baju, Tas, dan 
Sepatu. Toko ini mulai berjualan sejak tahun 2013, sehingga sudah memiliki pelanggan tetap sejak lama, 
dan tetap berusaha untuk mendapatkan pelanggan baru sampai saat ini.
 
Di awal tahun 2019, manajer toko berusaha memecahkan masalah yang ada di tokonya, yaitu menurunnya 
pelanggan yang membeli kembali ke tokonya. Manajer toko mendefinisikan bahwa *customer* termasuk sudah bukan
disebut pelanggan lagi (*churn*) ketika dia sudah tidak bertransaksi ke tokonya lagi sampai 
dengan 6 bulan terakhir dari update data terakhir yang tersedia.  

Data transaksi dari tahun 2013 sampai dengan 2019 dalam bentuk csv dengan jumlah baris 100.000 baris data.
''')

st.write('##### ETL Data')
df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/data_retail.csv', sep=';')
st.write('Berikut adalah tiga data teratas,')
st.table(df.head(3))
#st.write('Dengan info data sebagai berikut,')
#st.image('data_info1.jpg')
st.write('\n')


st.write('''Data pada kolom *First_Transaction* dan *Last_Transaction* masih berupa data milidetik 
sehingga perlu diubah menjadi tipe data *datetime* terlebih dahulu untuk mempermudah proses pengolahan data.\n
Untuk mengubah menjadi tipe data *datetime*, digunakan perintah pd.to_datetime.
Sehingga isi data menjadi berikut,''')

#di kolom First_Transaction ini sebenarnya adalah data waktu dan tanggal, namun masih dalam bentuk angka 
# milisecond atau milidetik yang dihitung per tanggal 1 Januari 1970, ini info penting, nanti bisa dicari 
# lagi di google, kenapa dimulai pada tanggal tersebut
#nah, agar menjadi second, kita bagi dengan 1000, hasilnya di kolom First_Transaction_per_1000
#untuk kolom Datetime_First_Transaction_per_1000, itu hanya saya masukkan ke fungsi pd.to_datetime() tanpa 
# mengkustom parameter apapun, sekarang keliatan kalau semuanya berada di tanggal 1 Januari 1970, lalu 
# bagaimana biar terkonversi ke tanggal sebenarnya?
#cek di kolom Datetime_First_Transaction_per_1000_UnitOrigin, di sini, pada fungsi pd.to_datetime(), 
# diberikan isian pada parameter unit yakni s atau second, artinya, kita mengonversi angka di kolom kedua 
# berdasarkan satuan detik atau second
#dan yang kedua, parameter origin yang mengatur waktu awal atau asal atau origin nya, untuk menghitung 
# kalau "Sejak 1 Januari 1970, sudah berlalu selama ... detik"

# Kolom First_Transaction
df['First_Transaction'] = pd.to_datetime(df['First_Transaction']/1000, unit='s', origin='1970-01-01')
# Kolom Last_Transaction
df['Last_Transaction'] = pd.to_datetime(df['Last_Transaction']/1000, unit='s', origin='1970-01-01')
st.table(df.head(3))
#st.image('data_info2.jpg')

st.write('''Dari kolom *Last_Transaction* dapat ditentukan kapan transaksi terakhir di dalam dataset. 
Yaitu pada tanggal,''')
a=max(df['Last_Transaction'])
st.warning(a)
st.write('''Kemudian *customer* diklasifikasikan menjadi dua, *churn* dan tidak.\n 
*Churn* adalah ketika *customer* tidak ada traksaksi selama enam bulan dari tanggal transaksi terakhir 
dalam dataset.
''')
#menentukan is_churn
df.loc[df['Last_Transaction'] <= '2018-08-01', 'is_churn'] = True 
df.loc[df['Last_Transaction'] > '2018-08-01', 'is_churn'] = False
#st.table(df.head())
st.write('''Kolom *no* dan *Row_Num* tidak dibutuhkan dalam pengolahan data, jadi perlu dihapus.
Sehingga tampilan data menjadi seperti berikut,''')
#menghapus kolom yang tidak diperlukan
del df['no']
del df['Row_Num']
st.table(df.head(16))
#st.table(df.tail())

st.write('##### Visualisasi Data')
# Kolom tahun transaksi pertama
df['Year_First_Transaction'] = df['First_Transaction'].dt.year
# Kolom tahun transaksi terakhir
df['Year_Last_Transaction'] = df['Last_Transaction'].dt.year

st.write('''*Customer acquisition* merupakan salah satu konsep marketing yang paling sering dilakukan di 
dalam bisnis, karena setiap bisnis membutuhkan *customer* untuk bisa berkembang.\n
*Customer acquisition* penting untuk mengembangkan bisnis secara berkesinambungan. Melalui *acquisition*, 
dapat diciptakan *customer loyalty* dalam bisnis. Jika *acquisition* berhasil, maka perusahaan akan mempunyai 
banyak *customer*, dan dengan banyaknya *customer* maka perusahaan dapat membuat program *loyalty* yang 
dapat membuat *customer* semakin senang berlangganan dengan brand perusahaan.''')
st.write('###### *Trend of customer acquisition by year* digambarkan dalam bentuk grafik bar.\n')
fig=plt.figure(figsize=(10,5)) 
df_year = df.groupby(['Year_First_Transaction'])['Customer_ID'].count()
df_year.plot(x='Year_First_Transaction', y='Customer_ID', kind='bar', title='Graph of Customer Acquisition')
plt.xlabel('Year_First_Transaction')
plt.ylabel('Num_of_Customer')
plt.tight_layout()
#plt.show()
col1,col2,col3=st.columns([1,8,1])
col2.pyplot(fig)

st.write('###### Grafik dari tren transaksi per tahun.\n')
fig=plt.figure(figsize=(10,5)) 
plt.clf()
df_year = df.groupby(['Year_First_Transaction'])['Count_Transaction'].sum()
df_year.plot(x='Year_First_Transaction', y='Count_Transaction', color='Green', kind='bar', 
            title='Graph of Transaction Customer')
plt.xlabel('Year_First_Transaction')
plt.ylabel('Num_of_Transaction')
plt.tight_layout()
col1,col2,col3=st.columns([1,8,1])
col2.pyplot(fig)

st.write('''Dari dua grafik diatas dapat dilihat pada **tahun 2018**, nilai *Customer Acquisition* yang 
tinggi ternyata tidak diimbangi dengan nilai transaksinya. Nilai transaksi pada tahun 2018 turun cukup jauh
dari tahun sebelumnya. Hal ini disebabkan adanya *churned customer*, *curtomer* yang tidak berbelanja lagi
selama enam bulan dari transaksi terakhirnya.''')
st.write('\n')
st.write('###### Grafik tren rata-rata jumlah transaksi untuk tiap-tiap produk per tahun.\n')
fig=plt.figure(figsize=(10,5)) 
plt.clf()
sns.pointplot(data = df.groupby(['Product', 'Year_First_Transaction']).mean().reset_index(), 
              x='Year_First_Transaction', 
              y='Average_Transaction_Amount', 
              hue='Product')
plt.tight_layout()
col1,col2,col3=st.columns([1,8,1])
col2.pyplot(fig)

st.write('###### Proporsi *churned customer* untuk setiap produk.\n')
fig=plt.figure(figsize=(10,7))
plt.clf()
# Melakukan pivot data dengan pivot_table
df_piv = df.pivot_table(index='is_churn', 
                        columns='Product',
                        values='Customer_ID', 
                        aggfunc='count', 
                        fill_value=0)
# Mendapatkan Proportion Churn by Product
plot_product = df_piv.count().sort_values(ascending=False).head(5).index
# Plot pie chartnya
df_piv = df_piv.reindex(columns=plot_product)
df_piv.plot.pie(subplots=True,
                figsize=(10, 7),
                layout=(-1, 2),
                autopct='%1.0f%%',
                title='Proportion Churn by Product')
#plt.pie(df_piv, subplots=True, layout=(-1, 2), autopct='%1.0f%%', startangle=90)
plt.tight_layout()
col1,col2,col3=st.columns([2,6,2])
col2.pyplot()

#st.write('Grafik kategori jumlah transaksi')
fig=plt.figure(figsize=(10,5))
plt.clf()
# Kategorisasi jumlah transaksi
def func(row):
    if row['Count_Transaction'] == 1:
        val = '1. 1'
    elif (row['Count_Transaction'] > 1 and row['Count_Transaction'] <= 3):
        val ='2. 2 - 3'
    elif (row['Count_Transaction'] > 3 and row['Count_Transaction'] <= 6):
        val ='3. 4 - 6'
    elif (row['Count_Transaction'] > 6 and row['Count_Transaction'] <= 10):
        val ='4. 7 - 10'
    else:
        val ='5. > 10'
    return val
# Tambahkan kolom baru
df['Count_Transaction_Group'] = df.apply(func, axis=1)
 
df_year = df.groupby(['Count_Transaction_Group'])['Customer_ID'].count()
df_year.plot(x='Count_Transaction_Group', y='Customer_ID', kind='bar', color='Purple',
            title='Customer Distribution by Count Transaction Group')
plt.xlabel('Count_Transaction_Group')
plt.ylabel('Num_of_Customer')
plt.tight_layout()
st.write('''###### *Customer* dikategorikan berdasarkan rentang jumlah nilai transaksinya. 
Diperlihat dengan grafik sebagai berikut,
''')
col1,col2,col3=st.columns([1,8,1])
col2.pyplot(fig)

#st.write('distribusi kategorisasi average transaction amount')
fig=plt.figure(figsize=(10,5))
plt.clf()
# Kategorisasi rata-rata besar transaksi
def f(row):
    if (row['Average_Transaction_Amount'] >= 100000 and row['Average_Transaction_Amount'] <= 250000):
        val ='1. 100.000 - 250.000'
    elif (row['Average_Transaction_Amount'] > 250000 and row['Average_Transaction_Amount'] <= 500000):
        val ='2. >250.000 - 500.000'
    elif (row['Average_Transaction_Amount'] > 500000 and row['Average_Transaction_Amount'] <= 750000):
        val ='3. >500.000 - 750.000'
    elif (row['Average_Transaction_Amount'] > 750000 and row['Average_Transaction_Amount'] <= 1000000):
        val ='4. >750.000 - 1.000.000'
    elif (row['Average_Transaction_Amount'] > 1000000 and row['Average_Transaction_Amount'] <= 2500000):
        val ='5. >1.000.000 - 2.500.000'
    elif (row['Average_Transaction_Amount'] > 2500000 and row['Average_Transaction_Amount'] <= 5000000):
        val ='6. >2.500.000 - 5.000.000'
    elif (row['Average_Transaction_Amount'] > 5000000 and row['Average_Transaction_Amount'] <= 10000000):
        val ='7. >5.000.000 - 10.000.000'
    else:
        val ='8. >10.000.000'
    return val
# Tambahkan kolom baru
df['Average_Transaction_Amount_Group'] = df.apply(f, axis=1)
 
df_year = df.groupby(['Average_Transaction_Amount_Group'])['Customer_ID'].count()
df_year.plot(x='Average_Transaction_Amount_Group', y='Customer_ID', kind='bar', color='Orange',
            title='Customer Distribution by Average Transaction Amount Group')
plt.xlabel('Average_Transaction_Amount_Group')
plt.ylabel('Num_of_Customer')
plt.tight_layout()
st.write('''###### *Customer* dikategorikan berdasarkan rentang rata-rata besar nilai transaksinya. 
Diperlihat dengan grafik sebagai berikut,
''')
col1,col2,col3=st.columns([1,8,1])
col2.pyplot(fig)

#selanjutnya akan menentukan feature columns dari dataset yang dimiliki, di sini dipilih kolom 
# Average_Transaction_Amount, Count_Transaction, dan Year_Diff. 
# Silakan dicreate dahulu kolom Year_Diff ini dan kemudian assign dataset dengan feature columns ini 
# sebagai variabel independent X. 
#Untuk target tentunya persoalan costumer dengan kondisi churn atau tidak, assign dataset untuk target 
# ini ke dalam variabe dependent y.

# Feature column: Year_Diff
df['Year_Diff'] = df['Year_Last_Transaction'] - df['Year_First_Transaction']
# Nama-nama feature columns
feature_columns = ['Average_Transaction_Amount', 'Count_Transaction', 'Year_Diff']
# Features variable
X = df[feature_columns] 
# Target variable
y = df['is_churn'].astype('bool')
st.write('''Dibuat variable X yang berisikan *feature column* dari kolom *Average_Transaction_Amount, 
Count_Transaction*, dan *Year_Diff*.\n 
Dibuat juga variabel y, sebagai variabel target yaitu kondisi *churn* dari *customer*.
''')

#Setelah variabel independent X dan variabel dependent y selesai dilakukan, maka pecahlah X dan y ke dalam
#bagian training dan testing. Bagian testing 25% dari jumlah entri data.
from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
st.write('''Variabel X dan y ini akan digunakan dalam permodelan prediksi. Yang sebelumnya harus dilakukan
 *training and testing*, dengan komposisi *testing* 25% dari jumlah data. Pemodelan dilakukan dengan metode
 **Logistic Regression** untuk memprediksi *churned customer*.
''')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
 
# Inisiasi model logreg
logreg = LogisticRegression()
 
# fit the model with data
logreg.fit(X_train, y_train)
 
# Predict model
y_pred = logreg.predict(X_test)
 
# Evaluasi model menggunakan confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
#print('Confusion Matrix:\n', cnf_matrix)
st.write('Untuk mengevaluasi model yang sudah dibuat, digunakan *confusion matrix* ')
st.write('Confusion Matrix:\n', cnf_matrix)

#Confusion matrix yang telah dihitung sebelumnya dapat divisualisasikan dengan menggunakan heatmap dari 
# seaborn.
fig=plt.figure(figsize=(6,3))
plt.clf()
# name  of classes
class_names = [0, 1] 
fig, ax = plt.subplots()
 
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
 
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g')
ax.xaxis.set_label_position('top')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
col1,col2,col3,col4=st.columns([1,4,4,1])
col2.write('*Confusion matrix* dapat divisualisasikan menggunakan headmap menjadi >>>')
col3.pyplot(fig)

from sklearn.metrics import accuracy_score, precision_score, recall_score
 
st.write('Keakuratan dari model yang sudah dibuat dapat dilihat dengan *performance metrics*-nya,') 
#Menghitung Accuracy, Precision, dan Recall
#print('Accuracy :', accuracy_score(y_test, y_pred))
st.write('Nulai *Accuracy* :', accuracy_score(y_test, y_pred))
#print('Precision:', precision_score(y_test, y_pred, average='micro'))
st.write('Nilai *Precision*:', precision_score(y_test, y_pred, average='micro'))
#print('Recall   :', recall_score(y_test, y_pred, average='micro'))
st.write('Nilai *Recall*   :', recall_score(y_test, y_pred, average='micro'))


col1,col2=st.columns([8,2])
col2.caption('Ayu Megah, Agustus 2022')
