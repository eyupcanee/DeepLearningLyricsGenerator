# Gerekli Kütüphanelerin Import Edilmesi
from __future__ import print_function
import io
import os
import sys
import string
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding

translator = str.maketrans('', '', string.punctuation)

# Verilerin Import Edilmesi

df = pd.read_csv("./lyrics.csv", sep="\t")
df.head()
pdf = pd.read_csv("./PoetryFoundationData.csv", quotechar='"')
pdf.head()


# Veri temizleme işlemleri.
# Bu işlemi lyrics.csv verimizi incelediğimizde [Intro], [Verse 1...5], [Chorus] olarak ayrıldığını görüyoruz.
# Bunları ayırıp introyu aradan kaldırarak olduğu kadar verse ile chorusu birleştirerek tek bir metin elde ediyoruz.
def split_text(x):
   text = x['lyrics']
   if pd.isna(text):
       return np.nan
   sections = text.split('\\n\\n')
   keys = {'Verse 1': np.nan,'Verse 2':np.nan,'Verse 3':np.nan,'Verse 4':np.nan, 'Chorus':np.nan}
   lyrics = str()
   single_text = []
   res = {}
   for s in sections:
       key = s[s.find('[') + 1:s.find(']')].strip()
       if ':' in key:
           key = key[:key.find(':')]
          
       if key in keys:
           single_text += [x.lower().replace('(','').replace(')','').translate(translator) for x in s[s.find(']')+1:].split('\\n') if len(x) > 1]
          
       res['single_text'] =  ' \n '.join(single_text)
   return pd.Series(res)
df = df.join( df.apply(split_text, axis=1))
df.head()

## Şiir verisetimiz için de temizleme işlemleri.
pdf['single_text'] = pdf['Poem'].apply(lambda x: ' \n '.join([l.lower().strip().translate(translator) for l in x.splitlines() if len(l)>0]))
pdf.head()

## İki farklı veri setimizi bir araya getirerek modelimizi eğiteceğimiz veriyi hazırlamış oluyoruz.
sum_df = pd.DataFrame(df['single_text'])
sum_df = sum_df.append(pd.DataFrame(pdf['single_text']))
sum_df.dropna(inplace=True)

## Veri setimizdeki yaygın ve yaygın olmaya kelimeleri filtreleme.
text_as_list = []
frequencies = {}
uncommon_words = set()
MIN_FREQUENCY = 7
MIN_SEQ = 5
BATCH_SIZE =  32
def extract_text(text):
   global text_as_list
   text_as_list += [w for w in text.split(' ') if w.strip() != '' or w == '\n']
sum_df['single_text'].apply( extract_text )
print('Toplam kelimeler: ', len(text_as_list))
for w in text_as_list:
   frequencies[w] = frequencies.get(w, 0) + 1
  
uncommon_words = set([key for key in frequencies.keys() if frequencies[key] < MIN_FREQUENCY])
words = sorted(set([key for key in frequencies.keys() if frequencies[key] >= MIN_FREQUENCY]))
num_words = len(words)
word_indices = dict((w, i) for i, w in enumerate(words))
indices_word = dict((i, w) for i, w in enumerate(words))
print('Belirli bir sıklıktan daha az {} görülen kelimeler: {}'.format( MIN_FREQUENCY, len(uncommon_words)))
print('Belirli bir sıklıktan daha çok {} görülen kelimeler: {}'.format( MIN_FREQUENCY, len(words)))
valid_seqs = []
end_seq_words = []
for i in range(len(text_as_list) - MIN_SEQ ):
   end_slice = i + MIN_SEQ + 1
   if len( set(text_as_list[i:end_slice]).intersection(uncommon_words) ) == 0:
       valid_seqs.append(text_as_list[i: i + MIN_SEQ])
       end_seq_words.append(text_as_list[i + MIN_SEQ])
      
print('Belirli bir uzunluktaki {} geçerli diziler: {}'.format(MIN_SEQ, len(valid_seqs)))

## Verilerimizi train ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(valid_seqs, end_seq_words, test_size=0.02, random_state=42)


