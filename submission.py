from keras.models import load_model
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

data = pd.read_csv('test.csv')
scene = data['Scene']

scene2 = []
for i in scene:
  replaced = re.sub(r"{.*?}", '', i)
  scene2.append(replaced)

stopwords_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

scene3 = []
for j in scene2:
  query = j
  resultwords  = [word for word in re.split("\W+",query) if word.lower() not in stopwords_list]
  result = ' '.join(resultwords)
  scene3.append(result)

num_words = 2000
tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,split=' ')
tokenizer.fit_on_texts(scene3)
X = tokenizer.texts_to_sequences(scene3)

word_index = tokenizer.word_index

max_length_of_text = 200
X = pad_sequences(X, maxlen=max_length_of_text)

model = load_model('weights.hdf5')
results = model.predict(X)

results = np.argmax(results, axis=1)

predictions = []

conv = {0: "POSTIVE",
	   1: "NEGATIVE",
	   2: "MIXED",
	   3: "NEUTRAL"}

for i in results:
	predictions.append(conv[i])

new_df = pd.DataFrame({"Index": [i+1 for i in range(len(predictions))], "Sentiment": predictions})
new_df.to_csv("solution.csv", index=False)