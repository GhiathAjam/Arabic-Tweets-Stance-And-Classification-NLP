# Arabic-NLP-Text-Stance-And-Classification
NLP Python code that calculates stance and classifies Arabic tweets about COVID-19 vaccination.

<!--
<img src="/deliverables/Presentation/01.jpg" width="800" />
-->

![1](./deliverables/Presentation/01.jpg "1")
---
## Pipeline
![2](./deliverables/Presentation/02.jpg "2")
---
## Data Cleaning
![4](./deliverables/Presentation/04.jpg "4")

### Data Balancing
Problem was un-avoidable as unlike 'accuracy', 'macro f1' score will just collapse when we ignore some very low probabilty classes.
We implemented two approaches:
1. **Oversampling**

   More samples for minoriy classes, using `class_weight='balanced'` in `scikit-learn` classifiers.
2. **Penalizing mistakes**

   Higher penalty for minority classses, using `imblearn`.

### Data preprocessing:
1. Removing Diacritization (التشكيل) and punctuation.
2. Replacing links, numbers and mentions with <link>, <num> and <mt>.
3. Converting emojis to equivalent text. (😂 -> face_tearing_with_joy)
4. Normalizing letters. (أ إ آ  -> اا)
5. Lemmatization using multiple tools. (camel, farasa, nltk stem)
6. Converting English text to lowercase.
7. Repeating hashtag words n times.
8. Removing stopwords. (combined nltk and Arabic-Stopwords) e.g. 'وأيها' , 'عندنا' , 'معي'.
9. Removing duplicate rows.
10. Tokenization using camel-tools simple word tokenizer.

#### Additional Ideas:
1. Translate English text.
2. Do everything in Arabic. (emoji meanings, tokens, english text translation, …)
3. Remove numbers or convert to word representation.
4. Better stop words datasets, can also remove too rare / too frequent words.
5. Named entity recognition. (NER)
---
## Feature Extraction
* **Bag of words (BOW)**
   * Simplified representation of data, where we don’t care about the context but only about what words occurred in what tweets and by how much.
   * The feature is a matrix |Vocab| * |tweet | -> cell[word, tweet] == count(word in tweet).
* **Continuous BOW (word embeddings / vectors)**
   * Word2Vec model where a word is predicted based on surrounding words, a 3-layer NN is chosen and trained using a huge corpora.
   * The embeddings for each word is the weights between the input layer and the hidden layer.
* **Skip-gram (word embeddings / vectors)**
   * Word2Vec model where surrounding words are predicted based on current word, a 3-layer NN is chosen and trained using huge corpora.
   * The embeddings for each word is the weights between the input layer and the hidden layer.
* **TF-IDF**
   * Term Frequency-Inverse Document Frequency.
   * TF(w, tweet) = count(w in tweet), this eliminates very rare words.
   * IDF(w) = #tweets / #(tweets with the word w), this eliminates too frequent words.
   * We used both Word n-grams and Character n-grams.
* **Arabert Embeddings as a feature for SVM**
   * We took the “pooler output” from Bert, which resemble embeddings and feed them to SVM as a feature
---

## Models
### Classical Models
  * SVM
  * Naive Bayes
  * KNN
  * Decision Trees
  * Random Forest with `n_estimators = 1000`
  * Logistic Regression with `n_iterations = 300`

#### Some Results for Stance Detection
Best results was:
Preprocessing | Features | Classifier Model | Acc | F1
--- | --- | --- | --- | ---
Farasa lemmatize + dediacritized camel lemmatize + original text | TFIDF char + word | Linear SVM | 80 | 56

Other results:
Features | Classifier Model | Acc | F1
--- | --- | --- | ---
BOW | Naive Bayes | 56 | 42
BOW | Logistic Regression | 77 | 55
BOW | Random Forest | **80** | 46
BOW | Linear SVM | 76 | 51
CBOW | Ridge Classifier | 73 | 38
S-Gram | Ridge Classifier | 80 | 40
---
### Sequence Models
In the sequence models family, we’ve built a 3-layer LSTM followed by a linear neural network layer.

LSTM’s Training Settings:
Epochs | Batch Size | Learning rate | Embedding Dimension | LSTM Hidden layers Dimension
--- | --- | --- | ---- | ----
50 | 256 | 0.001 | 300 | 50

Best result on **Categorization** problem:
Data | Features | Classifier Model | Acc | F1
--- | --- | --- | --- | ---
Oversampled data | Embedding Layer | 3-layer LSTM + 1 NN layer | 56.6 | 25.9
---
### AraBert (Transformers)

<p align="middle">
  <img src="https://github.com/aub-mind/arabert/blob/master/arabert_logo.png" width="150" align="left"/>
  <img src="https://github.com/aub-mind/arabert/blob/master/AraGPT2.png" width="150"/>
  <img src="https://github.com/aub-mind/arabert/blob/master/AraELECTRA.png" width="150" align="right"/>
</p>

In the transformers family, we’ve fine-tuned an arabic bert model on our dataset.
* The arabic bert used was aubmindlab/bert-base-arabertv02-twitter from hugging face. We’ve chosen this model because it was trained on ~60 Million Arabic tweets.
* As per the documentation, we’ve used the preprocessing and tokenizer that was used when the model authors built their model.
Fine-tuning:
* We use the araBert as a feature extractor, by first freezing the bert’s parameters, then passing the data through this arabert model, and producing the embedding as output. The sentence embedding is calculated by taking the last layer hidden-state of the first token of the sequence [CLS token].
* Then the sentence embeddings enters a classifier head we’ve built. The classifier head consists of 2 neural network layers in order to fine tune the weights of the model on our data.
* Arabert Training Settings:

   Epochs | Batch Size | Learnign rate
   --- | --- | ---
   50 | 16 | 0.001
* Arabert embeddings with SVM classifier:

   Extract sentences’ embeddings using arabert, then train these embeddings using a linear-kernel SVM.
---
## Final Results
![13](./deliverables/Presentation/13.jpg "13")

<!-- 
![3](./deliverables/Presentation/03.jpg "3")

![5](./deliverables/Presentation/05.jpg "5")
![6](./deliverables/Presentation/06.jpg "6")
![7](./deliverables/Presentation/07.jpg "7")
![8](./deliverables/Presentation/08.jpg "8")
![9](./deliverables/Presentation/09.jpg "9")
![10](./deliverables/Presentation/10.jpg "10")
![11](./deliverables/Presentation/11.jpg "11")
![12](./deliverables/Presentation/12.jpg "12")

-->
### Libraries Used
+ nltk
+ Arabic-Stopwords
+ camel-tools==1.2.0
+ farasapy
+ arabert
+ pandas
+ scikit-learn
+ gensim
+ Transformers
+ imblearn
---
## الحمدلله
![14](./deliverables/Presentation/14.jpg "14")
