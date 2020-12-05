# Overview

This project implements the Bidirectional Encoder Representations from Transformers (BERT) model on the following [Wine Reviews dataset](https://www.kaggle.com/zynicide/wine-reviews?select=winemag-data-130k-v2.csv) to create a recommendation system. This model takes a description and price range as input and recommends a specific wine as its output. 


### Wine Reviews Data Set 

This data was scraped from WineEnthusiast in 2017 and it contains the following variables: 


| Variable      | Description   | 
| ------------- | ------------- | 
| country       | Country that the wine is from  |
| description   | Distinctive elements of the taste of the wine  |
| designation   | Vineyard within winery where the graphes that made the wine are from  |
| points        | Rating on a scale from 1-100   |
| price         | Cost per bottle of wine  |
| province      | Province or state wine is from  |
| region_1      | Specified growing area within province  |
| region_2      | Additional growing areas  |
| taster_name   | Rater of the wine  |
| title         | Name of wine with vintage  |
| variety       | Type of grapes used to make wine | 
| winery        | Winery of specificied wine | 


### BERT Model 

The BERT Model is a state-of-the-art natural language processor recently developed by researchers at [Google AI Language](https://arxiv.org/pdf/1810.04805.pdf). Past models looked at a text sequence from left to right or combined left-to-right and right-to-left training, BERT, on the other hand, implements bidirectional training of a Transformer and thus has a deeper flow and language context than single-direction models. It is largely applied in classification tasks (including sentiment analyses), question answering tasks and named entity recognition (NER). Fundamentally this transformer includes two mechanisms: an encoder that reads the text input and a decoder that produces a prediction for task. 

BERT uses two training strategies: 
  1. Masked LM: 15% of the words in a sequence fed into BERT are replaced with a Mask token. The model predicts the original value of the masked words based on the context of the non-masked. Its loss function only considers the prediction of the masked values. 
  2. Next Sentence Prediction: 50% of inputs are a pair in which the second sentence is a subsequent sentence in the original and the other 50% is where random sentences from the text are chosen as the second sentence. This trains the model to disconnect the random sentence from the first.  
  
 We use BERT to extract descriptive terms from the Wine Reviews dataset to product prediction modelling. 

# Methodology  
### Reading and Cleaning the Data: 

Filtered for country, description, price, province, title, variety; dropped rows without the price and NaN entries; removed datapoints with same title and description

```markdown
df = df[['country','description','price','province','title','variety']]
df = df.dropna()
df = df.drop_duplicates(['description, 'title']) 
df.head()
```
| country       | description                                        | price  | province      | title                                         | variety       |    
| ------------- | -------------------------------------------------  | ------ | ------------- | -------------                                 |------------- | 
| Portugal      | This is ripe and fruity, a wine that is smooth...  | 15.0   | Douro         | Quinta dos Avidagos 2011 Avidagos Red (Douro) | Portuguese Red
| US            | Tart and snappy, the flavors of lime flesh and...  | 14.0   | Oregon        | Rainstorm 2013 Pinot Gris (Willamette Valley)	| Pinot Gris
| US            | Pineapple rind, lemon pith and orange blossom ...  | 13.0   | Michigan      | St. Julian 2013 Reserve Late Harvest Riesling | Riesling 
| US            | Much like the regular bottling from 2012, this...	 | 65.0   | Oregon        | Sweet Cheeks 2012 Vintner's Reserve Wild Child| Pinot Noir 
| Spain         | Blackberry and raspberry aromas show a typical...	 | 15.0   | Northern Spain| Tandem 2011 Ars In Vitro Tempranillo-Merlot   | Tempranillo-Merlot



### Exploratory Data Analysis: 
Analyzed X, Y, Z * insert graphs here 

### Text Preprocessing: 
Standardized description variable through stemming, lemmatization, stopword removal, noise removal and lowercasing 

```markdown
df['description']= df['description'].str.lower()
df['description']= df['description'].apply(lambda elem: re.sub('[^a-zA-Z]',' ', elem))  
df['description'].head()
```
| Variable      | Description   | 
| ------------- | ------------- | 
| 0             | this is ripe and fruity  a wine that is smooth...  |
| 1             | tart and snappy  the flavors of lime flesh and...  |
| 2             | pineapple rind  lemon pith and orange blossom ...  |
| 3             | much like the regular bottling from       this...   |
| 4             | blackberry and raspberry aromas show a typical...  |

```markdown
stopword_list = stopwords.words('english')
ps = PorterStemmer()
words_descriptions = words_descriptions.apply(lambda elem: [word for word in elem if not word in stopword_list])
words_descriptions = words_descriptions.apply(lambda elem: [ps.stem(word) for word in elem])
df['processed_description'] = words_descriptions.apply(lambda elem: ' '.join(elem))

all_words = [word for tokens in words_descriptions for word in tokens]
VOCAB = sorted(list(set(all_words)))
```

### Implementing "Bag of Words" Model:
Tokenized sentences to lists of words and created a vocabulary list of those words 

```markdown
tokenizer = RegexpTokenizer(r'\w+')
words_descriptions = df['description'].apply(tokenizer.tokenize)
words_descriptions.head()
```
| Variable      | Description   |
| ------------- | ------------- | 
| 0             | [this, is, ripe, and, fruity, a, wine, that, i... |
| 1             |[tart, and, snappy, the, flavors, of, lime, fl...  |
| 2             |[pineapple, rind, lemon, pith, and, orange, bl...  |
| 3             |  [much, like, the, regular, bottling, from, thi...  |
| 4             | [blackberry, and, raspberry, aromas, show, a, ...  |

```markdown
all_words = [word for tokens in words_descriptions for word in tokens]
df['description_lengths']= [len(tokens) for tokens in words_descriptions]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
```

### Splitting Data into Testing/Training Set (80/20 Split): 

Our final goal is to output variety and country. However, those 2 outputs are in 2 different columns in the dataframe. Therefore we are running to regressions:

1. Logistic Regression (Processed Description --> Variety) 
2. Multinomial Logistic Regression (Variety, Price --> Country)

We use GridSearchCV to find the optimal hyperparameters of this model to increase its accuracy. 

# Results 

# Conclusion

# Acknowledgements & References 
This project would not have been possible without the support from [Professor Mike Izbicki](https://izbicki.me/). 
