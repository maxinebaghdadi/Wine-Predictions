# Overview

This project implements the Bag of Words model on the following [Wine Reviews dataset](https://www.kaggle.com/zynicide/wine-reviews?select=winemag-data-130k-v2.csv) to create a recommendation system. 

### Background and Goal

The holidays are coming up and you are at Trader Joe's or Costco and you have a small family gathering later today. Being not the best cook, you decide that it is safe just to bring a bottle of wine (or two) to the party. You are overwhelmed with the variety of choices and have a budge contraint. You know what kinds of fruity flavors that your grandpa likes, and your aunt mentioned something about liking "dry" wines. However, you are not that familar with specific charactersitcs of certian kinds of wines and their countires of origin. What is the difference beween a Pinot Noir from France and a Cabernet Sauvignon from the Napa Valley of California?


Our project aims to create a predictive model to recommend wine based on the description and price. The ultimate goal of the project is to let a user have a good recommendation of a wine based on their taste preferences and budget. 


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


### Bag of Words Model 

The Bag of Words Model is a state-of-the-art natural language processor that represents a text (such as a sentence or a document) as a bag (multiset) of words. This model disregards grammar, the order of words, but keeps track of multiplicity (how often a word occurs in the text). In practice, Bag of Words is mainly used as a tool for feature generation. 

Below is an example of how Bag of Words works in a simple text document. The key of each mapping result is the word, and it is followed by the number of occurences of that word in the given string of text.

1. Nina's favorite dessert is chocolate cake. She likes plain dark chocolate as well.
2. Jenny prefers chips over chocolate cake. She thinks that chocolate cake is too rich.

```markdown
BoW1 = {"Ninas":1, "favorite":1, "dessert":1, "is":1, "chocolate":2, "cake":1, "She":1, "likes":1, "plain":1, "dark":1, "well":1}
BoW2= {"Jenny":1, "prefers":1, "chips":1, "over":1, "chocolate":2, "cake":2, "She":1, "thinks":1, "that":1, "is":1, "too":1, "rich":1}
```

For this project specifically, we looked at the most commonly used words to describe wines and mapped them to what type of wines they are  matched to. For example, in the descriptions of Pinot Noirs versus Chardonnays, we looked at which type of wine has the most occurences of the word "dry" in their description. 

Something to note is that frequencies are not neccesarily the best representations of the text. Especially while describing wine, we want to get rid of filler words such as ‘and’, ‘the’, and ‘a’. Instead, we want to focus on the adjectives that are more significant to describing wine such as ‘crisp’, ‘vanilla’, and ‘bright’. This is something we accounted for in our methodology. 

# Methodology  
### Reading and Cleaning the Data: 

In order to clean up the data, we filtered for country, description, price, province, title, variety. We also dropped rows without the price and NaN entries and removed datapoints with same title and description.

```markdown
df = df[['country','description','price','province','title','variety']]
df = df.dropna()
df = df.drop_duplicates(['description, 'title']) 
df.head()
```


| country       | description                                        | price  | province      | title                                         | variety       |    
| ------------- | -------------------------------------------------  | ------ | ------------- | --------------------------------------------- |------------- | 
| Portugal      | This is ripe and fruity, a wine that is smooth...  | 15.0   | Douro         | Quinta dos Avidagos 2011 Avidagos Red (Douro) | Portuguese Red
| US            | Tart and snappy, the flavors of lime flesh and...  | 14.0   | Oregon        | Rainstorm 2013 Pinot Gris (Willamette Valley)	| Pinot Gris
| US            | Pineapple rind, lemon pith and orange blossom ...  | 13.0   | Michigan      | St. Julian 2013 Reserve Late Harvest Riesling | Riesling 
| US            | Much like the regular bottling from 2012, this...	 | 65.0   | Oregon        | Sweet Cheeks 2012 Vintner's Reserve Wild Child| Pinot Noir 
| Spain         | Blackberry and raspberry aromas show a typical...	 | 15.0   | Northern Spain| Tandem 2011 Ars In Vitro Tempranillo-Merlot   | Tempranillo-Merlot



### Exploratory Data Analysis: 

To get familiar with the data, we conducted some basic EDA. 

***What is the country wise average wine price?***

<img src="avg. country price.png" alt="hi" class="inline"/>

***What scores did the frequent varieties receive? 





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

### Model 1: Run MultinomialNB Regression
(processed description → variety)

The purpose of this model is to give you a variety of wine based on your flavor preferences.

### Model 2: Run Multinomial Logistic Regression
(variety, price → country)

The purpose of this model is to give you a more precise recommendation since we also need to account for price. 
We realize that the first model just gives you a variety, such as Pinot Noir or Malbec, but there are also drastic variety in prices. For example, a Swiss Pinot Noir on average is more expensive than Chile Pinot Noir. 

### Model 3: Run Logistic Regression
(price → country)

This was a third model that we ran just to gather some more data for our experiment

# Results 

# Conclusion

There a few areas for improvement that we wanted to noted out. The first areas is to use either the BERT model or word2vec instead of Bag of Words. 

The BERT Model is a state-of-the-art natural language processor recently developed by researchers at [Google AI Language](https://arxiv.org/pdf/1810.04805.pdf). Past models looked at a text sequence from left to right or combined left-to-right and right-to-left training, BERT, on the other hand, implements bidirectional training of a Transformer and thus has a deeper flow and language context than single-direction models. It is largely applied in classification tasks (including sentiment analyses), question answering tasks and named entity recognition (NER). Fundamentally this transformer includes two mechanisms: an encoder that reads the text input and a decoder that produces a prediction for task. 
BERT uses two training strategies: 
  1. Masked LM: 15% of the words in a sequence fed into BERT are replaced with a Mask token. The model predicts the original value of the masked words based on the context of the non-masked. Its loss function only considers the prediction of the masked values. 
  2. Next Sentence Prediction: 50% of inputs are a pair in which the second sentence is a subsequent sentence in the original and the other 50% is where random sentences from the text are chosen as the second sentence. This trains the model to disconnect the random sentence from the first.  
  
Word2Vec is another technique for natural language processing. This algorithm uses a shallow two layer neural network model to learn word associations from a large structure of texts. The way that the algorithm works is that Word2vec represents each distinct word with a particular vector and produces a vector space that is typically several hundreds of dimensions. The way that the words are placed in the vector space are based on how sematantically related they are to each other. 

The second area of improvement is to experiment with different classification models. Most of the algorithms that we tried are under the topic of Multi-Nomial Classification. Some popular algorithms that we tried are Naive Bayes and Logsitic Regression, but for the future, we want to see if we can try Decision Trees, k-Nearest Neighbors, and maybe even Random Forest. 

# Acknowledgements & References 
This project would not have been possible without the support from [Professor Mike Izbicki](https://izbicki.me/). 


