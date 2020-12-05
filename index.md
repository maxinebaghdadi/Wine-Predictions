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

# Procedure  

1. **Reading and Cleaning the Data:** We focused in on country, description, price, province, title, variety. We dropped rows without the price and NaN entries. 
2. **





```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/maxinebaghdadi/winepredictions/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
