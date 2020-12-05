# Overview

This project implements the Bidirectional Encoder Representations from Transformers (BERT) model on the following [Wine Reviews dataset](https://www.kaggle.com/zynicide/wine-reviews?select=winemag-data-130k-v2.csv) to create a recommendation system. This model takes a description and price range as input and recommends a specific wine as its output. 


### Wine Reviews Data Set 

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for


### BERT Model 

The BERT Model is a state-of-the-art natural language processor recently developed by researchers at [Google AI Language](https://arxiv.org/pdf/1810.04805.pdf). In contrast to past models that looked at a text sequence from left to right or combined left-to-right and right-to-left training, BERT implements bidirectional training of a Transformer and thus has a deeper flow and language context than single-direction models. Fundamentally this transformer includes two mechanisms: an encoder that reads the text input and a decoder that produces a prediction for task. 

BERT uses two training strategies: 
  1. Masked LM: 15% of the words in a sequence fed into BERT are replaced with a Mask token. The model 



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
