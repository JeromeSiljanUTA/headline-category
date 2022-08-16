# Headline Category
Yes, it's a dumb name. I just wanted to acquire some NLP skills.

The objective of this project is to predict the news category of the article from its headline.

## Dataset
[News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
This dataset has about 200k news headlines and their respective news "category", such as politics, wellness, entertainment, etc.

## Cleaning the Data
I decided to use `pandas` to read the json file since each line of the file is its own json object, which means that importing it 

For the purpose of this classification project, I didn't need the article link, authors, description, or date, so I dropped those columns. ~~Instead of using `sklearn`'s `train_test_split`, I decided to come up with my own solution because I wanted my training and testing data to be evenly distributed for each news category.~~ I found out that my problem could be solved by using `train_test_split` and enabling stratified sampling. 
