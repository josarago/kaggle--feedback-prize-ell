# [Feedback Prize - English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/overview)
 
## My first(-ish) Kaggle competition

I'm new to Kaggle and wanted to take this challenge as an opportunity to :
- learn how to compete in Kaggle. You might not realize it if you have been competing for a while, but there is a learning curve.
- start using Pytorch (until now, I have been using Tensorflow for ANNs)
- see what I can learn about NLP

### Why you might want to read this
Along the way I did a couple things you might find interesting:
- The code is synced with the Github repo  through [Github Actions](https://github.com/josarago/kaggle--feedback-prize-ell/blob/main/.github/workflows/main.yml) which could be useful for your next competition
- I implementing what I think is a weird but probably sometimes useful pattern: I use the deberta model solely as a way to create a simple feature: Mean Pooling from the last hidden layer from a pretrained model. You probably generally want to fine tune the model on your task but was curious about what kind of score this more *out-of-the-box* approach would give. ~0.45 with either XGBoost, LightGBM or a relatively simple neural net. [The pretrained deberta model is wrapped in a sklearn transformer](https://github.com/josarago/kaggle--feedback-prize-ell/blob/fe7a51ce11021d2e773284addc16a08426228494/sklearn_transformers.py#L68)


## My approach
### Setting things up
I started by creating a simple class that made predictions from a Dummy regressor. That allowed me to set up loading the data, a training script, the evaluation metrics, the function writing the submission file. 

### Baseline model
For this I would need a few useful features and a very simple model. Quite frankly I think one could probably create a very interpretable linear model by creating an exhaustive list of english rules and corresponding functions to catch errors. Using a TF-IDF transformer to capture *good* and *bad* vocabularies. Is there the right number of whitespaces after and before punctuation marks? But also how many words did the studen use? etc...
If you perform a PCA on the scores or target variables, you'll see that **a single dimension accounts for ~75% of the variance**. That means that most students ratings are 



Is started by creating a notebook with a dummy regressor to make sure that I could actually properly make a submission.
Then I wanted to try something simple: write some simple functions that would capture obvious issues with the essays: issue with punctuation
