# [Feedback Prize - English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/overview): My first(-ish) Kaggle competition

I'm new to Kaggle and wanted to take this challenge as an opportunity to:

- learn how to compete in Kaggle: How to set things up, manage the constraint of number of submissions etc.. You might not realize it if you have been competing for a while, but there is a learning curve.
- start using Pytorch (until now, I have been using Tensorflow for ANNs)
- learn more about NLP


My score during the competition was really bad: I didn't have time to do much more than setting things up and create a dummy model. But I kept working on it after it ended and obtained a decent score of **0.45** with a relatively simple solution that doesn't involve fine tuning a Large Language Model, although it relies on using one. I think this is not bad considering the best score was **0.43** and a dummy model gets a score of **0.63**.



Along the way, I've built and come across a few things that are hopefully worth your time. The main ones are:

- **The way I setup the code**: The code is executed from this [notebook](https://www.kaggle.com/code/josarago/best-tree-based-model-score-but-there-s-a-catch) which imports code from a [Kaggle Dataset](https://www.kaggle.com/datasets/josarago/kaggle--feedback-prize-ell) and synced with a [Github repo](https://github.com/josarago/kaggle--feedback-prize-ell) through [Github Actions](https://github.com/josarago/kaggle--feedback-prize-ell/blob/main/.github/workflows/main.yml). Meaning you can work in your favorite IDE, keep things neatly organize the way you like, push code to Github and finally run the code in Kaggle to generate submissions. I would imagine I am not the only one who has a hard time with 10,000 lines notebooks. Just the scrolling itself become a distraction.
- **The pattern I used to wrap pretrained models into sklearn transformer**: I looked around and couldn't find anything (please point me in the direction of any similar implementation). This is hopefully a reusable [pattern for transfer learning](https://github.com/josarago/kaggle--feedback-prize-ell/blob/fe7a51ce11021d2e773284addc16a08426228494/sklearn_transformers.py) and allowed me to leverage two pretrained models ([fasttext-langdetect](https://pypi.org/project/fasttext-langdetect/) and [DeBERTa](https://www.kaggle.com/datasets/kojimar/fb3models)) in combination with other features all while using sklearn pipelines. This also make it very easy to try different models using the same features. **However this is NOT useful if your goal is to fine-tune a Large Language Model using Pytorch.**
- **Some thoughts I share about the competition format**: again I am new to Kaggle so if my thoughts are naive or misguided, I would love to hear different point of views.


## Summary of the approach
I'll describe briefly the steps I took to get to 0.45. Again, this is far from being the best solution but it gets us *90% of the way* and is relatively low effort.

### EDA

Looking at the available data, I made a few observations:

- The target variables, these 6 English scores, are very highly correlated. The Pearson correlation coefficients between target variables are typically between ~0.64 and 0.74. Another way to look at it is by performing a PCA on the target variable. The first component accounts for 73.9% of the variance, while the other components are relatively evenly distributed, ranging from 3.7% to 6.6%. So it is likely that a few relevant features would do most of the work.
- Looking at the lowest and highest scores is useful to find feature ideas:
  - Many students with low scores used sms like english and used many contractions
  - High scores essay tend to be longer, and have more structure (paragraphs)
  - Some students used other languages than English. They might have recently immigrated and just started learning English (at least student mentions this in their essay), but since these essays are rated for English quality, they usually get lower scores.
  - Some students systematically put a whitespace **before** periods not after, for instance "My sentence just ended .Then it started again .". In general, respect for punctuation rules seems to correlate with higher scores.

### Setting things up

After finding a few simple feature ideas, I started writing a notebook and created a skeleton that can accommodate increasing number of features, different models and manages loading the data, a training script, the evaluation metrics and the function writing the submission file. 
I started by creating a simple class that made predictions from a **Dummy regressor** (always predicting the mean scores) which will be later swapped with an hopefully more useful model. The dummy regressor, which is not using the text at all, obtained a score of **0.632**. It's always good to start with something as simple as possible: In fact, if you look at some of the scores in the competition, a number of submission are worse than this dummy model, which suggests there is something wrong with the code.

### Baseline model

I then created a few features:
  - **total number of unigram used**: we said earlier that longer essays tend to have better ratings. 
  - **number of line breaks**: this is arguably a bit funny but an attempt to capture the fact that better essays have a better structure.

As you might expect that would not get us very far, these are actually not great first choices, but I was just curious what would happen. I'm not planning of doing a systematic breakdown of feature importances and I just wanted to see what these vaguely relevant features would give me. We should still see some progress. Using a Linear Regression model with these simple features gets us to **0.616**. This is not impressive by any mean, but we are moving in the right direction.

### More features and tree based model
I then kept going adding features that were easy to implement although unlikely to cause a huge improvement in the score. I just wanted to see how far I could go using simple but highly interpretable features:

- **TF-IDF**: it can help the model understand what vocabulary the students are using. At first I would used a dictionary listing common English contractions (I'll, we'd've, it'd etc..) but then realized I might as well use the full vocabulary instead as it should include the relevant contractions.
- **punctuation errors**: missing trailing whitespaces, extra leading whitespaces, missing brackets or quotes etc...
- **English score** using `fasttext-langdetect`. Like I mentioned earlier, some students used non English language which tend to get them a bad rating.

At this point, I started using a tree based model. I used lightGBM because, in my experience it is faster and usually works a little better than XGBoost with the default parameters, before any hyperparameter tuning (I could use a small dopamin hit to keep going). The score dropped to **~0.52**. That's a lot better but still far from **0.43**.


### More features and tree based model

I started the competition pretty late and, by the time I was setup to iterate conveniently on the model, the highest score was already the final score **0.43**. By that point it was clear I would not have time to get a good rank, so I decided to focus on learning from the best solutions instead. It's nice for people with a top score to be sharing their notebooks, although as I argue below, I think it limits the outcome of the competition. Most of the best solutions seemed to use DeBERTa, which provided me with a simple next step. Use pretrained DeBERTa model, feed the text through the model and, like so many, use the last hidden layer and a pooling layer to get a low(-ish) dimensional representation of each essay. But most people were actually fine tuning the model for the task at end which, quite frankly, is a probablly a better approach. But I decided to take another route for a few reasons:
- I had never touched Pytorch so far so just adding an extra feature in a similar way to what I did with `fasttext-langdetect` seemed easier than fine-tuninig the model
- Everyone was fine-tuning DeBERTa anyways, what could I get with a more straightforward/out-of-the-box approach? I was just curious.
- I wasn't sure how to combine the already built features/sklearn pipelines and didn't want to just discard them.

So I ended up building a [sklearn transformer](https://github.com/josarago/kaggle--feedback-prize-ell/blob/da76be074a8c757646adeb86d7ca0701e7249949/sklearn_transformers.py#L68) that apply mean pooling to the last hidden layer of the model after making predictions on the essay:
- the `fit` method function does nothing in this case, since we just use the pretrained model as is.
- the `transform` method can call either `simple_transform` or `batch_transform` that I ended up using for the submissions after having some out of memory issues when running the inference on GPU.
- in the end the transformer is wrapped in a [pipeline](https://github.com/josarago/kaggle--feedback-prize-ell/blob/da76be074a8c757646adeb86d7ca0701e7249949/pipelines.py#L99) so that I can just add it, using `FeatureUnion` to the existing features. Note that I had to add [this extremely weird](https://github.com/josarago/kaggle--feedback-prize-ell/blob/da76be074a8c757646adeb86d7ca0701e7249949/pipelines.py#L102) `FunctionTransformer` to make sure we reset the input DataFrame index is reset. without it, the indexes shuffled from the `train_test_split` call cause the `DataLoader` object to throw an error.

Doing this with LightGBM, XGBoost or a simple Pytorch model (not fine tuned) gives essentially the same results of **~0.45** ([notebook here](https://www.kaggle.com/code/josarago/best-tree-based-model-score-but-there-s-a-catch)) which I think is noteworthy if we consider the best model for this problem gets 0.43. Now that I have a simple way to generate features from any Language Model, if I have the time, I would like to explore other models than DeBERTa. After all there are tons of models specifically designed to find grammtical errors. This [list from Sebastian Ruder](https://github.com/sebastianruder/NLP-progress/blob/master/english/grammatical_error_correction.md) seems like a good place to start.



## Thoughts on the competition format

Kaggle is a great platform. I don't know of any other open platform that give access to so many interesting datasets and let you explore other users' approaches to solve real world problems. The competition format is instrumental as it incentivizes participants to give their best and keep improving their models. Howevever, I was very surprised to realize that people were sharing their detailed notebooks as the competition was still runnning. Looking back my impression is that ***people should not be able to share code related to the competition, while it's running.***

This may be an unpopular opinion, but I think it really beats the purpose of the competition format:

- **For the organizers, you get less bang for your buck**: If you organize a Kaggle competition, you invest time and money to solve a problem. The more you let people think independently about a problem the more creative people get, and avenues are explored. I think it would have been a lot better to have 10 solutions with a **0.45** scores, each with their own original approach, rather than hundreds of **0.43** solution that are often indistinguishable from each others, Sure it's not a zero sum game as people who just copied pasted the published solution with the highest score might not have spent time exploring anything else. But I feel like seeing seeing so many teams with the highest score acts a bit as a deterrent: It feels like the solution has already converged to the best possible and it will be extremely hard to beat it. Also, you might just want to do a deep dive into the top 10 scores and build a new model from scratch, combining all the most interesting ideas: you could also consider putting all these solutions into an ensemble model that will likely get you a better score.
- **For hiring organizations, a way to find the best ML practitioners**: If the top solution is shared, people can just come and copy and paste a solution and achieve a top score. These copy-pasted top scores are then hard to distinguish from original solution ones. So the ranking loses its value
- **For pariticipants**, it limits your creativity to see write ups on the highest score. It's very hard not to be tempted to focus on understanding them rather than pursuing your own ideas. It's also a bit unfair: a copy pasted solution could definitely obtain the final top score just by chance, which makes it less motivating.


All in all I really enjoyed the competition and learned a ton of things so, as far as I'm concerned, I'm very happy with my experience!
