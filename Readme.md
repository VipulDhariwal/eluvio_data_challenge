# Eluvio Data Challange
## _By Vipul Dhariwal_


This is my attempt to increase the efficiency of scene segment prediction. There are two different files, one which requires higher memory and parallel processing capabilities and other that even worked on my personal computer. 

The task was to improve the already given prediction for a binary classification problem of scene segmentation. For the high-memory code I achieved following benchmarks:

- Mean AP: 48.65% 
- Mean Miou: 50.06%

These measures even surpassed the scores of [Rao]. The low-memory code yield perhaps a little imporvement but improvement nonetheless. The scores for the code were:
 - Mean AP: 45.55%
 - Mean Miou: 46.87%

## Analytical Insights

- Initial 2 and last 3 predictions probabilties given were 0, the reason might be intial prediction had differentiation of features.
- Class labels are imbalanced (generally around 9:1) for non-scene-transition and scene-transition.
- Scenes usually have multiple shots, with an average scene having 9.67 shots. 

The main idea in both the code is inspired the solution of [Rao]:

> We train the nueral network on the entire
> movie to let it learn the effect of 'sequence'
> of shots. 


## Common Approach in both the codes

Both codes had a number of approaches to increase the efficiency of prediction:

- The 64 movies were merged into 1 dataframe with movie_id (imdb_id) as one of the coulmns to ensure that the model differentiate the movies.
- The dataframe was divided with Stratified 4-fold cross validation. Stratification is done for minimizing class imbalance.
- Different hyperparameters were tested (150 combinations) to find the best combination.
- Learning rate was optimal at 0.1 and sigmoid activation function for probability generation.
- Movie title was replaced with a number and included in the training set.
- The ground truth was encoded as 0 or 1 from True or False.

And of course the weights were set to 1:9 for non-scene transitions and scene transitions respectively.

Requirements:

    Python 3.9.1
    NumPy v1.20.0
    scikit-learn 0.24.1
    tensorflow 2.2.0

## High-memory code
Steps followed were -
- All tensors were opened and flattened into a dataframe. Hence, 'place' is converetd into 2048 features and 512 features for each 'cast', 'audio' and 'action' features.
- All the values were normalized between 0 and 1 for effiecient training.
- The dataframe was cleaned for accidental NaN values.
- All 5 features (Place, Cast, Action, Audio and Prediction) were converetd into a flat N X 3585 dataframe.
- Sequential model with 50 LSTM layers was used with loss function as binarty crossentropy and dropout rate = 0.2.
- 80 Epochs were used as cross-validation results suggested it as optimal number.

## Low-Memory code
Steps followed were -
- Only 'scene_transition_boundary_prediction' was considered along with movie_id for training.
- 5 layers of Dense was used with 'relu' activation.
- Output was given by a Dense layer of sigmoid activation and learning rate was optimal at 0.1.

At the end the predicted list was extracted from the dataframe based on the movie_id and stored in the 64 movie files as pickle.

I also tried using the ensemble method for high-memory code. The idea was to get a prediction list based just on four features (place, cast, audio and action) and then combine the predicted list with given predicted list to train a new model to get the final probabilty list. However that model did not gain greater accuracy and need more time to fine tune.

Thank you Eluvio team for this wonderful well-thought challange. I really enjoyed working on it.

References:
[Rao]: <https://arxiv.org/pdf/2004.02678.pdf>
