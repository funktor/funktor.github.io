---
layout: post
title:  "Distributed Training of a Deep Recommender System"
date:   2026-03-09 18:50:11 +0530
categories: ml
---
Designing a recommender system is not a trivial problem to solve and big tech companies have invested hundreds of millions into building the best recommender systems platform. While I will not go through the details of designing a full recommender system in this post and I would like to keep that for a future post. In this post I would be walking through the steps I followed to train a deep recommender system (a recommender system implemented using deep neural networks) in a distributed environment i.e. where we have a cluster of nodes/pods each with a limited memory and limited number of CPUs/GPUs.<br/><br/>
Please note that this was a weekly side project that I felt would be fun to do. Many of the codes or design that are shown here may not be fully optimized for production. Athough I will try to point out scopes for improvements wherever possible to the best of my knowledge. Also, I leveraged some of my team's reserved and not in use GPU pods to run the model so to keep the post short I will not go through the setup of the distributed environment such as provisioning nodes in GCP or writing kebernetes YAML to spin up multiple GPU pods etc.<br/><br/>
The dataset used for training the deep recommender system is `MovieLens-32M` (~32 million movie ratings).<br/><br/>
The dataset comprises of multiple files corresponding to ratings (user id, movie id, rating, timestamp),  movies metadata such as title, genre etc. and movie tags. I will be using all of these data as features to train a `regression model` to predict the rating given the user and movie input features and any other features. To be more precise I am going to use the following features for the model.<br/><br/>
```
User Features
1. user_id
2. previously rated N movie_ids
3. previous N movie ratings

Movie Features
1. movie_id
2. genres
3. description + tags metadata
4. year of release

Ratings - Normalized Rating corresponding to user_id and movie_id
```
<br/><br/>

![Data Generator Pipeline](/docs/assets/data_gen.png)
## Step 1
The 1st step would be to read the dataset files. Since the dataset size is approximately `1 GB`, I can comfortably read the dataset into `Pandas` dataframes and do the processing on top of pandas. Although it is highly recommended to use either `Spark` to read and process the dataset on low memory systems or use `Polars` instead of Pandas due to Polars being significantly faster than Pandas for data processing. I wrote this simple Python function to read the dataset into dataframes as follows:<br/><br/>
```python
def get_ml_32m_dataframe(path:str):
    """
    Read dataset file and create pandas dataframes
    """
    ratings_path = os.path.join(path, 'ratings.csv')
    movies_path = os.path.join(path, 'movies.csv')
    tags_path = os.path.join(path, 'tags.csv')

    rating_column_names = ['userId', 'movieId', 'rating', 'timestamp']
    movies_column_names = ['movieId', 'title', 'genres']
    tags_column_names   = ['userId', 'movieId', 'tag', 'timestamp']

    df_ratings = \
        pd.read_csv(
            ratings_path,
            sep=',',
            names=rating_column_names,
            dtype={
                'userId':'int32',
                'movieId':'int32',
                'rating':float,
                'timestamp':'int64'
            },
            header=0
        )

    df_movies = \
        pd.read_csv(
            movies_path,
            sep=',',
            names=movies_column_names,
            dtype={
                'movieId':'int32',
                'title':'object',
                'genres':'object'
            },
            header=0)

    df_tags = \
        pd.read_csv(
            tags_path,
            sep=',',
            names=tags_column_names,
            dtype={
                'userId':'int32',
                'movieId':'int32',
                'tag':'object',
                'timestamp':'int64'
            },
            header=0
        )

    df_ratings.dropna(inplace=True, subset=['userId', 'movieId', 'rating'])
    df_movies.dropna(inplace=True, subset=['movieId', 'title', 'genres'])
    df_tags.dropna(inplace=True, subset=['userId', 'movieId', 'tag'])
    df_tags.drop(columns=["userId","timestamp"], inplace=True)

    # Extract movie genres
    df_movies['genres'] = df_movies['genres'].apply(lambda x: x.lower().split('|'))

    # Extract movie year
    df_movies['movie_year'] = \
        df_movies['title'].str.extract(r'\((\d{4})\)').fillna("2025").astype('int')

    # Extract movie title, replace non alpha-numeric characters with blanks,
    # lowercase and remove stopwords

    df_movies['title'] = df_movies['title'].str.replace(r'\((\d{4})\)', '', regex=True)
    df_movies['title'] = df_movies['title'].str.replace(r'[^a-zA-Z0-9\s]+', '', regex=True)
    df_movies['title'] = df_movies['title'].apply(lambda x: x.strip().lower().split(" "))
    df_movies['title'] = df_movies['title'].apply(lambda x: remove_stop(x))

    # Extract movie tags, replace non alpha-numeric characters with blanks,
    # lowercase and remove stopwords. Group tags per movie.

    df_tags['tag'] = df_tags['tag'].str.replace(r'[^a-zA-Z0-9\s]+', '', regex=True)
    df_tags['tag'] = df_tags['tag'].apply(lambda x: x.strip().lower())
    df_tags = df_tags.groupby("movieId").agg(set).reset_index()
    df_tags['tag'] = df_tags['tag'].apply(list)
    df_tags['tag'] = df_tags['tag'].apply(lambda x: flatten_lists(x))
    df_tags['tag'] = df_tags['tag'].apply(lambda x: remove_stop(x))
    df_tags['tag'] = df_tags['tag'].astype("object")

    df_movies = df_movies.merge(df_tags, on=['movieId'], how='left')
    df_movies["tag"] = df_movies["tag"].fillna({i: [""] for i in df_movies.index})
    df_movies["description"] = df_movies["title"] + df_movies["tag"]
    df_movies.drop(columns=["tag"], inplace=True)
    df_movies.drop(columns=["title"], inplace=True)

    return df_ratings, df_movies

print("Reading datasets from path...")
df_ratings, df_movies = get_ml_32m_dataframe(dataset_path)
```
<br/><br/>
The above function uses certain UDFs to preprocess data. The columns I am interested in are : `[user_id, movie_id, rating, timestamp, description, genres, movie_year]`.<br/><br/>
The column `description` is a derived column from title and tags. I am assuming a 1-gram language model and extracting the words as tokens. The entire codes can be found [here](https://github.com/funktor/distributed-recsys/blob/main/data_generator.py).
<br/><br/>

## Step 2
Normalize the ratings - Normalize each rating in `N(0, 1)` by the mean and standard deviation of all ratings given by the user id because each user has their own preference and rating standard thus it does not make sense to normalize using all the users. One can also build a model with the mean and standard deviation as learnable parameters using the `negative log likelihood` of `normal distribution` as the loss function. It would usually make more sense to do that.<br/><br/>
```python
def normalize_ratings(df:pd.DataFrame):
    """
    Normalize ratings
    """
    df2 = \
        df[["userId", "rating"]]\
            .groupby(by=["userId"])\
            .agg(
                mean_user_rating=('rating', 'mean'),
                std_user_rating=('rating', 'std')
            )
    df = df.merge(df2, on=["userId"], how="inner")
    df["normalized_rating"] = (df["rating"] - df["mean_user_rating"])/df["std_user_rating"]
    df["normalized_rating"] = df["normalized_rating"].fillna(df["rating"])
    df.drop(columns=["rating"], inplace=True)
    return df
```
<br/><br/>

## Step 3
Split the dataset of ratings into train, test and validation. I choose to do a `time based split` by first sorting on timestamp. In this way I can ensure that historical features such as previously rated movies and ratings do not leak from training into testing or validation. I chose to use `80%` of the ratings for training and `20%` for testing. Out of 80% in training 20% is used for validation after each epoch of training. The movies metadata dataset is not splitted as it is used to join with the ratings dataset in train, test and validation.<br/><br/>
```python
def split_train_test(
    df:pd.DataFrame,
    min_rated=10,
    test_ratio=0.8,
    val_ratio=0.8
):
    """
    Split dataset into train, test and validation
    """
    print("Splitting data into train test and validation...")
    # Split data into training, testing and validation
    df = df.sort_values(by='timestamp')
    df2 = df[["userId", "movieId"]].groupby(by=["userId"]).agg(list).reset_index()

    # Filter all user_ids who have rated more than 'min_rated' movies
    df2 = df2[df2.movieId.apply(len) > min_rated]
    df = df.merge(df2, on=["userId"], how="inner", suffixes=("", "_right"))
    df.drop(columns=['movieId_right'], inplace=True)

    n = df.shape[0]
    m = int(test_ratio*n)

    df_train_val = df[:m]
    df_test = df[m:]

    k = int(val_ratio*m)
    df_train = df_train_val[:k]
    df_val = df_train_val[k:]

    return df_train, df_val, df_test
```
Before splitting, I am filtering ratings data by users who have given at-least min-rated number of ratings so as to reduce noise in the training dataset due to long tail users with 1 or 2 ratings only.
<br/><br/>

## Step 4
Compute the vocabularies for the categorical features only on the training data. Apply the learnt vocabularies on the validation and testing datasets.<br/><br/>
```python
def fit_vocabulary(df_ratings:pd.DataFrame, df_movies:pd.DataFrame):
    """
    Fit vocabulary
    """
    vocabulary = {}
    max_vocab_size = {'userId':1e100, 'movieId':1e100, 'description':1e5, 'genres':100, 'movie_year':1e100}

    for col in ['userId', 'movieId']:
        print(col)
        df_ratings[col], v = categorical_encoding(df_ratings, col, max_vocab_size[col])
        vocabulary[col] = v

    for col in ['description', 'genres', 'movie_year']:
        print(col)
        df_movies[col], v = categorical_encoding(df_movies, col, max_vocab_size[col])
        vocabulary[col] = v

    for col in ['movieId']:
        print(col)
        df_movies[col] = df_movies[col].apply(lambda x: transform(x, vocabulary[col]))
    
    return vocabulary, df_ratings, df_movies

def score_vocabulary(df_ratings:pd.DataFrame, vocabulary:dict):
    """
    Score vocabulary
    """
    df_ratings = df_ratings.reset_index()
    for col in ['userId', 'movieId']:
        print(col)
        df_ratings[col] = df_ratings[col].apply(lambda x: transform(x, vocabulary[col]))
    
    return df_ratings
```
To limit the vocabulary size, I am using a frequency based criteria wherein I keep the top N values per feature based on frequency of occurrence. This is useful for categorical features with millions or         billions of categories such as language models. Again the entire code for vocabulary fitting and scoring can be found [here](https://github.com/funktor/distributed-recsys/blob/main/data_generator.py).
<br/><br/>

## Step 5
Compute the historical user features such as the previously rated N movies and previous N ratings each for training, testing and validation datasets separately. With pandas, computing the historical features becomes too much time consuming task so I wrote the `Cython` modules for the same which improved the run time from `1 hour` to only `1.5 mins`.<br/><br/>
The C++/Cython module for the `ml_32m_py.py_get_historical_features` can be found in github [here](https://github.com/funktor/distributed-recsys/blob/main/ml_32m_dp.cpp).<br/><br/>
In order to build the Cython module, follow the steps shown below:<br/><br/>
```
pip install --upgrade Cython
python setup_ml_32m_gcp.py bdist_wheel
pip install --force-reinstall dist/*.whl
```
<br/><br/>

## Step 6
Convert the pandas dataframes into `parquet` files as parquet format is quite generic and efficient for column based feature datasets and if you are going to use spark in the future, you do not need to change the model dataloader. Save the parquet files in the `GCS buckets` in the cloud.<br/><br/>
```python
def save_dfs_parquet(
    out_dir:str, 
    vocabulary:dict, 
    df_ratings_train:pd.DataFrame, 
    df_ratings_val:pd.DataFrame, 
    df_ratings_test:pd.DataFrame, 
    df_ratings_full:pd.DataFrame, 
    df_movies:pd.DataFrame, 
    num_partitions:int=32
):
    """
    Save dataframe into parquet files
    """
    # Partition the training data so as we can read only a subset of partitions
    # if required for debugging etc.

    df_ratings_train["partition"] = df_ratings_train.index % num_partitions
    df_ratings_val["partition"]   = df_ratings_val.index % num_partitions
    df_ratings_test["partition"]  = df_ratings_test.index % num_partitions
    df_ratings_full["partition"]  = df_ratings_full.index % num_partitions

    # shuffle the datasets so that user ids are randomly distributed across rows

    df_ratings_train = df_ratings_train.sample(frac=1).reset_index()
    df_ratings_val   = df_ratings_val.sample(frac=1).reset_index()
    df_ratings_test  = df_ratings_test.sample(frac=1).reset_index()
    df_ratings_full  = df_ratings_full.sample(frac=1).reset_index()

    if os.path.exists(out_dir):
        try:
            shutil.rmtree(out_dir)
        except:
            pass

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(vocabulary, f"{out_dir}/vocabulary.pkl")
    df_ratings_train.to_parquet(out_dir + "/train/", partition_cols=["partition"])
    df_ratings_val.to_parquet(out_dir + "/validation/", partition_cols=["partition"])
    df_ratings_test.to_parquet(out_dir + "/test/", partition_cols=["partition"])
    df_ratings_full.to_parquet(out_dir + "/full_data/", partition_cols=["partition"])
    df_movies.to_parquet(out_dir + "/movies.parquet")
```
<br/><br/>

## Model Architecture
As mentioned earlier that I am solving this recommender system problem as a `regression` over the `normalized ratings`. Regression is not the only way to solve. One can also solve this as a `binary classification` problem by turning ratings and views into binary labels for e.g. with normalized ratings, one can assign a label of 1 for all ratings >= 0 and a label of 0 to all ratings < 0 since normalized ratings has a mean of 0. There are few challenges I saw when implementing a binary classification approach to a recommender system:<br/><br/>

### Challenge 1
Defining positive and negative examples correctly. One strategy is as mentioned above where we consider all ratings above the mean for that user as positives and rest as negatives. But this is only possible  where explicit ratings are available.

### Challenge 2
Even with explicit ratings available, should I consider the unrated movies as negatives too because the user might have chose to ignore watching them or worse decided to not rate them at all.

### Challenge 3
If ratings are missing, one way to implicitly label is to assign 1 to watched movies and 0 to not watched movies. But this could lead to severe class imbalance issues as number of unwatched movies far exceeds watched movies.

### Challenge 4
Assigning 0 or negative to unwatched movies could lead to bias in training data because if someone has not watched a movie and you train them as negatives, the model will learn to rank them and similar movies lower thus reducing the diversity in recommendations.
<br/><br/>

### Movie Features
![Movie Embeddings](/docs/assets/movie_emb.png)
<br/><br/>

### Complete Model
![Full Model](/docs/assets/model.png)
<br/><br/>


Onwards with our model architecture. The following architecture is quite simplistic compared to some of the latest developments around deep recommender systems. I am planning to upgrade the following architecture into a generative recommender system in the future.<br/><br/>
1. On a high level, the model comprises of two towers, one for user features and another for movie features. The outputs from 2 towers are concatenated and passed through a MLP layer with a single output i.e. the predicted rating for the user and movie.<br/><br/>
2. The movie tower accepts the movie features (as mentioned earlier) and passes them through Embedding Layers to convert the one-hot features into embeddings. There are some features like the genres and description/tags which is multi-hot i.e. they have multiple values. For these features, I am just computing average embedding after masking the 0s. Finally all movie features are concatenated and passed through a MLP layer with a non-linear activation function such as GeLU, Dropout and LayerNorm layers. Finally applying a cross feature layer using DCN on top of the output.<br/><br/>
3. Similar to movie tower, the user tower converts the one-hot features into embeddings first. The historical features are considered as sequential features and they are handled using a Transformer Layer where the previous N rated movies are passed through a Transformer layer and then I take the dot product of the output of Transformer with the previous N ratings corresponding to these movies.<br/><br/>
   
One definite challenge with a binary classification problem is class imbalance because 

    
