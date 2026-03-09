---
layout: post
title:  "Distributed Training of a Deep Recommender System"
date:   2026-03-09 18:50:11 +0530
categories: ml
---
Designing a recommender system is not a trivial problem to solve and big tech companies have invested hundreds of millions into building the best recommender systems platform. While I will not go through the details of designing a full recommender system in this post and I would like to keep that for a future post. In this post I would be walking through the steps I followed train a deep recommender system (a recommender system implemented using deep neural networks) in a distributed environment i.e. where we have a cluster of nodes/pods each with a limited memory and limited number of CPUs/GPUs.<br/><br/>
Please note that this was a weekly side project that I felt would be fun to do. Many of the codes or design that are shown here may not be fully optimized for production. Athough I will try to point out scopes for improvements wherever possible to the best of my knowledge. Also, I leveraged some of my team's reserved and not in use GPU kubernetes pods to run the model so to keep the post short I will not go through the setup of the distributed environment such as provisioning nodes in GCP or writing kebernetes YAML to spin up multiple GPU pods etc.<br/><br/>
The dataset used for training the deep recommender system is MovieLens-32M (~32 million movie ratings).<br/><br/>
The dataset comprises of multiple files corresponding to ratings (user id, movie id, rating, timestamp),  movies metadata such as title, genre etc. and movie tags. We will be using all of these data as features to train a regression model to predict the rating given the user and movie input features and any other features. To be more precise we are going to use the following features for the model.<br/><br/>
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
The 1st step would be to read the dataset files. Since the dataset size is approximately 1 GB, we can comfortably read the dataset into pandas dataframes and do the processing on top of pandas dataframes. Although it is highly recommended to use either Spark to read and process the dataset on low memory systems or use Polars instead of Pandas due to Polars being significantly faster than Pandas for data processing. We can write a simple Python function as below to read the dataset into dataframes as follows:<br/><br/>
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
    tags_column_names = ['userId', 'movieId', 'tag', 'timestamp']

    df_ratings = pd.read_csv(ratings_path, sep=',', names=rating_column_names, dtype={'userId':'int32', 'movieId':'int32', 'rating':float, 'timestamp':'int64'}, header=0)
    df_movies = pd.read_csv(movies_path, sep=',', names=movies_column_names, dtype={'movieId':'int32', 'title':'object', 'genres':'object'}, header=0)
    df_tags = pd.read_csv(tags_path, sep=',', names=tags_column_names, dtype={'userId':'int32', 'movieId':'int32', 'tag':'object', 'timestamp':'int64'}, header=0)

    df_ratings.dropna(inplace=True, subset=['userId', 'movieId', 'rating'])
    df_movies.dropna(inplace=True, subset=['movieId', 'title', 'genres'])
    df_tags.dropna(inplace=True, subset=['userId', 'movieId', 'tag'])
    df_tags.drop(columns=["userId","timestamp"], inplace=True)

    # Extract movie genres
    df_movies['genres'] = df_movies['genres'].apply(lambda x: x.lower().split('|'))

    df_movies['movie_year'] = df_movies['title'].str.extract(r'\((\d{4})\)').fillna("2025").astype('int')

    df_movies['title'] = df_movies['title'].str.replace(r'\((\d{4})\)', '', regex=True)
    df_movies['title'] = df_movies['title'].str.replace(r'[^a-zA-Z0-9\s]+', '', regex=True)
    df_movies['title'] = df_movies['title'].apply(lambda x: x.strip().lower().split(" "))
    df_movies['title'] = df_movies['title'].apply(lambda x: remove_stop(x))

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
```
<br/><br/>
The above function uses additional UDFs to preprocess data. The entire codes can be found [here](https://github.com/funktor/recsys/blob/main/data_generator.py).<br/><br/>
The next steps in data processing pipeline would be as follows:<br/><br/>
1. Normalize the ratings - We normalize each rating to N(0.0, 1.0) by the mean and standard deviation of all ratings given by the user id because each user has their own preference and rating standard thus it does not make sense to normalize using all the users.<br/><br/>
    ```python
    def normalize_ratings(df:pd.DataFrame):
        """
        Normalize ratings
        """
        df2 = df[["userId", "rating"]].groupby(by=["userId"]).agg(mean_user_rating=('rating', 'mean'), std_user_rating=('rating', 'std'))
        df = df.merge(df2, on=["userId"], how="inner")
        df["normalized_rating"] = (df["rating"] - df["mean_user_rating"])/df["std_user_rating"]
        df["normalized_rating"] = df["normalized_rating"].fillna(df["rating"])
        df.drop(columns=["rating"], inplace=True)
        return df
    ```
    <br/><br/>
2. Split the dataset of ratings into train, test and validation. We choose to do time based split by first sorting on timestamp. In this way we ensure that historical features such as previously rated movies and ratings do not leak from training into testing or validation. We chose to use 80% of the ratings for training and 20% for testing. Out of 80% in training 20% is used for validation after each epoch of training. The movies metadata dataset is not splitted.<br/><br/>
    ```python
    def split_train_test(df:pd.DataFrame, min_rated=10, test_ratio=0.8, val_ratio=0.8):
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
    <br/><br/>
3. Compute the vocabularies for the categorical features only on the training data. Apply the learnt vocabularies on the validation and testing datasets.<br/><br/>
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
    <br/><br/>
4. Compute the historical user features such as previously rated N movies and previous N ratings each for training, testing and validation datasets separately. With pandas computing the historical features were too much time consuming so I wrote the Cython modules for the same which improved the run time from 1 hour to 1.5 mins only.<br/><br/>
    ```python
    def get_historical_user_features_cpp(df:pd.DataFrame, max_hist=20):
        """
        Create historical sequential features of ratings
        """
        user_ids = df['userId'].to_numpy().astype(np.uint32)
        movie_ids = df['movieId'].to_numpy().astype(np.uint32)
        ratings = df['normalized_rating'].to_numpy().astype(np.float32)
        timestamps = df['timestamp'].to_numpy().astype(np.uint64)
    
        prev_movie_ids, prev_ratings  = ml_32m_py.py_get_historical_features(user_ids, movie_ids, timestamps, ratings, df.shape[0], max_hist)
    
        df["prev_movie_ids"] = prev_movie_ids
        df["prev_ratings"] = prev_ratings
    ```
    <br/><br/>
    The C++/Cython module for the `ml_32m_py.py_get_historical_features` can be found in github [here](https://github.com/funktor/recsys/blob/main/ml_32m_dp.cpp).<br/><br/>
5. Convert the pandas dataframes into parquet files as parquet format is quite generic and if you are going to use spark in the future, you do not need to change the model dataloader. Save the parquet files in the GCS buckets in the cloud.<br/><br/>
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
        df_ratings_train["partition"] = df_ratings_train.index % num_partitions
        df_ratings_val["partition"]   = df_ratings_val.index % num_partitions
        df_ratings_test["partition"]  = df_ratings_test.index % num_partitions
        df_ratings_full["partition"]  = df_ratings_full.index % num_partitions
    
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
    
