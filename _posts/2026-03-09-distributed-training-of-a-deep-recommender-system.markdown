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

## Data Generation
![Data Generator Pipeline](/docs/assets/data_gen.png)

### Step 1
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

### Step 2
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

### Step 3
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

### Step 4
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

### Step 5
Compute the historical user features such as the previously rated N movies and previous N ratings each for training, testing and validation datasets separately. With pandas, computing the historical features becomes too much time consuming task so I wrote the `Cython` modules for the same which improved the run time from `1 hour` to only `1.5 mins`.<br/><br/>
The C++/Cython module for the `ml_32m_py.py_get_historical_features` can be found in github [here](https://github.com/funktor/distributed-recsys/blob/main/ml_32m_dp.cpp).<br/><br/>
In order to build the Cython module, follow the steps shown below:<br/><br/>
```
pip install --upgrade Cython
python setup_ml_32m_gcp.py bdist_wheel
pip install --force-reinstall dist/*.whl
```
<br/><br/>

### Step 6
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
DCN v2 is a cross feature layer to handle feature-feature interactions. Here is a snippet of the code for computing the movie embeddings as shown above:<br/><br/>
```python
class MovieEncoder(nn.Module):
    def __init__(
            self, 
            movie_id_size, 
            movie_desc_size,
            movie_genres_size,
            movie_year_size, 
            embedding_size, 
            dropout=0.0
        ) -> None:
        
        super(MovieEncoder, self).__init__()
        
        self.movie_id_emb = MovieId(movie_id_size, embedding_size)
        self.movie_desc_emb = nn.Embedding(movie_desc_size, 256, padding_idx=0)
        self.movie_genres_emb = nn.Embedding(movie_genres_size, 8, padding_idx=0)
        self.movie_year_emb = nn.Embedding(movie_year_size, 16, padding_idx=0)

        self.fc_concat = nn.Linear(embedding_size + 280, embedding_size)
        init_weights(self.fc_concat)

        self.fc = nn.Sequential(
            self.fc_concat,
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_size)
        )

        self.cross_features = CrossFeatureLayer(embedding_size + 280, 3, 0.0)

    def forward(
            self, 
            ids:torch.Tensor, 
            descriptions:torch.Tensor, 
            genres:torch.Tensor, 
            years:torch.Tensor
        ):

        id_emb = self.movie_id_emb(ids) # (batch, embedding_size)
        desc_emb = emb_averaging(descriptions, self.movie_desc_emb) # (batch, 256)
        genres_emb = emb_averaging(genres, self.movie_genres_emb) # (batch, 8)
        years_emb = self.movie_year_emb(years) # (batch, 16)

        movie_embedding = torch.concat([id_emb, desc_emb, genres_emb, years_emb], dim=-1) # (batch, embedding_size + 280)
        movie_embedding = self.cross_features(movie_embedding) + movie_embedding # (batch, embedding_size + 280)
        movie_embedding = self.fc(movie_embedding) # (batch, embedding_size)

        return movie_embedding
```
The full code with all the components can be found at the [github repo](https://github.com/funktor/distributed-recsys/blob/main/model.py).<br/><br/>

### Complete Model
![Full Model](/docs/assets/model.png)
<br/><br/>
Recommender system class definition:
```python
class RecommenderSystem(nn.Module):
    def __init__(
            self, 
            user_id_size, 
            user_embedding_size, 
            user_prev_rated_seq_len, 
            user_num_encoder_layers, 
            user_num_heads, 
            user_dim_ff,
            user_dropout,
            movie_id_size, 
            movie_desc_size,
            movie_genres_size,
            movie_year_size, 
            movie_embedding_size, 
            movie_dropout,
            embedding_size,
            dropout=0.0
        ) -> None:

        super(RecommenderSystem, self).__init__()

        self.user_embedding_size = user_embedding_size
        self.movie_embedding_size = movie_embedding_size

        self.movie_encoder = \
            MovieEncoder\
            (
                movie_id_size, 
                movie_desc_size,
                movie_genres_size,
                movie_year_size, 
                movie_embedding_size, 
                movie_dropout
            )
        
        self.user_encoder = \
            UserEncoder\
            (
                user_id_size,
                user_embedding_size
            )
                
        self.movie_id_emb = self.movie_encoder.movie_id_emb

        self.user_prev_positional_encoding = \
            PositionalEncoding(
                movie_embedding_size, 
                user_prev_rated_seq_len, 
                user_dropout
            )
        
        self.user_prev_encoder_block = \
            Encoder(
                user_num_encoder_layers, 
                movie_embedding_size, 
                user_num_heads, 
                user_dim_ff, 
                user_dropout
            )

        self.user_prev_num_heads = user_num_heads

        self.fc_concat = nn.Linear(user_embedding_size + movie_embedding_size, embedding_size)
        init_weights(self.fc_concat)

        self.fc = nn.Sequential(
            self.fc_concat,
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_size)
        )

        self.cross_features = CrossFeatureLayer(embedding_size, 3, 0.0)

        self.fc_out = nn.Linear(embedding_size, 1)
        init_weights(self.fc_out)

        self.out = nn.Sequential(
            self.fc_out
        )

    def get_movie_embeddings(
            self, 
            movie_ids:torch.Tensor, # (batch,)
            movie_descriptions:torch.Tensor, # (batch, ntokens)
            movie_genres:torch.Tensor, # (batch, ntokens)
            movie_years:torch.Tensor # (batch,))
    ):
        
        return \
            self.movie_encoder\
                (
                    movie_ids, 
                    movie_descriptions, 
                    movie_genres, 
                    movie_years
                ) 
    
    def get_user_embeddings(
            self, 
            user_ids:torch.Tensor # (batch,)
    ):
        
        return \
            self.user_encoder\
                (
                    user_ids
                )  

    def forward(
            self, 
            user_ids:torch.Tensor, # (batch,)
            user_prev_rated_movie_ids:torch.Tensor, # (batch, seq)
            user_prev_ratings:torch.Tensor, # (batch, seq)
            movie_ids:torch.Tensor, # (batch,)
            movie_descriptions:torch.Tensor, # (batch, ntokens)
            movie_genres:torch.Tensor, # (batch, ntokens)
            movie_years:torch.Tensor # (batch,)
        ):
        
        movie_embeddings = \
            self.get_movie_embeddings\
                (
                    movie_ids,
                    movie_descriptions,
                    movie_genres,
                    movie_years
                )                                  # (batch, movie_embedding_size)
        
        user_embeddings = \
            self.get_user_embeddings\
                (
                    user_ids
                )                               # (batch, user_embedding_size)
        
        # mask the paddings from attention
        mask = (user_prev_rated_movie_ids != 0).float().unsqueeze(-1) # (batch, prev_rated_seq_len, 1)
        mask = torch.matmul(mask, mask.transpose(-2,-1)).unsqueeze(1).repeat(1, self.user_prev_num_heads, 1, 1) # (batch, num_heads, prev_rated_seq_len, prev_rated_seq_len)
        
        rated_movie_emb = self.movie_id_emb(user_prev_rated_movie_ids)   # (batch, prev_rated_seq_len, movie_embedding_size)
        rated_movie_emb = self.user_prev_positional_encoding(rated_movie_emb) # (batch, prev_rated_seq_len, movie_embedding_size)
        rated_movie_emb = self.user_prev_encoder_block(rated_movie_emb, mask) # (batch, prev_rated_seq_len, movie_embedding_size)

        rated_movie_ratings = user_prev_ratings.unsqueeze(1) # (batch, 1, prev_rated_seq_len)
        # weighted sum of ratings
        rated_movie_emb_weighted = torch.matmul(rated_movie_ratings, rated_movie_emb).squeeze(1) # (batch, movie_embedding_size)

        movie_embeddings = movie_embeddings + rated_movie_emb_weighted # (batch, movie_embedding_size)
        
        emb_concat = torch.concat([movie_embeddings, user_embeddings], dim=-1) # (batch, movie_embedding_size + user_embedding_size)
        
        emb  = self.fc_concat(emb_concat) # (batch, embedding_size)
        emb  = self.cross_features(emb) + emb  # (batch, embedding_size)
        out  = self.out(emb).squeeze(-1)  # (batch,)

        return out
```
<br/><br/>

## Trainer
Finally we come to the trainer part wherein we will explore distributed training using PyTorch. PyTorch provides multiple strategies for distributed training. The two most popular are `DDP` (Distributed Data Parallel) and `FSDP` (Fully Sharded Data Parallel). In both `DDP` and `FSDP`, the training data is partitioned across multiple workers across nodes and each worker works with only its data to compute the loss during forward passes and compute gradients in backward passes. The gradients are then averaged and broadcasted to all workers so that each worker now sees the same gradient values. In FSDP, the model is also partitioned across the workers. This is the case where the size of model is too large to fit in the memory of a single node or worker. But in FSDP communication overhead increases as compared to DDP because each worker also needs to coordinate with other workers in the forward passes too.<br/><br/>
In the example that I am working with, the model size is small enough to fit in the memory of a worker and thus I am going to use DDP. DDP uses multiple backend protocols for communication between workers such as `MPI`, `Gloo` and `NCCL`. For training on GPUs, almost always NCCL performs better than MPI or Gloo. We will deep dive each of these protocols and implement a custom MPI based distributed training in the next post.<br/><br/>
PyTorch provides 2 important tools for running a trainer script across multiple workers and/or multiple nodes - `torchrun` and `mpirun`.<br/><br/>
While torchrun is easy to work with as it does not require installing `OpenMPI` libraries or overhead of enabling ssh and tcp communication between the workers as in mpirun but in order to use torchrun, one needs to login to all the nodes/pods individually and run the script in each node individually. This might not be an ideal situation when we have to deploy the trainer in production and run the training jobs using a job scheduler such as `Airflow`. That is why I prefer to use `mpirun` instead of `torchrun` for this example.<br/><br/>
In both torchrun and mpirun, environment variables are set for individual workers and nodes. The most important in mpirun are the following:<br/><br/>
```
OMPI_COMM_WORLD_SIZE - World size i.e. total number of workers across all nodes.
For e.g. if there are 2 nodes each with 8 GPU workers, then OMPI_COMM_WORLD_SIZE=16.
For torchrun it is WORLD_SIZE.

OMPI_COMM_WORLD_LOCAL_RANK - Local rank of a worker.
For e.g. if there are 2 nodes each with 8 GPU workers, then OMPI_COMM_WORLD_LOCAL_RANK ranges from 0-7 in node 0 and 0-7 in node 1.
For torchrun it is LOCAL_RANK.

OMPI_COMM_WORLD_RANK - Global rank of a worker.
For e.g. if there are 2 nodes each with 8 GPU workers, then OMPI_COMM_WORLD_RANK ranges from 0-7 in node 0 and 8-15 in node 1
For torchrun it is RANK.
```
<br/><br/>
In the example I am working with I tweaked the DDP training strategy a bit. In DDP each worker initially has a view of all the training data and then at the start of each epoch, data is sharded across all the workers and nodes. On the other hand I am tweaking this a bit so that each worker only downloads an equal sized shard from GCP bucket and works with only the same set of training data for all epochs. There are some advantages and disadvantages with this approach over vanilla DDP.
<br/><br/>

### Advantage
The workers does not spend time in sharding the data before the start of each epoch and thus training is usually faster. Also since each worker downloads only a subset of the training data, memory requirement is lower as compared to vanilla DDP where each worker works with all of the data and thus memory requirement is higher.

### Disadvantage
The vanilla DDP is useful when we have to shuffle the data after each epoch and thus in each epoch, each worker "sees" a different subset of data from the other epochs. Thus each worker works gets the chance to train with all the training data. But in the altered method, each worker sees the same subset of data in all epochs. With large datasets this should not make a lot of difference if we pre-shuffle the data once before epoch 0.

### Setup DDP
The first step is to initialize DDP before the start of training.
```python
def ddp_setup(rank_local, rank_global, world_size):
    """
    Setup DDP
    """
    init_process_group(backend="nccl", world_size=world_size, rank=rank_global)
    torch.cuda.set_device(rank_local)

rank_local  = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]) if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ else int(os.environ["LOCAL_RANK"])
rank_global = int(os.environ["OMPI_COMM_WORLD_RANK"]) if "OMPI_COMM_WORLD_RANK" in os.environ else int(os.environ["RANK"])
world_size  = int(os.environ["OMPI_COMM_WORLD_SIZE"]) if "OMPI_COMM_WORLD_SIZE" in os.environ else int(os.environ["WORLD_SIZE"])

print("Setting up DDP...")
if dist.is_initialized() is False:
    ddp_setup(rank_local, rank_global, world_size)
```
<br/><br/>
Since I am using GPU to train thus the backed "nccl" is the preferred protocol.

### Get Datasets
The next step is to for each worker GPU, download an "equal" sized shard from GCP. Note that during data generation, I partitioned the parquet dataset into 32 partitions. Thus if there are 8 GPU workers, each GPU downloads approximately 4 partitions of the parquet dataset. Also note that if the total number of records across the 32 partitions is not divisible evenly by 32, then all partitions may not have the same number of records. This could potentially lead to uneven number of batches which could in turn lead to stalling of GPU which we will see later how to overcome. For now assume that the dataset number of records is divisible by 32.
```python
def get_dataset(path:str, cache_dir:str, world_size:int, rank:int, in_memory:bool=False, path_is_dir:bool=True):
    """
    Get dataset by prepartitioning files to each worker process
    """
    if path_is_dir:
        files = pre_partitions_for_download(path, world_size, rank)
    else:
        files = path

    os.makedirs(cache_dir, exist_ok=True)

    dataset = datasets.load_dataset("parquet", data_files=files, split="train", cache_dir=cache_dir, keep_in_memory=in_memory)
    return dataset


def get_datasets(path:str, world_size:int, rank:int):
    """
    Get train, validation and movies datasets
    """
    ratings_train = \
        get_dataset(
            f"{path}/train", 
            f"/tmp/huggingface/{rank}/train", 
            world_size, 
            rank
        )
    ratings_train.set_format('pandas')
    
    ratings_val = \
        get_dataset(
            f"{path}/validation", 
            f"/tmp/huggingface/{rank}/val", 
            world_size, 
            rank
        )
    ratings_val.set_format('pandas')
    
    movies_dataset = \
        get_dataset(
            f"{path}/movies.parquet", 
            f"/tmp/huggingface/{rank}/movies", 
            world_size, 
            rank, 
            in_memory=True, 
            path_is_dir=False
        ).to_pandas()

    return ratings_train, ratings_val, movies_dataset

print("Getting datasets...")
ratings_train, ratings_val, movies_dataset = dataloader.get_datasets(datasets_gcs_path, world_size, rank_global)
```
    
