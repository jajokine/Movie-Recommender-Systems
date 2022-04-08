<h1 align='center'>Recommender Systems</h1>
<h2 align='center'>with Neural Collaborative Filtering and Matrix Factorization</h2>

<p align='center'>
      <img
           width='300'
           height='300'
           src='https://github.com/jajokine/Recommender-Systems/blob/main/movies.png'
      >
</p>

A Recommender System is a process that seeks to predict or filter user preferences according to the user's choices that are based on explicit (i.e. direct user interactions) and/or implicit feedback (i.e. indirect user interactions) from the user in order to provide recommendations what the user should do next. These choices are usually analyzed through methods such as clustering, nearest neighbor or matrix factorization, and the systems are widely being used in search queries in general, as well as various online products and services such as videos, movies, music, news, and books.

The main idea is that we have a large database of users, but only a few of these people have actually given any explicit feedback from their interactions. This creates a database that is very sparse with insightful information and the key is to find a way to fill the missing entries either through feature engineering where we transform the explicit interactions into implicit ones, and hence create a different problem that is easier to solve or we try to fill the empty entries with different kinds of predictive methods.  

We can break down the methods roughly into three categories where in the first, the recommendation is based solely on the features of items being used by the user which leads to the promotion of other items with similar features. This is the item-based filtering which relies on the descriptions of the item and the profile of the user's past preferences. Item-based recommenders are faster to deploy than user-based when the dataset is large. 

The second system is based on Collaborative Filtering (CF) which uses a combination of your past behavior and the experiences of other people in making the recommendation through an item or user based recommender system. These don't ncessary require features about the items or users to be known if the options available are diverse enough.  In the item-based, other users who browsed similar items will be recommended, and on the user based, items that similar users have browsed will be recommended. Both methods demand eventually a large pool of users from which the recommendation can be made, however data sparsity can affect the quality of user-based recommenders.

The third category which is the one that is being used today, uses a combination of the previous systems, for example through low-rank matrix factorization that decomposes large matrices into compressed representation of the data with Singular Value Decomposition (SVD) or with deep learning systems through learned embeddings. The benefits of using the latest hybrid methods is that they are easier to scale to larger data sets and can better derive both tastes and preferences from the user patterns than the previous two, which rely heavily on the distances between the specific data points that are usually quite sparsely represented in large datasets, and hence, can lead to overfitting or just noisy representations of user tastes and preferences as there is just not enough information available. Once combined though, the methods can provide recommendations for items that the user hasn't seen or thought of before.

In this notebook I will explore different hybrid methods and compare the recommendations made from the different approaches. For the Deep Learning models, I will be using PyTorch Lightning which is a recent PyTorch library that enables to scale training on multiple GPUs with no code changes and offers precoded boilerplates to speedup modeling for production. This makes the implementation of models quite easy and fun to do.

The first method uses a Neural Colloborative Filtering (NCF) that captures the implicit feedback from positively and negatively labeled interactions through user and item embeddings inorder to make predictions what movies the user would like to see. this will be followed up by a model that
uses the same idea of embeddings, but this time the learning will happen through the explicit feedback from the ratings. We will then make some movie recommendations and compare this to a more complicated model that uses fully connected layers together with dropout layers that again learn from the embeddings but this time on its own without any added calculations. The final model became famous through the Netflix prize competition and uses a completely different approach, namely a statistical technique that calculates predictions rather simply but very effectively from the entire dataset through matrix factorization with SVD.

## Dataset

The dataset comes from MovieLens project collected by GroupLens Research at the University of Minnesota. GroupLens have gathered datasets of movie ratings of various sizes ranging from 100,000 to 20,000,000. In order to capture more recent films, I will be using a subset from the largest dataset that has 27,278 movies which were collected between 1995-2015 and published in 2016 (Research paper: http://files.grouplens.org/papers/harper-tiis2015.pdf).

## Access and Requirements

The file recommender.ipynb is the Jupyter Notebook that contains all the code and analysis of the project.

The dependencies and requirements can be seen from requirements.txt that can be installed in shell with the command:

      pip install -r requirements.txt
