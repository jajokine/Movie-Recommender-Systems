# Recommender-Systems

![alt text](https://github.com/jajokine/Recommender-Systems/blob/main/movies.png)

A Recommender System is a process that seeks to predict or filter user preferences according to the user's choices that are based on explicit (i.e. direct user interactions) and/or implicit feedback (i.e. indirect user interactions) from the user in order to provide recommendations what the user should do next. These choices are usually analyzed through methods such as clustering, nearest neighbor or matrix factorization, and the systems are widely being used in search queries in general, as well as in various online products and services such as videos, movies, music, news, and books.

We can break down the methods roughly into three categories where in the first, the recommendation is based solely on the features of items being used by the user which leads to the promotion of other items with similar features. This is the content-based filtering which relies on the descriptions of the item and the profile of the user's past preferences. 

The second system is based on Collaborative Filtering (CF) which uses a combination of your past behavior and the experiences of other people in making the recommendation through an item or user based recommender system.  In the item based, other users who browsed similar items will be recommended, and on the user based, items that similar users have browsed will be recommended. Both methods demand eventually a large pool of users from which the recommendation can be made.

The third category which is the one that is being used today, uses a combination of the previous systems, for example through low-rank matrix factorization that decomposes large matrices into compressed representation of the data with Singular Value Decomposition (SVD) or with deep learning systems through learned embeddings.

The benefits of using the latest hybrid methods is that they are easier to scale to larger data sets and can better derive both tastes and preferences from the user patterns than the previous two. which rely heavily on the distances between the specific data points that are usually quite sparsely represented in large datasets, and hence, can lead to overfitting or just noisy representations of user tastes and preferences as there is just not enough information available, especially explicit feedback or that it can be difficult to distinguish positive implicit feedback from negative ones.

In this notebook I will explore three hybrid methods. The first method uses a Neural Colloborative Filtering (NCF) model with PyTorch Lightning that captures the implicit feedback from positively and negatively labeled interactions inorder to make predictions what movies the user would like to see. 

The second model follows the same idea with a bit more complicated architecture that is based on explicit feedback from the ratings. We will then make some movie recommendations and compare these to a third model that uses a completely different approach through matrix factorization with SVD so it will be interesting to see how all these approaches perform.

## Dataset

The dataset comes from MovieLens project collected by GroupLens Research at the University of Minnesota. GroupLens have gathered datasets of movie ratings of various sizes ranging from 100,000 to 20,000,000. I will be using a subset from the largest dataset that has 27,278 movies which were collected between 1995-2015 and published in 2016 (Research paper: http://files.grouplens.org/papers/harper-tiis2015.pdf).

## Access and Requirements

The file recommender.ipynb is the Jupyter Notebook that contains all the code and analysis of the project.

The dependencies and requirements can be seen from requirements.txt that can be installed in shell with the command:

      pip install -r requirements.txt
