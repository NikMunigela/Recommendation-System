# Recommendation-System #
A recommender system that suggests relevant movies to the user based on previously rated and viewed movies, and on the movies that can be classified under similar features and genres using various Information Retrieval techniques based on dimensionality reduction and feature extraction.<br/>
Recommender systems are an integral part of various product and movie streaming services to recommend corresponding commodities to the user based on their interests.

## Functionality ##
The program works by implementing and comparing techniques such as SVD (Singular Value Decomposition), CUR Decomposition, Collaborative filtering and Latent Factor models for building the recommender systems. Further, the systems are evaluated using RMSE(Root Mean Square Error) and MAE(Mean Average Error) error measures to ascertain and compare efficiencies these recommendation techniques. Time taken to employ the different  recommendation systems has also been calculated and tabulated.

## Work Flow ##
* Reads the dataset containing the ratings of all movies given by the users.
* Generate the User-Movie matrix
* Design Algorithms for Collaborative filtering, SVD, CUR and Latent Factor models
* Pass the User-Movie matrix to all the above algorithms.
* Generate the predicted User_Movie matrix for each model
* Pass original and predicted matrices to Error measures such as RMSE and MAE
* Accuracy of the models is computed
* Results are tabulated along with the time taken for model development.

## Team ##
* *Nikhil Munigela*
* *Jathin Badam*
* *Karthik Jagini*
* *Nikhil Kandukuri*
