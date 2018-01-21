# Online User Trajectory Clustering
This project is a continuation of the text classification task described in the Text Classification of Online Community Posts repository. The aim of this project is to identify and analyze patterns in the user trajectories. A 'trajectory' in our case is a temporal representation of the activity of a user per month, from the time of his joining until the time he leaves the OHC. It is a 5-element vector which records the number of posts of each kind([COM, PES, PIS, SS, N/A]) per month. 

The code associated with this repository only builds the user trajectories which are then clustered using a Partitioning Around Medoids(PAM) algorithm, which can be found in the link below and modified the distance metric to Directed Hausdorff:

https://github.com/daveti/pycluster

