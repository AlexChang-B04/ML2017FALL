#ML2017FALL final project

##Conversations in TV shows

###Package Requirement

* gensim (3.2.0)
* jieba (0.39)
* numpy (1.14.0)
* scipy (1.0.0)

###Execution Command

Only prediction:
```sh
bash python3 final_reproduce.sh <training_data folder> <testing_data path> <prediction_file path>
```
Including preprocessing, word2vec training, prediction:
```sh
bash python3 final.sh <training_data folder> <testing_data path> <prediction file path>
```
