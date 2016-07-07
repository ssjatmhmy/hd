# My solution for Kaggle 'Home Depot Product Search Relevance' competition

This is my solution for the 'Home Depot Product Search Relevance' competition on Kaggle, which scored 0.45766 (76/2125) on Public Leaderboard and 0.45751 (83/2125) on Private Leaderboard. 

Kaggle ID : http://www.kaggle.com/ssjatmhmy

## Generating the solution

step 1  
Run the following command to generate word2vec training data 'data.text'.  
cd code  
python3 gen_text.py  
Then copy the generated file 'data.text' into the 'tools-w2v' folder.  

step 2  
Run the following command to generate word2vec model, and then re-enter the 'code' folder:  
cd tools-w2v  
python3 train\_word2vec\_model.py  
cd ../code  

step 3  
Run the following command to generate cleaned data:  
python3 run.py -PrepareData  

step 4  
Run the following command to generate tsne features. Note that it has to be run with python2, because the lib we used does not support python3.  
python2 python2_tsne.py  

step 5  
Run the following command to generate all features:   
python3 run.py -GenerateFeatures  

step 6  
Run the following command to generate submission:  
python3 run.py -GetSubmission  
