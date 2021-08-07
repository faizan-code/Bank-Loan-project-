Bank Loan Approval Prediction and Analysis Using Random Forest Regressor Algorithm

Abstract
A loan is the financial sector's primary source of income as well as its primary source of financial risk. Large amounts of a bank's assets are directly derived from interest received on loans made. Every year, the number of persons or organizations requesting for loans grows across the world. All of the organizations are attempting to produce efficient business approaches in order to encourage more clients to apply for loans. Nearly every day, a huge number of people apply for loans for a variety of reasons. However, all of these candidates are untrustworthy, and no one can be approved. Every year, we learn about a number of situations in which people do not repay the majority of their loans to banks, causing them to incur significant losses. The primary goal of this paper is to analyze whether or not a loan granted to a certain person or organization should be accepted. The Random Forest Regressor model has been run, and various performance measurements have been derived. The suitable consumers to target for loan granting may be easily identified by analyzing their chance of loan default using Random Forest Regressor. The model concludes that a bank should not just target affluent clients for loan approval, but it should also analyze other customer qualities that play a significant role in credit granting choices and forecasting loan defaulters. 
Key words: Bank loan approval analysis, financial risk, Loan prediction, Random forest regressor, Machine learning

1.	Introduction

Banking is a regulated industry in most countries since it is an essential role in determining a country's financial stability. The disbursement of loans is nearly every bank's primary activity. The majority of the bank's assets are directly derived from the profits made on loans provided by the bank. The primary objective of the banking industry is to put their funds in secure hands with a low risk of failure. Presently, many banks and financial institutions authorize loans after a difficult, long, and exhausting verification procedure, but there is no guarantee that the chosen applicant is trustworthy, or that he will be able to repay the loan with interest within the specified time frame.
In the current scenario, a loan must be manually sanctioned by a bank manager, which means that individual will be responsible for determining if the individual is eligible for the loan and assessing the risk inherent with it. Because it is done by a human, it is a time-consuming and error-prone procedure. If the loan is not returned, the bank suffers a loss, because banks make the majority of their earnings from interest payments. If banks make a loss, it will lead to a banking collapse. These financial crises have an impact on the country's economy. As a result, it is critical that the loan be authorized with the least amount of mistake in risk calculation and in the shortest period of time feasible.
In response to the above scenario, the aim of this paper is to discuss the application of various machine learning models in the loan lending process and determine the best way for a financial institution that accurately identifies whom to lend to and assists banks in identifying loan defaulters for much-reduced credit risk. Classifiers that we used to build the model are Random Forest Regressor.
Loan prediction is a frequent real-world issue that every financial business encounters in their lending operations. If the loan approval process is automated, it may save a lot of time and increase the pace with which clients are served. Customer satisfaction has increased significantly, and operating expenses have been reduced significantly. However, the advantages can only be realized if the bank has a reliable model in place to properly forecast which loans should be approved and which should be rejected in order to reduce the risk of loan default. The purpose of this Paper is to give a rapid, immediate, and simple method for selecting meritorious applicants. It may offer the bank with particular benefits. 
First, we performed a bank loan acceptance analysis and a visual analysis of the elements that influence, enhance, or decrease a person's chances of obtaining a bank loan. Then we created a classification ML model that can predict whether or not a loan would be granted. It is a binary classification problem in which we must forecast either of the two classes given, i.e., granted 1 or not granted 0.

2.	Literature Survey 

This section provides a summary of some of the previous work on developing machine learning and deep learning models using different algorithm to enhance the loan prediction process and assist banking authorities and financial firms in selecting a qualified applicant with extremely low credit risk.

The author used several machine learning algorithms on the dataset to discover which methods are most suited for analyzing bank credit datasets. Apart from the Gaussian Naive Bayes and Nearest Centroid, the study indicated that the rest of the algorithms perform fairly well in terms of accuracy and other performance assessment criteria. Each of these algorithms had an accuracy percentage ranging from 76% to more than 80%. They also identified the most critical factors that impact a customer's trustworthiness. They developed a predictive model for estimating consumer trustworthiness using linear regression and the most significant characteristics [1]. The author utilized a logistic regression technique; the suitable clients to target for loan giving may be easily identified by analyzing their risk of loan default. The model suggests that a bank not only should target affluent clients for loan granting, but it should also analyze other consumer qualities that play a significant role in credit granting choices and forecasting loan defaulters [2].  Pidikiti Supriya et al. [3] implemented their model using decision trees as a machine learning technique. They began their investigation with data cleaning and pre-processing, followed by missing value imputation, information extraction, and ultimately model construction and assessment. On a public test set, the authors achieved the highest accuracy of 81%. This study conducts a thorough and comparative examination of two algorithms decision trees and random forest. They were tested on the same dataset, and the findings showed that the random forest method beat the decision tree approach with significantly higher accuracy. The random forest classifier delivered an accuracy of 80%, whereas the decision tree approach delivered an accuracy of 73% [4]. The experiments done in article [5] utilizing the C4.5 algorithm in decision trees revealed that the highest accuracy value attained was 78.08 percent with a data partition of 90:10 and the highest recall value was 96.4 percent with an 80:20 data partition. As a result, the division of 80:20 was determined to be the best due to its high accuracy and recall value. The authors [6] conducted an exploratory data analysis. The primary goal of the article was to categorize and investigate the characteristics of loan applicants. Seven distinct graphs were plotted and displayed, and using these graphs, the analyses showed that most loan applicants chose short-term loans. Syed Zamil Hasan Shoumo et al. [7] determined that Support Vector Machines outperformed other models employed in the research for comparative performance evaluation, such as logistic regression and random forest. They create decision tree, Support Vector Machine (SVM), AdaBoost, Bagging and Random Forest models and evaluate their prediction accuracy to a Logistic Regression model benchmark. The comparisons are assessed using standard classification performance measures. When compared to other models, our results suggest that Adaboost and Random Forest perform better. Furthermore, Support Vector Machine models perform poorly when employing both linear and nonlinear kernels. Their findings indicate that there are values that create possibilities for business to make default prediction models by experimenting with machine learning approaches [8].  In this work, we addressed how to estimate loan default probability using classifiers based on machine and deep learning models using real-world data. The most significant characteristics from multiple models are chosen and then utilized in the modeling process to compare the performance of Random Forest classifiers and decision tree classifiers on data to assess their reliability. The author achieved 78.64 percent effectiveness from the random forest classifier when we use parameter tuning with the proper hyper parameter. They obtained an efficiency of 85.3 percent, which is comparable to the decision tree classifier's prediction efficiency [9]. This study develops a loan default prediction model based on real-world customer loan data from Lending Club using the Random Forest method. To address the issue of imbalanced classes in the dataset, the SMOTE technique is used, followed by a series of procedures such as data cleaning and dimensionality reduction. In terms of forecasting default samples, the experimental findings demonstrate that the Random Forest method beats logistic regression, decision trees, and other machine learning techniques [10]. Kvamme et al. [11] used Convolutional Neural Networks (CNN) to forecast loan default by analyzing time series data from customer transactions in savings accounts, current accounts and credit cards. Ma et al. [12] employ XGBoost, LightGBM, Random Forest, and Logistic Regression (LR) to build a set of prediction models for determining the likelihood of a customer's loan default.  Zhu et al. [13] used the random forest method to create a model for predicting loan default in the lending club, and the results were compared to the results of three other algorithms: LR, DT, and SVM. The experiment revealed that the random forest method outperforms the other three algorithms in terms of loan default prediction and has a high capacity to generalize.   Khan et al. [14] utilized predictive models based on LR, DT, and Random Forest. The accuracy of DT is more than Random Forest and LR, while cross-validation is 80.130%, 72.213%, and 80.945% respectively. The predictive model is beneficial in terms of decreasing the time and efforts necessary to approve loans as well as filtering out the best applicants for granting loans. Further study can be found in the following works [15-25].

Literature survey is mentioned in section 2. Materials and methods are presented in section 3. Result analysis and future work with conclusion is depicted in section 4 and 5 respectively. 

3.	Material and Methods

3.1.	Dataset representation
The training dataset contains 25 columns as represented in Table 1 which are supplied to machine learning model; on the basis of this data set the model is trained. Every new applicant detail filled at the time of application form acts as a test data set. After the operation of testing, model predict whether the new applicant is a fit case for approval of the loan or not based upon the inference it concludes on the basis of the training data sets.





![image](https://user-images.githubusercontent.com/67271184/128602656-6ba648be-d5b9-4a9f-9b2e-409b863fb1c8.png)



3.2	Mathematical Model of Random Forest.
Random forests (RF) construct many individual decision trees at training. Predictions from all trees are pooled to make the final prediction; the mode of the classes for classification or the mean prediction for regression. As they use a collection of results to make a final decision, they are referred to as Ensemble techniques.
	Feature Importance
Feature importance is calculated as the decrease in node impurity weighted by the probability of reaching that node. The node probability can be calculated by the number of samples that reach the node, divided by the total number of samples. Higher the value more important the feature.
	Implementation in Scikit-learn
For each decision tree, Scikit-learn calculate a nodes importance using Gini Importance, assuming only two child nodes (binary tree):

〖Km〗_n=W_(n ) T_n - W_(left(n) ) T_(left(n))- W_(right(n) ) T_(right(n))


	〖Km〗_n= the importance of node n 
	W_(n ) = weighted number of samples reaching node n
	T_n= the impurity value of node n
	left(n) = child node from left split on node n
	right(n) = child node from right split on node n


F_(m )=(∑_(n:node n splits on feature m)▒〖Km〗_n )/(∑_(k∈ all nodes)▒〖Km〗_k )

	F_(m )= the importance of feature m
	〖Km〗_n= the importance of node n

These can then be normalized to a value between 0 and 1 by dividing by the sum of all feature importance values:

〖NF〗_(m )=F_(m )/(∑_(n∈ all nodes)▒F_(n ) )


The final feature importance, at the Random Forest level, is it’s average over all the trees. The sum of the feature’s importance value on each trees is calculated and divided by the total number of trees:

〖RF〗_(m )=(∑_(n∈ all trees)▒〖NF〗_(mn ) )/To

	〖RF〗_(m )= the importance of feature m calculated from all trees in the Random Forest model
	〖NF〗_(mn )= the normalized feature importance for m in tree n
	To = total number of trees

3.3	Model evaluated using Random Forest:
Model evaluation is technique which is used for theevaluating the performance of the model based on someconstraints it should be kept in mind while evaluating the modelthat it can’t underfoot or overfit the model. Various methods arepresent to evaluate the performance of the model such asConfusion metrics, Accuracy, Precision, Recall, F1 score etc.

Extracting the Importance Features for Predicting
Credit Defaulters
Random forest
Random forest or random decision forests are an ensemble learning method used for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes or mean prediction of the individual trees.
Random Forest is a supervised learning algorithm. It is like an ensemble of decision trees with bagging method. The general idea of the bagging method is that a combination of learning models improves the overall result. The Random Forest algorithm randomly selects observations and features to build several decision trees and then averages the results.
Random forest prevents overfitting problem for most of the time, as it creates several random subsets of the features and only construct smaller subtrees.
Step 1: construct a random forest model by default using Scikit-learn package and conduct a confusion matrix to see how the model performs on the loan repayment dataset.


Random forest is a supervised learning algorithm. The "forest" it builds, is an ensemble of decision trees, usually trained with the “bagging” method. The general idea of the bagging method is that a combination of learning models increases the overall result.
So, in brief we can say thatrandom forest builds multiple decision trees and merges them to get a more accurate and stable prediction.



![image](https://user-images.githubusercontent.com/67271184/128602676-00f3d888-4395-41f1-9969-ed151911f6e9.png)



The Random Forest Classifier
Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each tree in the random forest spits out a class prediction and the class with the most votes become our model’s prediction





![image](https://user-images.githubusercontent.com/67271184/128602690-cb55f946-cf73-4e07-a5b0-d90f16149cbf.png)




In the above picture, we can see how an example is classified using n trees where the final prediction is done by taking a vote from all n trees.

Working model of Decision Tree


![image](https://user-images.githubusercontent.com/67271184/128602721-27d22614-c939-44a2-8438-cac88f756b39.png)


Working model of Random Forest

As here the Model used is Random Forest and we have used n_estimators value as 100, Here the value of n_estimators states that how many decision trees, is used in the process. 
In selected data samples, gets prediction from each tree and selects the best solution by means of voting. It also provides a pretty good indicator of the feature importance. The working of this models is Like – 
1) Select random samples from the Dataset.
2) Construct a decision tree for each sample and get a prediction result for each decision tree.
3) Then a vote has been performed to predict result.
4) Then the result with the most vote is selected, and the final prediction is given.




![image](https://user-images.githubusercontent.com/67271184/128602739-83d75760-513e-473f-ac52-0243ae187372.png)


1.	Here we can see how our Random Forest Model Works by Constructing decision trees for each sample and get a prediction result from each decision tree.




Result and Analysis

![image](https://user-images.githubusercontent.com/67271184/128602760-7a1f2b87-6b99-4223-b466-6000fec7f811.png)









