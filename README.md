## BANK LOAN APPROVAL PREDICTION WITH DATA ANALYSIS


By – Faizan Ashraf
At First, I have done Bank loan Approval Analysis and have done Visual Analysis on the factors which Influence, increases, or decreases the chances of a person getting a Bank Loan.
Then I have made a classification ML model where the model can predict whether a loan will be approved or not. Specifically, it is a binary classification problem where we have to predict either one of the two classes given i.e., approved 1 or not approved 0.

Data – 
The data contains 24 Columns which are –:
1)	Branch Code

2)	Application In-take Date

3)	Application Input Date

4)	Applied Loan Amount

5)	Applied Loan Tenor

6)	Loan Purpose

7)	Title

8)	Gender

9)	Age

10)	Marital Status
11) Education Level
12) Residential Status
13) Monthly Housing/Rental
14) Contract Staff (Y/N)
15) Contract End Date
16) Employment Type
17) Nature of Business
18) Job Position
19) Monthly Income
20) Office (Area)
21) Date (Full Doc)
22) Date (Pending Doc)
23) Date (Pending Approval)
24) Final Status
25) Indicators



### COUNTING REJECTED AND ACCEPTED APPLICAN

![image](https://user-images.githubusercontent.com/67271184/128602814-d5b87c57-932d-47d0-bb13-f3fc40e83836.png)


The above bar Graph gives us the Idea regarding the distribution of Loan Acceptations, Rejections and Approval pending.

From the Data We have, we can conclude that Rejected Number of Loan Applications is 6278 and number of Approval Drawdown are 5345 and the number of Approval Pending are 46.


### Loan Approval based on Gender 

![image](https://user-images.githubusercontent.com/67271184/128602819-668e47d0-c2e9-44b2-a33a-60032b6be230.png)

1) We see that 68% of loan Application is given by Male. And around 32% are Female. 

2) This Gender Distribution graph gives us the idea of how many Applicants who apply for a bank loan are Male and how many out of them are Female.


### Loan Applicants Distribution based on Marital Status
                         

![image](https://user-images.githubusercontent.com/67271184/128602826-88079be4-4c29-478a-bfef-6557b56e54b3.png)


1) Here we see that around 62% of loan applications are given by Married. 
2) Around 32% of Loan Applicants are Single 
3) People who have taken Divorced and taking a loan are around 4%.
4) Around 2%-3% of the Loan Applicants have not given any response on their marital status by this we can say that around 2%-3% of the people do not want to tell about their marital status.
5) 1% of the Loan Applicants are Widowed.
 




### Loan Approval based on Loan Purpose Education Level and Residential Status.


![image](https://user-images.githubusercontent.com/67271184/128602832-7ce1933c-8c84-4ae7-a964-bea7502b96a3.png)


From The First Graph Loan Purpose
1) From the above analysis we see that around 50% of the loan applicants take the loan for Personal Use.
2) The second reason for which people have applied for a loan is to pay their taxes.
3) Then at Third we see that around 10% of the Loan Applicants apply for Loan to pay their Credit card bills.





### From The Second Graph Education Level of Loan Applicants.

1) From the above analysis we see that around 40% of the loan applicants have done their education till Secondary Schooling.
2) 25% of the Loan Applicants who have applied for Loan are University Students or University Passout.
3) 22% of the Loan Applicants who have applied for Loan are Post-Graduate Students or have done Post-Graduation. 
4) 4% of the Loan Applicants who have applied for Loan are Post-Secondary Students or have done Post-Secondary Education Level.


### From The Third Graph Education Level of Residential Status

1) From the above analysis we see that around 45% of the loan applicants Live with their Relatives.
2) Around 20% of the loan applicants Live at a rented house.
3) Around 18% of the loan applicants live at Martgaged Private Housing.
4) Around 13% of the loan applicants live at Self Owned Private Houses.








### Loan Approval on the basics of Employment type

![image](https://user-images.githubusercontent.com/67271184/128602843-4fc5b434-9d4c-4594-b917-d9f3515c50de.png)

Here From the above analysis, we see that around 72% of the loan applicants are Fixed Income Earners. 
12% of the loan Applicants are Civil servants and 9-8% are Non-Fixed income Earners Or self-employed






### Loan Analysis on the basics of Nature of Business

![image](https://user-images.githubusercontent.com/67271184/128602849-08b4ae94-a3b9-4c91-a4e4-b98f5cc95333.png)

1) Here From the above analysis, we see that around 28% of the loan applicants are Manager 
2) Around 18% are Office Workers.

3 ) 13% of the Loan Applicants are in the Service sector.
4) Around 10% of the Loan Applicants are Executive and 7% are Owner of Business


               


	



### Loan Approval Analysis on the basics of Monthly Income

![image](https://user-images.githubusercontent.com/67271184/128602862-dde82dbd-4658-42a1-86a2-4f10c32831a8.png)


It can be inferred that most of the data in the distribution of applicant Mounty's income are towards the left which means it is not normally distributed. The distribution is right-skewed (positive skewness).
The boxplot confirms the presence of a lot of outliers/extreme values. This can be attributed to the income disparity in society. Part of this can be driven by the fact that we are looking at people with different education levels.








### Loan Application Analysis on the basics of Education Level

![image](https://user-images.githubusercontent.com/67271184/128602872-c901ee51-46e3-4870-b3f0-6176d00d4a71.png)

Analysis on how final result varies with Married.

![image](https://user-images.githubusercontent.com/67271184/128602894-6ee1f537-fb54-4e5c-a1b3-c184f0b6b9cf.png)

![image](https://user-images.githubusercontent.com/67271184/128602902-19a744b2-b8b7-4ffc-af0a-073cabe95990.png)

FROM THE BAR CHARTS ABOVE WE CAN SAY THAT - :
. Male and Female have a quite similar chance of getting a loan.

. Married people have a 50% chance of getting the loan approved.

. Civil servants have the most chance of getting the loan approved.

. People who are post Graduate or have a University degree have the most chance of getting the loan approved.

. People with Government/Semi-government have a 70% chance of getting their Loan Approved.

. In the business line, people who are Executive and Professional have the most chance of getting a loan that is around 70%.

. People who take the loan for Tax Payment purposes have a major chance of getting a loan that is around 72%.







### Analysis on how final result varies with Education Level

![image](https://user-images.githubusercontent.com/67271184/128602918-c4e16a7f-2cfa-4939-b9b2-34020d99ce3d.png)



FROM THE BAR CHARTS ABOVE WE CAN SAY THAT -:
. Male and Female have a quite similar chance of getting a loan.

. Married people have a 50% chance of getting the loan approved.

. Civil servants have the most chance of getting the loan approved.

. People who are post Graduate or have a University degree have the most chance of getting the loan approved.

. People with Government/Semi-government have a 70% chance of getting their Loan Approved.

. In the business line, people who are Executive and Professional have the most chance of getting a loan that is around 70%.

. People who take a loan for Tax Payment purposes have a major chance of getting a loan that is around 72%.






###             -------------------------------------------------------ML MODELING-----------------------------------------------------------


So, the Algorithm we have used to use this Data and make a model that can predict whether any applicants will get a loan or loan or the basics of different criteria.

The ML model we are using here is Random Forest Regressor

### HOW RANDOM FOREST WORKS


Random forest is a supervised learning algorithm. The "forest" it builds, is an ensemble of decision trees, usually trained with the “bagging” method. The general idea of the bagging method is that a combination of learning models increases the overall result.
So, in brief, we can say that random forest builds multiple decision trees and merges them to get a more accurate and stable prediction.


![image](https://user-images.githubusercontent.com/67271184/128602947-6f4564da-db71-459b-9141-7e4982fc28a3.png)


### The Random Forest Classifier

Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each tree in the random forest spits out a class prediction and the class with the most votes become our model’s prediction




#### How Random Forest algorithm works?


![image](https://user-images.githubusercontent.com/67271184/128602952-2c36b46b-14c2-4ef1-b9aa-efab37bd9138.png)


In the above picture, we can see how an example is classified using n trees where the final prediction is done by taking a vote from all n trees.





### HOW A DECISION TREE REPRESENTATION USED IN OUR PROJECT WORKS

![image](https://user-images.githubusercontent.com/67271184/128602964-aee6a826-4765-44a8-b7ff-5c988793b7b3.png)


### FULL WORKING OF OUR RANDOM FOREST

![image](https://user-images.githubusercontent.com/67271184/128602970-d03d686b-4f4c-41f7-a683-179f3e55dad7.png)





### MATHEMATICS BEHIND RANDOM FOREST

Random Forests
Random forests (RF) construct many individual decision trees at training. Predictions from all trees are pooled to make the final prediction; the mode of the classes for classification or the mean prediction for regression. As they use a collection of results to make a final decision, they are referred to as Ensemble techniques.


Feature Importance
Feature importance is calculated as the decrease in node impurity weighted by the probability of reaching that node. The node probability can be calculated by the number of samples that reach the node, divided by the total number of samples. The higher the value the more important the feature.



### Implementation in Scikit-learn
For each decision tree, Scikit-learn calculates the importance of a node using Gini Importance, assuming only two child nodes (binary tree):
 
•	ni sub(j)= the importance of node j
•	w sub(j) = weighted number of samples reaching node j
•	C sub(j)= the impurity value of node j
•	left(j) = child node from left split on node j
•	right(j) = child node from right split on node j

 
•	fi sub(i)= the importance of feature i
•	ni sub(j)= the importance of node j

These can then be normalized to a value between 0 and 1 by dividing by the sum of all feature importance values:
 
The final feature importance, at the Random Forest level, is its average over all the trees. The sum of the feature’s importance value on each tree is calculated and divided by the total number of trees:
 
•	RFfi sub(i)= the importance of feature I calculated from all trees in the Random Forest model
•	normfi sub(ij)= the normalized feature importance for I in tree j
•	T = total number of trees

















