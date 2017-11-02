# Prediction on Term Deposit's subscription
A course project that uses several machine learning algorithms to analyze data

## Data Source: https://www2.1010data.com/documentationcenter/prod/Tutorials/MachineLearningExamples/BankMarketingDataSet.html

## Business Problem & Business Value: 

A Portuguese banking institution attempts to get more of its clients to subscribe for a term deposit. A higher amount of term deposit subscription creates more opportunities for the bank to increase profit, which allows the bank to invest in higher gain financial products and to pay higher interest to its customers. 

## Supervised data mining problem, Data Features, Target Variable & Possible Useful Features:

Given a customer's basic information, past campaign results, several current economic indicators and whether the customer finally subscribe a term deposit or not, we will train a predictive model. This predictive model can effectively predict whether a customer will subscribe a term deposit based on his or her personal information, past campaign results and current economic environment.

The dataset contains 41,188 data instances. Each data instance includes 20 features, which can be categorized into personal information, previous market campaign result and current economic indicators. Personal information includes age, job, marital status, education, default, housing, loan, and contact method; previous market campaign result includes days and month that last contact was made, duration of last contact in seconds, number of days since the client was last contacted in a previous campaign, number of contacts performed during this campaign for this client and outcome of the previous marketing campaign; current economic indicators includes employment variation rate, consumer price index, consumer confidence index, euribor 3-month rate and number of employees.

The target variable named "Y" in the dataset, is binary and indicates whether the client has subscribed for a term deposit. 

Among the 20 features, contact communication type will not be considered as explanatory variable because it only tells either the client was contacted by telephone or cellular. All rest features can be considered as useful variables so far. 

## Business Scenario:

The predictive model will contribute to bank’s marketing team to better targeting their potential clients who are more likely to make subscriptions for term deposit. When given a new client’s information, this model will give a predictive result of whether the customer will subscribe for a term deposit with the bank. Then, the marketing team can focus on advertising the bank’s term deposit products to those clients whose predictive results are positive. Also, the dataset includes some data about when and how often the client was contacted. By dealing with those data, we might be able to give suggestions on how to conduct campaigns in a more effective way.

Nov 2, 2017
