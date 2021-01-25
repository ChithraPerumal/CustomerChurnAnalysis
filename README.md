# CustomerChurnAnalysis
Predict customers likely to churn for an e-commerce company<br>

<b<Problem Statement</b>
An online retail (E commerce) company wants to know the customers who are going to churn, so accordingly they can approach customer to offer some promos.

Below is the description of variables:
•	Customer ID  : Unique customer ID<br>
•	Churn : Churn is the target variable; Indicates whether a customer churned or not. 0 indicates not Churned and 1 is churned.<br>
•	Tenure : Number of months since customers first transaction<br>
•	Preferred Login Device : Preferred login device of customer<br>
•	City Tier : Indicates Customer lives in Tier 1,Tier 2 or Tier 3 city. Tier 1 is metro and Tier 3 is town.<br>
•	Warehouse To Home : Distance in between warehouse to home of customer<br>
•	Preferred Payment Mode : Preferred payment method of customer. It is a categorical variable and takes values Debit card, Credit Card, E wallet, UPI, COD ,CC, Cash on Delivery<br>
•	Gender : Gender of customer – Male, Female<br>
•	Hour Spend On App : Average number of hours spend on mobile application or website last month<br>
•	Number Of Device Registered : Total number of devices  registered on the customer account<br>
•	Preferred Order Cat : Preferred order category of customer in last month. It is a categorical variable and takes values Laptop & Accessory, Mobile Phone, Fashion, Mobile, Grocery and Others.<br>
•	Satisfaction Score : Satisfaction score of the customer. This is an ordinal variable. 1 is Very highly satisfied ,2 is highly Satisfied, 3 is Satisfied, 4 is Not satisfied and 5 is Disappointed<br>
•	Marital Status : Marital status of the customer – Single, Married, Divorced<br>
•	Number Of Address : Total number of address on the customer account<br>
•	Complain : Whether any complaint has been raised in last month<br>
•	Order Amount Hike From Last Year : Percentage increase in order amount from last year<br>
•	Coupon Used : Total number of coupons  used  last month<br>
•	Order Count : Total number of orders placed last month<br>
•	Day Since Last Order : Number of days since last order was placed by customer<br>
•	Cashback Amount : Average cashback amount in customer’s account last month<br>


<b>Data Cleaning</b><br>
Dataset contains some text values which are inconsistent which were cleaned.
•	In Preferred payment mode COD ,Cash on Delivery and CC, Credit Card are same categories. So, I combined them into  a single category.
•	In Preferred Order Cat, Mobile and Mobile Phone  has same churn rate. So, I am combining them into a single category.
•	In marital status due to difference in churn rate I am keeping Single and Divorced as separate categories.
•	In Preferred Login, Mobile phone and Phone has different churn rate. So I am keeping them as separate categories.
Refer Churn vs categories for churn rate for different categories.

Missing Values<br>
Dataset contains a few missing values in Day Since Last Order, Order Amount Hike From Last Year, Tenure, Order Count, Coupon Used, Hour Spend On App, Warehouse To Home. 
Approach 1
I imputed the missing values with median. 

Outlier treatment<br>
•	In WarehouseToHome  excluding outliers, next highest is 36. I considered outliers as data error and dropped it.
•	I dropped extreme outliers in Tenure and DaysinceLastOrder. Even though itseems like valid data outliers mess with model performance. So I dropped it.
•	Outliers in HourSpendOnApp,NumberOfDeviceRegistered, OrderAmountHikeFromlastYear,   CashbackAmount  are valid. So I am going to leave it like that for now. 
•	In NumberOfAddress excluding outliers, next highest value is 11. I considered 15 as data error and dropped it and replaced value above 10 with 10..
•	Less than 1% customers have CouponUsed greater than 9.Since we are not able to distinctively identify churn with Coupon used even for higher values , I replaced CouponUsed >10 with 10.

Variable transformation
Some algorithms (eg. distance based algorithm, mlp)work well with normalized data. So I normalized data using StandardScalar().
For Linear discriminant analysis, I did log transformation of data.
I did one hot encoding of categorical variables.
Removal/Addition of new variables
Dropped Customer Id during EDA and model building.
By using logistic regression, HourSpendOnApp,NumberOfDeviceRegistered,OrderAmountHikeFromlastYear were identified as insignificant variables. This is in par with EDA and so I  removed it from model building.
I created a new variable Score. This helps us identify unhappy customers.
Score = (5*Complain +  1 ) * SatisfactionScore<br<

<b>Model building</b><br>
•	I split the data into train(75%) and test set(25%). Train set has 4215 records and Test set has 1406 records.
•	I build a few classification model using different algorithms. 
•	First I build a model with default parameters. 
•	Tuned hyperparameter using GridSearchCV to improve model precision and recall and build a few model with different set of hyper parameters. 
•	Many of the models were overfitting after hyperparameter tuning. So I reduced the number of features by removing less important variables.
•	Did model comparison and chose the best model.
