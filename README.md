# Forecasting-the-of-travellers-for-next-few-months-in-python-time-series-analysis-
The dataset can be downloaded from Hackathon : https://datahack.analyticsvidhya.com/contest/practice-problem-time-series-2/


Unicorn Investors wants to make an investment in a new form of transportation - JetRail. 
JetRail uses Jet propulsion technology to run rails and move people at a high speed! 
The investment would only make sense, if they can get more than 1 Million monthly users with in next 18 months. 
In order to help Unicorn Ventures in their decision, you need to forecast the traffic on JetRail for the next 7 months. 
You are provided with traffic data of JetRail since inception in the test file.

## Hypothesis:
a) There will be an increase in the traffic as the years pass by; as in general the population has a general upward trend with time.
So we can expect more people to travel by JetRail. 

b) Traffic on weekdays will be more as compared to weekends/holidays.

c) Traffic during the peak hours will be high.


## Understanding data

Look at the structure, size and its content by writing
~~~
train.columns, test.columns
train.shape, test.shape
train.dtypes, test.dtypes
~~~


This gives out

```(Index(['ID', 'Datetime', 'Count'], dtype='object'),
 Index(['ID', 'Datetime'], dtype='object'))
 ```
 ```((18288, 3), (5112, 2))
 ```
 ```(ID           int64
 Datetime    object
 Count        int64
 dtype: object, ID           int64
 Datetime    object
 dtype: object)
 ```
- ID is the unique number given to each observation point.
- Datetime is the time of each observation.
- Count is the passenger count corresponding to each Datetime.
- We have 18288 different records for the Count of passengers in train set and 5112 in test set.
- ID and Count are in integer format while the Datetime is in object format for the train file.
- Id is in integer and Datetime is in object format for test file.

