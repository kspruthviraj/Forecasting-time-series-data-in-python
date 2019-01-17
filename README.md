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


## Data exploration

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
 
 ```
 ((18288, 3), (5112, 2))
 ```

```
(ID           int64
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


First we have to change the data type of ``datetime`` such that we can extract feature easily.

```
train['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
test['Datetime'] = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
test_original['Datetime'] = pd.to_datetime(test_original.Datetime,format='%d-%m-%Y %H:%M')
train_original['Datetime'] = pd.to_datetime(train_original.Datetime,format='%d-%m-%Y %H:%M')
```

Once converted, lets extract hour, day, month, year separately

```
for i in (train, test, test_original, train_original):
    i['year']=i.Datetime.dt.year 
    i['month']=i.Datetime.dt.month 
    i['day']=i.Datetime.dt.day
    i['Hour']=i.Datetime.dt.hour 

```

Lets test first hypothesis, by creating another variable that says whether the day is weekend or not

```
train['day of week']=train['Datetime'].dt.dayofweek
temp = train['Datetime']

def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0

temp2 = train['Datetime'].apply(applyer)
train['weekend']=temp2

```

Visualize the data

```
train.index = train['Datetime'] # indexing the Datetime to get the time period on the x-axis.
df=train.drop('ID',1)           # drop ID variable to get only the Datetime on x-axis.
ts = df['Count']
plt.figure(figsize=(16,8))
plt.plot(ts, label='Passenger Count')
plt.title('Time Series')
plt.xlabel("Time(year-month)")
plt.ylabel("Passenger count")
plt.legend(loc='best')


```

**FIGURE**


Now, lets test 2nd and 3rd hypothesis

```train.groupby('Hour')['Count'].mean().plot.bar()```

```train.groupby('weekend')['Count'].mean().plot.bar()```










