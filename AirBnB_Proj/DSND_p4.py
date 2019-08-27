# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:00:06 2019

@author: Mitchell Rea
@MMM UID: A7HN3ZZ
@Github: https://github.com/MitchRea
"""

"""
Using the Seattle Open AirBnB data I want to answer the following questions:
A. Basic Statistics
    1) Average price per neighbourhood
    2) Property/Room Type Statistics
    
B. "Heathly" Pipelines
    1) Are there inherent attributes that contribute to the pipeline?
    2) How important are reviews to pipeline health?
    3) How does cancellation policy affect long-term vs short-term pipeline?
"""
## Package and Data Imports

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_listings = pd.read_csv("Udacity/07 - Intro to Data Science/listings.csv")

df_listings.neighbourhood_group_cleansed.head()
df_listings['id'].groupby(df_listings['calendar_updated']).count()

df_A = df_listings[['neighbourhood_group_cleansed', 'property_type', 'room_type', 'amenities', 'beds', 'price']]
df_A['price_per_bed'] = df_A['price'] / df_A['beds']
df_A.info()


## A1 Section

# DF Manipulation
df_A1 = df_A[['neighbourhood_group_cleansed', 'beds', 'price', 'price_per_bed']]
df_A1 = df_A1.groupby('neighbourhood_group_cleansed').mean().reset_index()
df_A1 = df_A1.sort_values('price', ascending=False)
A1_neighbourhoods = df_A1.neighbourhood_group_cleansed.unique()

# Chart Building
barWidth = 0.4

bars1 = df_A1.price
bars2 = df_A1.price_per_bed
#bars3 = df_A1.beds

r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
#r3 = [x + barWidth for x in r2]

plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='Avg. Listing Price')
plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='Avg. Bed per Listing')
#plt.bar(r3, bars3, width=barWidth, edgecolor='white', label='Avg. Price per Bed')

plt.xlabel('Neighbourhood', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], A1_neighbourhoods, rotation='vertical')

plt.legend()
plt.show()


## A2 Section

# Lists
roomtype = ['Entire Home/Apt', 'Private Room', 'Shared Room']
nbh = ['Magnolia', 'Queen Anne', 'Downtown', 'West Seattle', 'Cascade']
A2_results = []

#DF Manipulation
df_A2 = df_A[['neighbourhood_group_cleansed', 'room_type', 'price', 'price_per_bed']]
df_A2.set_index('neighbourhood_group_cleansed', inplace = True)
df_A2.loc[['Magnolia', 'Queen Anne', 'Downtown', 'West Seattle', 'Cascade']]

for i in (nbh): 
    print(df_A2.loc[i].groupby('room_type').mean())
    A2_results.append(df_A2.loc[i].groupby('room_type').mean())
    
# Chart Building
for i, t in zip(range(5), nbh):

    bars1 = A2_results[1].iloc[:, 0]
    bars2 = A2_results[1].iloc[:, 1]
    
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='Avg. Listing Price')
    plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='Avg. Price per Bed')

    plt.xlabel('Room Type', fontweight='bold')
    plt.xticks([r + (barWidth/2) for r in range(len(bars1))], roomtype)
    
    plt.title(t)

    plt.legend()
    plt.show()

## B2 Section

# DF Manipulation
df_B = df_listings[['calendar_updated', 'availability_30', 'availability_60', 'availability_90', 'availability_365',
                     'number_of_reviews', 'cancellation_policy']]

# Isolating Listings that have availability calendars updated within the last week
df_B.calendar_updated.unique()
B_list1 = ['never', 'today', 'yesterday', '2 days ago', '3 days ago', '4 days ago', '5 days ago', 
            '6 days ago', 'a week ago', '1 week ago', '2 weeks ago', '3 weeks ago', '4 weeks ago', 
            '5 weeks ago', '6 weeks ago', '7 weeks ago', '2 months ago', '3 months ago', '4 months ago', 
            '5 months ago', '6 months ago', '7 months ago', '8 months ago', '9 months ago', '10 months ago', 
            '11 months ago', '12 months ago', '13 months ago', '14 months ago', '15 months ago', '16 months ago', 
            '17 months ago', '22 months ago', '30 months ago']
B_list2 = list(range(-1, 33))
B_dict = dict(zip(B_list1, B_list2))
df_B.replace({"calendar_updated": B_dict}, inplace=True)
df_B = df_B.loc[(df_B['calendar_updated'] >= 0) & (df_B['calendar_updated'] <= 7)]
df_B = df_B.loc[(df_B['number_of_reviews'] > 0)]

df_B = df_B.sort_values('number_of_reviews', ascending=False)

# Determine Occupancy Rates for each time period
df_B['Occ_Rate_30'] = 1 - (df_B['availability_30'] / 30)
df_B['Occ_Rate_60'] = 1 - (df_B['availability_60'] / 60)
df_B['Occ_Rate_90'] = 1 - (df_B['availability_90'] / 90)
df_B['Occ_Rate_365'] = 1 - (df_B['availability_365'] / 365)

# Group Occupanacy Rates by # of Reviews Buckets
bucket = np.array([0, 1, 5, 10, 25, 50 , 75, 100, 125, 150, 200, 250, 300, 400, 500])
df_B1 = df_B.copy()
df_B1['review_bucket'] = pd.cut(df_B1.number_of_reviews, bins=bucket)
df_B1.drop(['calendar_updated', 'availability_30', 'availability_60', 'availability_90', 
            'availability_365', 'cancellation_policy', 'number_of_reviews'], axis='columns', inplace=True)
df_B1 = df_B1.groupby('review_bucket').mean().reset_index()

# Chart Building
plt.plot(bucket[0:14], df_B1['Occ_Rate_30'], color='b')
plt.plot(bucket[0:14], df_B1['Occ_Rate_60'], color='g')
plt.plot(bucket[0:14], df_B1['Occ_Rate_90'], color='r')
plt.plot(bucket[0:14], df_B1['Occ_Rate_365'], color='c')

plt.xlabel("Number of Reviews")
plt.ylabel("Percent Occupancy Rate")

plt.legend()

## B3 Section

# DF Manipulation
df_B2 = df_B.copy()
df_B2.drop(['calendar_updated', 'availability_30', 'availability_60', 'availability_90', 
            'availability_365', 'number_of_reviews'], axis='columns', inplace=True)
df_B2 = df_B2.groupby('cancellation_policy').mean().reset_index()
B2_policy = df_B2.cancellation_policy.unique()

# Chart Building
barWidth = 0.25

bars1 = df_B2.Occ_Rate_60
bars2 = df_B2.Occ_Rate_90
bars3 = df_B2.Occ_Rate_30

r1 = np.arange(len(bars1))
r2 = [x + (barWidth) for x in r1]
r3 = [x - (barWidth) for x in r1]

plt.bar(r3, bars3, width=barWidth, edgecolor='white', label='Occupied 30')
plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='Occupied 60')
plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='Occupied 90')

plt.xlabel('Cancellation Policy', fontweight='bold')
plt.xticks([r  for r in range(len(bars1))], B2_policy, rotation='vertical')

plt.legend()
plt.show()
