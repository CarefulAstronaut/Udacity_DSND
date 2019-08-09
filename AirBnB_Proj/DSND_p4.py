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
barWidth = 0.25

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

#DF Manipulation
df_A2 = df_A[['neighbourhood_group_cleansed', 'room_type', 'price', 'price_per_bed']]
df_A2.set_index('neighbourhood_group_cleansed', inplace = True)
df_A2.loc[['Magnolia', 'Queen Anne', 'Downtown', 'West Seattle', 'Cascade']]

for i in (['Magnolia', 'Queen Anne', 'Downtown', 'West Seattle', 'Cascade']): 
    print(df_A2.loc[i].groupby('room_type').mean())
