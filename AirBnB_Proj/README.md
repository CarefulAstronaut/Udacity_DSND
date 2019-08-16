# AirBnB Project
### Udacity DSND Project 4

## Business Case

This project was designed to help AirBnB hosts with their business.  It has some basic demographic statistics around neighbourhood prices and price per bed, but also digs into some of the business decisions a host needs to make.  Is it in your interest as a host to garner more reviews? How does the cancellation policy change how far in advance people will book your unit?

## Included Files

This project was done entirely with the AirBnB Seattle open dataset from Kaggle, specifically the listings.csv file.  The listings.csv file (full) is the unedited version, while the listings.csv file has some value type edits to help import values as numeric and not string.  

## Libararies

This repo uses some basic python libraries to dig through the data.  Most of this work is done through visualiations and grouping buckets together, but it is still effective to solve our business questions. 

Libraries: pandas, numpy, matplotlib

## Summary

The neighbourhood price statistics are pretty standard.  Interesting to note that the most expensive neighbourhood (Magnolia) is only viable if the entire house/apt is offered, as they have some of the lowest cost private/shared room listings.  

The overall number of reviews doesn't have a direct correlation to how many bookings a unit has, but there may be more to dig into there with negative/postive reviews, accuracy scores, etc.  Cancellation policy has a huge impact on advance bookings, and we dig into the business ramifcations of that choice in the notebook
