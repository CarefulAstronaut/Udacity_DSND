# Bertelsmann/Arvato Project
### Udacity DSND Capstone Project

## Business Case

This project is a fairly common thing in business.  How to maximize results with the least amount of costs; maximize Return On Investment. This particular case is about sending letter to potential customers, important for charities, non-profits and non-sales businesses alike. 

## Included Files

This project incldues 4 datasets:

1. Arvato - Demographic information about a general subset of the German population (890k x 366)
2. Customers - Demographic information about actual customers, plus 3 additional columns around purchases (190k x 369)
3. Train - Demographic information about individuals who were mailed materials, along with their responses (43k x 367)
4. Test - Demographic information about individuals who were mailed materials, used for scoring purposes (43k x 366)

## Libararies

This repo uses some basic python libraries to dig through the data.  Most of this work is done through visualiations and grouping buckets together, but it is still effective to solve our business questions. 

Libraries: pandas, numpy, matplotlib, dill, sklearn, workspace_utils (custom .py file from Udacity)

## Summary

The results from this project were not very good, as I only managed to get an AUC score of .507.  I didn't plan my prcedure based on what the project goals were, and wasn't thinking about what I needed to get from part 1 to 2.  By using the K-Means clustering I was unable to pass on the learnings and scale of the data from part 1.   

In redoing this, I would change the algorithms to something creates a clustering scale rather than assigning a datapoint to a group.  DBSCAN or a GMM would give the information I need, and would hopefully get a better result.  
