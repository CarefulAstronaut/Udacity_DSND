# Disaster Response Project
### Udacity DSND Project 5

## Included Files

In the /Data folder, there are 2 .csv files "messages.csv" and "categories.csv".  These files contain both the translated messages and associated metadata, along with the category assignments for the suprvised learning section. 

Spread amongst the 3 folders are the relevant .py scripts to be used for the app.  ETL script in /Data, Model training in /Models, and the script to run the app in /App. 

## Libararies

This repo uses some basic python libraries to dig through the data, as well as SQLalchemy, pickle, and part of the NLTK library. 

Libraries: pandas, numpy, matplotlib, nltk, pickle, sys, sqla, sklearn

## Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Learning Summary

This project represents my first foray into app creation and writing a true python executable program.  Almost all of the my python expereince so far has been in cells and notebooks to run as one off code.  This represents a big learning step for me in my python and DS journey. 
