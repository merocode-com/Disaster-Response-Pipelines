# Disaster Response Pipeline Project

## Project Overview
This project is meant to implement a complete machine learning pipeline to demonstrate Data Engineering skills which is part of Udacity Data Scientist Nanodegree.

In this project, real disaster data from Figure Eight is analyzed to build a model for an API that classifies disaster messages.\
The used data set contains real messages that were sent during disaster events.\
A machine learning pipeline is implemented to categorize these messages so that later on every message can be sent to an appropriate disaster relief agency.\

The project includes a web app where an emergency worker can input a new message and get classification results in several categories.\

![screen recording of the web app](media/classifier_app.gif)



### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_messages.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_messages.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

