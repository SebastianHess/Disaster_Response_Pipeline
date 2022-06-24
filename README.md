## Data Scientist Udacity Nanodegree: Disaster Response Pipeline.

### Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)


## 1. Introduction <a name="introduction"></a>

<p align="center">
    <img src="/pics/pexels-denniz-futalan-942560.jpg">
</p>

<span style="text-align: right">[Source of picture](https://www.pexels.com/de-de/foto/silhouette-des-feuerwehrmanns-der-schlauch-halt-942560/ "Source of picture")</span>

  



### Project Overview

This project analyzes disaster data from <a href="https://appen.com/">Appen</a> (formally Figure 8) to build a model for an API that classifies disaster messages.

By creating a machine learning pipeline form a real messages data set to categorize the events it is possible to send the messages to an appropriate disaster relief agency.

The project includes a webb app which fulfills the following parts:

1. An input of an new message will create classified results in several categories. 

![1](/pics/disaster-response-project2.jpg)

2. It displays also visualizations of the data. 

![2](/pics/disaster-response-project1.jpg)

<span style="font-size:1em;">[Source of picture](https://learn.udacity.com/nanodegrees/nd025/parts/cd0018/lessons/ea367f74-3d5a-42b1-92a3-d3d3734fd369/concepts/d7e645c3-a521-4214-8bd5-30e7137365cc")</span>


## 2. Installation <a name="installation"></a>

Everything will run with a Python 3.x version.
It is necessary to install the following libraries to run the code in Anaconda. 

#### ETL_Pipeline_Preparation.ipynb
```
import pandas as pd
from sqlalchemy import create_engine
```

#### ML_Pipeline_Preparation.ipynb
```
import nltk
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
```

## 3. Project Motivation<a name="motivation"></a>

#### ETL Pipeline

The first part of the data pipeline is the Extract, Transform, and Load process. The task is to clean the data and  store it in a SQLite database. The cleaning process is done with pandas. To load the data into an SQLite database, the pandas dataframe ```.to_sql()``` method is used with an SQLAlchemy engine. The cleaning code of the jupyter notebook is included in the final ETL script ```process_data.py```.

#### ML Pipeline

For the machine learning portion, the data is split into a training set and a test set. The machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and ```GridSearchCV``` is used to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). Finally the model is exported to a pickle file. The final machine learning code is included in the script ```train_classifier.py```.

After completing the jupyter notebooks for the ETL and machine learning pipeline, the work is transfered into Python scripts ```process_data.py``` and ```train_classifier.py```. It should be able to easily create a new model just by running your code, if revised or new datasets of messages are used. These Python scripts are runing with additional arguments specifying the files used for the data and model.


#### Flask App

The results are displayed in a Flask web app. A file structure with starter files is provided. **Html**, **css**, and **javascript** is used to make the web app including visualizations. 


## 4. File Descriptions <a name="files"></a>

### File structure
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```

### Running the Web App

1. Open a new terminal window. Use terminal commands to navigate inside the folder with the ```run.py``` file.

2. Type in the command line: ```python run.py```

Your web app should now be running if there were no errors.

### Setting up the database and model by runnig the following commands in the project root directory.

1. Run ETL pipeline:
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

2. Run ML pipeline:
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

3. Run web app:
* Navigating to the app's directory: ```cd app```
* Running web app: ```python app/run.py```

4. Open web app: http://0.0.0.0:3001/



## 5. Results<a name="results"></a>

MY Github repository: [https://github.com/SebastianHess/Disaster_Response_Pipeline.git](https://github.com/SebastianHess/Disaster_Response_Pipeline.git)



## 6. Licensing, Authors, Acknowledgements<a name="licensing"></a>

* Thanks to my employer for enabling me to take the Udacity Nanodegree and for supporting me.
* Thanks to Udacity for the amazing [Data Scientist Nanodegree Programm](https://www.udacity.com/course/data-scientist-nanodegree--nd025).