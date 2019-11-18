import sys
import pandas as pd
import sqlalchemy as sqla
import re
import nltk
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib 

def load_data(database_filepath):
    '''
    Loading the data in the model for categorization. 
    
    INPUT:
    datebase_filepath = filepath we assigned in the loading step
    
    OUTPUT:
    X = text/social media messages
    Y = binary categorization of messages
    '''
    engine = sqla.create_engine('sqlite:///data/DisasterResponse.db')
    table_name = engine.table_names()[0]
    
    df = pd.read_sql_table(table_name, engine)
    X = df.iloc[:, 1]
    Y = df.iloc[:, -36:]
    
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    '''
    Custom tokenization for the messages.  Lower-case letters, standard tokenization, 
    and standard lemmatizer. 
    
    INPUT:
    text = input text to be tokenized
    
    OUTPUT:
    clean_text = tokenized text
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", str(text)) 
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_text = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_text.append(clean_tok)

    return clean_text


def build_model():
    '''
    Builds the pipeline of algorithms to be used in this analysis
    
    INPUT:
    None
    
    OUTPUT:
    model = Fitted model against the X-train dataset
    '''
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, lowercase=False)), 
        ('tfidf', TfidfTransformer()), 
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Custom tokenization for the messages.  Lower-case letters, standard tokenization, 
    and standard lemmatizer. 
    
    INPUT:
    model = model built and defined as above
    X_test = X set for the model accuracy tests
    Y_test = Y set for the model accuracy tests
    
    OUTPUT:
    None
    '''
    Y_pred = model.predict(X_test)

    for i in range(0, 35):
        print("Scores for column: %s\n"
              % (Y_test.columns[i]))
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    Saving the model for 
    
    INPUT:
    
    OUTPUT:
    '''
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
