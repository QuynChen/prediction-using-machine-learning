# Import essential libraries
import argparse
import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from joblib import dump, load

# Data processing: Eliminate outliers
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
  """Compute outlier thresholds for a column of a dataframe based on the Interquartile Range (IQR).

  Args:
      dataframe (DataFrame): The dataframe containing the column.
      col_name (str): The name of the column for which outlier thresholds are computed.
      q1 (float, optional): The lower quantile defining the start of the interquartile range. Defaults to 0.05.
      q3 (float, optional): The upper quantile defining the end of the interquartile range. Defaults to 0.95.

  Returns:
      tuple: A tuple containing the lower and upper outlier thresholds.
  """
  quartile1 = dataframe[col_name].quantile(q1)
  quartile3 = dataframe[col_name].quantile(q3)
  interquartile_range = quartile3 - quartile1
  low_limit = quartile1 - 1.5 * interquartile_range
  up_limit = quartile3 + 1.5 * interquartile_range
  return low_limit,up_limit

def check_outlier(dataframe, col_name):
  """Check for outliers in a column of a dataframe.

  This function utilizes the outlier_thresholds function to compute outlier thresholds
  based on the Interquartile Range (IQR), and then checks if any values in the specified
  column fall outside these thresholds.

  Args:
      dataframe (DataFrame): The dataframe containing the column to be checked for outliers.
      col_name (str): The name of the column to be checked for outliers.

  Returns:
      bool: True if outliers are found, False otherwise.
  """
  low_limit, up_limit = outlier_thresholds(dataframe, col_name)
  if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
    return True
  else:
    return False

def replace_with_thresholds(dataframe, col_name):
  """Replace outliers in a column of a dataframe with their respective outlier thresholds.

  This function utilizes the outlier_thresholds function to compute outlier thresholds
  based on the Interquartile Range (IQR), and then replaces any values in the specified
  column that fall outside these thresholds with the corresponding threshold values.

  Args:
      dataframe (DataFrame): The dataframe containing the column with outliers to be replaced.
      col_name (str): The name of the column with outliers to be replaced.

  Returns:
      None
  """
  low_limit, up_limit = outlier_thresholds(dataframe, col_name)
  dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
  dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit

# Choose the appropriate algorithm  
def run_train_models(train_path, dev_path):
  """Train and evaluate multiple models using training and development datasets.

  Args:
      train_file (str): File path to the training dataset in JSON format.
      dev_file (str): File path to the development dataset in JSON format.

  Returns:
      None
  """
  train_data = pd.read_json(train_path, lines=True)
  dev_data = pd.read_json(dev_path, lines=True)

  X_train = None
  Y_train = None
  X_dev = None
  Y_dev = None
  imputer = SimpleImputer()
  scaler = StandardScaler()

  models = [LogisticRegression(random_state=42), RandomForestClassifier(random_state=42),
            HistGradientBoostingClassifier(random_state=42), DecisionTreeClassifier(random_state=42)]

  # Data processing 
  for col in train_data.columns:
    if check_outlier(train_data, col):
      replace_with_thresholds(train_data, col)

  # Note: It is known that humans cannot have variables with value 0 (except Pregnancies and Outcome). 
  # Therefore, it is necessary to process these values. The values 0 can be assigned NaN.
  zero_columns = [col for col in train_data.columns if (train_data[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
  for col in zero_columns:
    train_data[col] = np.where(train_data[col]==0, np.nan, train_data[col])

  # Prepare data for training
  X_train = train_data.drop('Outcome', axis=1)
  Y_train = train_data['Outcome']
  X_dev = dev_data.drop('Outcome', axis=1)
  Y_dev = dev_data['Outcome']

  X_train = imputer.fit_transform(X_train)
  X_train = scaler.fit_transform(X_train)
  X_dev = imputer.transform(X_dev)
  X_dev = scaler.transform(X_dev)


  for model in models:
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_dev)
    f1 = f1_score(Y_dev, Y_pred)
    print(f"{model}: ",f1, end='\n')

def run_train(train_path, dev_path, model_dir):
  """Train a machine learning model using training data and evaluate it on the development data.

  Args:
      train_path (str): File path to the training dataset in JSON format.
      dev_path (str): File path to the development dataset in JSON format.
      model_dir (str): Directory path to save the trained model.

  Returns:
      None
  """
  # Create a directory for the model if it doesn't already exist
  os.makedirs(model_dir, exist_ok=True)

  # Read training and development data
  train_data = pd.read_json(train_path, lines=True)
  dev_data = pd.read_json(dev_path, lines=True)
  
  # Data processing 
  for col in train_data.columns:
    if check_outlier(train_data, col):
      replace_with_thresholds(train_data, col)

  zero_columns = [col for col in train_data.columns if (train_data[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
  for col in zero_columns:
    train_data[col] = np.where(train_data[col]==0, np.nan, train_data[col])

  # Prepare data for training
  X_train = train_data.drop('Outcome', axis=1)
  Y_train = train_data['Outcome']
  X_dev = dev_data.drop('Outcome', axis=1)
  Y_dev = dev_data['Outcome']
  
  ros = RandomOverSampler(random_state=42)
  X_train, Y_train = ros.fit_resample(X_train, Y_train)
  
  # Model selection
  parameters = {
  'classifier__C': [0.1, 1.0, 10.0, 20.0],
  'classifier__multi_class': ['ovr', 'multinomial'],
  'imputer__strategy': ['mean', 'median'],
  'feature_selection__k': [3, 4, 5],
  }
  
  PL = Pipeline([
      ('imputer', SimpleImputer()),
      ('feature_selection', SelectKBest()),
      ('scaler', StandardScaler()), 
      ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
  GS = GridSearchCV(PL, param_grid=parameters, cv=5, scoring='f1_macro')
  GS.fit(X_train, Y_train)
  model = GS.best_estimator_
  params = GS.best_params_
  print('The best parameters', params, end='\n')
  train_f1 = GS.best_score_
  
  # perform predictions on dev set
  Y_pred = model.predict(X_dev)
  dev_f1 = f1_score(Y_dev, Y_pred)
  print('Training set F1 score: ', train_f1, end='\n')
  print("Development set F1 score: ", dev_f1)
  
  # Save model
  model_path = os.path.join(model_dir, 'trained_model.joblib')
  dump(model, model_path)


def run_predict(model_path, test_path, output_path):
  """Make predictions using a trained model and save the results to a JSON file.

  Args:
      model_path (str): File path to the trained model.
      test_path (str): File path to the test dataset in JSON format.
      output_path (str): File path to save the predictions in JSON format.

  Returns:
      None
  """
  model = load(model_path) # Load trained model
  test_data = pd.read_json(test_path, lines=True) #  Read test data
  X_test = test_data

  predictions = model.predict(X_test) #  Make prediction

  #  Convert to json and save
  pd.DataFrame(predictions, columns=['Outcome']).to_json(output_path, orient='records', lines=True)


# Main function to process commands from the command line
def main():
  # Create a parser for commands from the command line
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='command')
  
  # Create parser for 'train_models' command
  parser_train_models = subparsers.add_parser('train_models')
  parser_train_models.add_argument('--train_path', type=str)
  parser_train_models.add_argument('--dev_path', type=str)

  # Create parser for 'train' command
  parser_train = subparsers.add_parser('train')
  parser_train.add_argument('--train_path', type=str)
  parser_train.add_argument('--dev_path', type=str)
  parser_train.add_argument('--model_dir', type=str)

  # Create parser for 'predict' command
  parser_predict = subparsers.add_parser('predict')
  parser_predict.add_argument('--model_path', type=str)
  parser_predict.add_argument('--test_path', type=str)
  parser_predict.add_argument('--output_path', type=str)

  # Handles input arguments
  args = parser.parse_args()

  # Choose action based on command
  if args.command == 'train':
    run_train(args.train_path, args.dev_path, args.model_dir)
  elif args.command == 'train_models':
    run_train_models(args.train_path, args.dev_path)
  elif args.command == 'predict':
    run_predict(args.model_path, args.test_path, args.output_path)
  else:
    parser.print_help()
    sys.exit(1)

if __name__ == "__main__":
  main()
