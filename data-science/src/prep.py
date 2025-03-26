# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")  # Create an ArgumentParser object
    parser.add_argument("--raw_data", type=str, help="Path to raw data")  # Specify the type for raw data (str)
    parser.add_argument("--train_data", type=str, help="Path to train dataset")  # Specify the type for train data (str)
    parser.add_argument("--test_data", type=str, help="Path to test dataset")  # Specify the type for test data (str)
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")  # Specify the type (float) and default value (0.2) for test-train ratio
    args = parser.parse_args()

    return args

def main(args):  # Write the function name for the main data preparation logic
    '''Read, preprocess, split, and save datasets'''

    # Reading Data
    df = pd.read_csv(args.raw_data)

    # ------- WRITE YOUR CODE HERE -------

    # Step 1: Perform label encoding to convert categorical features into numerical values for model compatibility.  
    labelencoder = LabelEncoder()  # Create a LabelEncoder object
    df['Segment'] = labelencoder.fit_transform(df['Segment'])  # Fit and transform the 'Segment' column

    # Step 2: Split the dataset into training and testing sets using train_test_split with specified test size and random state.  
    train_data, test_data = train_test_split(df, test_size=args.train_test_ratio, random_state=42)  # Split the dataset into training and testing sets

    # Step 3: Save the training and testing datasets as CSV files in separate directories for easier access and organization.
    # Create directories for train and test datasets
    os.makedirs(args.train_data, exist_ok=True)  # Create the train_data directory
    os.makedirs(args.test_data, exist_ok=True)  # Create the test_data directory

    # Save the training and testing datasets as CSV files
    train_data.to_csv(os.path.join(args.train_data, 'data.csv'), index=False)  # Save the train_data as a CSV file
    test_data.to_csv(os.path.join(args.test_data, 'data.csv'), index=False)  # Save the test_data as a CSV file

    # Step 4: Log the number of rows in the training and testing datasets as metrics for tracking and evaluation.
    mlflow.log_metric(f"Number of rows in train data: {train_data.shape[0]}") # Log the number of rows in the train_data
    mlflow.log_metric(f"Number of rows in test data: {test_data.shape[0]}") # Log the number of rows in the test_data


if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()  # Call the function to parse arguments

    lines = [
        f"Raw data path: {args.data}",  # Print the raw_data path
        f"Train dataset output path: {args.train_test_ratio}",  # Print the train_data path
        f"Test dataset path: {args.train_data}",  # Print the test_data path
        f"Test-train ratio: {args.test_data}",  # Print the test_train_ratio
    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()
