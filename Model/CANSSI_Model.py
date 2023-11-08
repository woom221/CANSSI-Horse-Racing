import pandas as pd
import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time

random.seed(998905029)

# DataFrame Display Setting
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)

# Input column used for preprocessing stage
input_columns_pre = ["DamID", "Distance", "FrontShoes", "Gender", "HindShoes", "HorseAge", "HorseID", "JockeyID",
                     "RaceGroup", "RacingSubType", "Saddlecloth", "SexRestriction", "SireID", "StartType",
                     "StartingLine", "Surface", "TrainerID", "WetnessScale", "RaceStartTime", "RaceID", "WeightCarried"]

# Input column used for general analysis
input_columns = ["DamID", "Distance", "FrontShoes", "Gender", "HindShoes", "HorseAge", "HorseID", "JockeyID",
                 "RaceGroup", "RacingSubType", "Saddlecloth", "SexRestriction", "SireID", "StartType",
                 "StartingLine", "Surface", "TrainerID", "WetnessScale", "WeightCarried"]

# Full standard column used for output data
full_standard_columns = ["AgeRestriction", "Barrier", "ClassRestriction", "CourseIndicator",
                         "DamID", "Distance", "FoalingCountry", "FoalingDate",
                         "FrontShoes", "Gender", "GoingAbbrev", "GoingID",
                         "HandicapDistance", "HandicapType", "HindShoes", "HorseAge",
                         "HorseID", "JockeyID", "RaceGroup", "RaceID", "RacePrizemoney",
                         "RaceStartTime", "RacingSubType", "Saddlecloth", "SexRestriction",
                         "SireID", "StartType", "StartingLine", "Surface", "TrackID", "TrainerID",
                         "WeightCarried", "WetnessScale"]

# Performance column used for generating winning probability
performance_columns = ["BeatenMargin", "Prizemoney", "PIRPosition",
                       "PriceSP", "NoFrontCover", "WideOffRail",
                       "Disqualified", "FinishPosition"]

# Total columns
total_columns = input_columns + performance_columns


# Function used for generating winning category
# We consider horses who earned prize as 'Winning'
# notice that we check disqualified == true for winning because this column
# values are flipped for analysis purpose later. Thus, disqualified == true is
# the same as false in original sense
def winning_category(row):
    if row['Disqualified'] == True and row['Prizemoney'] > 0:
        result = 1
    else:
        result = 0
    return result


# Load the data
def load_data():
    pd.set_option('mode.chained_assignment', None)
    data = pd.read_parquet('trots_2013-2022.parquet', engine='fastparquet')
    data = data.sort_values(by="RaceStartTime")
    # This will be used for output purpose
    output_data = data.copy(deep=True)
    output_data = output_data[full_standard_columns]


    # Drop the unnecessary columns
    data.drop(columns=["AgeRestriction", "Barrier", "ClassRestriction", "FoalingCountry", "FoalingDate", "GoingAbbrev",
                       "GoingID", "HandicapDistance", "HandicapType", "RaceOverallTime", "RacePrizemoney",
                       "TrackID"], inplace=True)

    # standard df only consists of standard (given) columns except several
    standard_df = data[input_columns_pre + ['Disqualified', 'Prizemoney']]
    # overall df consists of every column
    overall_df = data[total_columns]

    # flip disqualified for StandardScalar
    overall_df['Disqualified'] = ~overall_df['Disqualified']

    # Add winning category
    overall_df['WinningCategory'] = overall_df.apply(winning_category, axis=1)
    standard_df['WinningCategory'] = standard_df.apply(winning_category, axis=1)


    # Change finish position for standard scalar
    overall_df['FinishPosition'] = overall_df['FinishPosition'].apply(lambda x: x.strip())
    # Penalize for letter situations in finish position
    # flip the numbers if we know that lower performs better from pre-analysis
    overall_df['BeatenMargin'] = overall_df['BeatenMargin'].apply(lambda x: 1/(x + 1))
    overall_df['FinishPosition'] = overall_df['FinishPosition'].apply(lambda x: 1/int(x) if x.isdigit() else -1)
    overall_df['PriceSP'] = overall_df['PriceSP'].apply(lambda x: 1/x if x > 0 else 0)
    overall_df['WideOffRail'] = overall_df['WideOffRail'].apply(lambda x: 1/x if x > 0 else 0)
    # build the regressor and scale the predictors using standard scaler
    logit = LogisticRegression(random_state=0, class_weight='balanced', solver='newton-cg', max_iter=300)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(overall_df[performance_columns])
    # fit logistic regression to generate winning probability
    logit.fit(X_sc, overall_df[['WinningCategory']].values.ravel())
    standard_df.insert(0, "WinningProbability", logit.predict_proba(overall_df[performance_columns])[:, 1])
    # scale the probability by Prizemoney. We add constants to avoid 0 probability
    standard_df['WinningProbability'] = (standard_df['WinningProbability'] + 0.1) * (standard_df["Prizemoney"] + 100)
    standard_df.drop(columns=['Disqualified', 'Prizemoney'], inplace=True)
    standard_df['TotalProbability'] = standard_df.groupby(['RaceID'])['WinningProbability'].transform('sum')
    standard_df.drop_duplicates()
    standard_df['WinningProbability'] = (standard_df['WinningProbability']) / standard_df['TotalProbability']
    # divide the dataset into test and training
    train_data = standard_df[(standard_df['RaceStartTime'] < '2021-11-01 00:00:00')]
    test_data = standard_df[(standard_df['RaceStartTime'] >= '2021-11-01 00:00:00')]
    output_data = output_data[(output_data['RaceStartTime'] >= '2021-11-01 00:00:00')]
    # remove introduced/unnecessary columns
    train_data.drop(columns=["RaceStartTime", 'TotalProbability'], inplace=True)
    test_data.drop(columns=["RaceStartTime", 'TotalProbability'], inplace=True)
    return train_data, test_data, output_data


def build_model(train_data, test_data, output_data):
    # check that test_data is basically the same as output data
    assert(test_data['RaceID'].equals(output_data['RaceID']))

    # drop unnecessary columns for predictions
    train_data_new = train_data.drop(columns=['RaceID', 'WinningCategory'], inplace=False)
    test_data_new = test_data.drop(columns=['RaceID', 'WinningCategory'], inplace=False)

    # create pipelines
    cat_selector = make_column_selector(dtype_include=[object])
    num_selector = make_column_selector(dtype_include=np.number)
    cat_ohe_processor = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=1000)
    # cat_label_processor = LabelEncoder()
    num_processor = make_pipeline(StandardScaler(), SimpleImputer(strategy="mean", add_indicator=True))
    ohe_preprocessor = make_column_transformer((num_processor, num_selector), (cat_ohe_processor, cat_selector))
    # label_preprocessor = make_column_transformer((num_processor, num_selector), (cat_label_processor, cat_selector))
    rf_pipeline = make_pipeline(ohe_preprocessor, RandomForestRegressor(n_estimators=50, random_state=0, min_samples_split=0.001, min_samples_leaf=0.0003, n_jobs=3))
    nn_pipeline = make_pipeline(ohe_preprocessor, MLPRegressor(random_state=0, max_iter=5000))
    # gbdt_pipeline = make_pipeline(ohe_preprocessor, HistGradientBoostingRegressor(random_state=0))

    # we used random forest and neural network with reasonable hyperparameters
    estimators = [
        ("Random Forest", rf_pipeline),
        ("Neural Network", nn_pipeline),
        # ("Gradient Boosting", gbdt_pipeline),
    ]

    # perform stacking regression
    stacking_regressor = StackingRegressor(estimators=estimators)
    X = train_data_new[input_columns]
    y = train_data_new['WinningProbability']
    # nn_pipeline.fit(X, y)
    stacking_regressor.fit(X, y)
    # rf_pipeline.fit(X, y)
    target_X = test_data_new[input_columns]
    y_pred = stacking_regressor.predict(target_X)
    # y_pred = nn_pipeline.predict(target_X)
    # y_pred = rf_pipeline.predict(target_X)
    # target_Y = np.array(test_data['WinningProbability'].tolist())
    test_data.insert(0, "PredictedProbability", y_pred)
    # calculate the standardized probability for each RaceID
    test_data['TotalProbability'] = test_data.groupby(['RaceID'])['PredictedProbability'].transform('sum')
    test_data.drop_duplicates()
    test_data['PredictedProbability'] = (test_data['PredictedProbability']) / test_data['TotalProbability']
    # calculate MSE
    mse = np.sum(np.square(test_data['PredictedProbability'] - test_data['WinningProbability'])) / len(test_data)
    # insert the calculated winprobability to our output data
    output_data['winprobability'] = test_data['PredictedProbability']

    return mse, output_data


if __name__ == "__main__":
    # load data
    train_data, test_data, output_data = load_data()
    initial_time = time.time()
    # get results
    mse, result = build_model(train_data, test_data, output_data)
    final_time = time.time()
    print(f'Computation took {round((final_time - initial_time)/60, 2)} minutes')
    print(f'Mean Squared Error is: {mse}')
    result.to_parquet('CANSSI forecast.parquet', index=False)
    print("Forecast has been saved as CANSSI forecast.parquet")
