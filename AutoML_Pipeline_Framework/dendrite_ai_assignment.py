import json
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, SGDClassifier, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
warnings.filterwarnings("ignore")



# Load JSON configuration
def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config


# Load data from CSV
def load_data(csv_path, target_var):
    data = pd.read_csv(csv_path)
    return data, target_var


# Handling missing features
def handle_features(data, config):
    for feature, details in config['design_state_data']['feature_handling'].items():
        if details['is_selected']:
            impute_with = details.get('feature_details', {}).get('impute_with', 'constant')
            impute_strategy = 'mean' if impute_with == 'Average of values' else 'constant'
            fill_value = details.get('feature_details', {}).get('impute_value', None) if impute_strategy == 'constant' else None
            
            imputer = SimpleImputer(strategy=impute_strategy, fill_value=fill_value)
            try:
                data[feature] = imputer.fit_transform(data[[feature]])
            except Exception as e:
                print(f"Error imputing feature '{feature}': {e}")
                raise
    return data


# Feature reduction
def get_feature_reduction(config):
    feature_reduction_cfg = config['design_state_data']['feature_reduction']
    method = feature_reduction_cfg['feature_reduction_method']

    if method == "PCA":
        num_components = int(feature_reduction_cfg['num_of_features_to_keep'])
        return PCA(n_components=num_components)
    elif method == "Tree-based":
        return SelectFromModel(
            estimator=RandomForestRegressor(
                n_estimators=int(feature_reduction_cfg['num_of_trees']),
                max_depth=int(feature_reduction_cfg['depth_of_trees'])
            )
        )
    else:
        return "passthrough"



def select_models(config, prediction_type):
    model_cfg = config['design_state_data']['algorithms']
    selected_models = []
    param_grid = {}

    # Helper function to add model and its hyperparameters
    def add_model(model, params):
        selected_models.append(model)
        param_grid[model] = params

    
    # Choosing the models based on the prediction type
    if prediction_type == "classification":
        if model_cfg['RandomForestClassifier']['is_selected']:
            add_model(
                RandomForestClassifier(),
                {
                    'model__n_estimators': range(model_cfg['RandomForestClassifier']['min_trees'],
                                                model_cfg['RandomForestClassifier']['max_trees']),
                    'model__max_depth': range(model_cfg['RandomForestClassifier']['min_depth'],
                                            model_cfg['RandomForestClassifier']['max_depth']),
                }
            )
        if model_cfg['GBTClassifier']['is_selected']:
            add_model(
                GradientBoostingClassifier(),
                {
                    'model__n_estimators': model_cfg['GBTClassifier']['num_of_BoostingStages'],
                    'model__learning_rate': [model_cfg['GBTClassifier']['min_stepsize'],
                                            model_cfg['GBTClassifier']['max_stepsize']],
                    'model__max_depth': range(model_cfg['GBTClassifier']['min_depth'],
                                            model_cfg['GBTClassifier']['max_depth']),
                }
            )
        if model_cfg['DecisionTreeClassifier']['is_selected']:
            add_model(
                DecisionTreeClassifier(),
                {
                    'model__max_depth': range(model_cfg['DecisionTreeClassifier']['min_depth'],
                                            model_cfg['DecisionTreeClassifier']['max_depth']),
                    'model__min_samples_leaf': model_cfg['DecisionTreeClassifier']['min_samples_per_leaf'],
                }
            )
        if model_cfg['LogisticRegression']['is_selected']:
            add_model(
                LogisticRegression(),
                {
                    'model__C': [model_cfg['LogisticRegression']['min_regparam'],
                                model_cfg['LogisticRegression']['max_regparam']],
                    'model__max_iter': range(model_cfg['LogisticRegression']['min_iter'],
                                            model_cfg['LogisticRegression']['max_iter']),
                }
            )
        if model_cfg['KNN']['is_selected']:
            add_model(
                KNeighborsClassifier(),
                {
                    'model__n_neighbors': model_cfg['KNN']['k_value'],
                }
            )
        if model_cfg['SVM']['is_selected']:
            add_model(
                SVC(),
                {
                    'model__C': model_cfg['SVM']['c_value'],
                    'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                }
            )
        if model_cfg['SGD']['is_selected']:
            add_model(
                SGDClassifier(),
                {
                    'model__alpha': model_cfg['SGDClassifier']['alpha'],
                    'model__max_iter': model_cfg['SGDClassifier']['max_iter'],
                    'model__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                }
            )
        if model_cfg['neural_network']['is_selected']:
            add_model(
                MLPClassifier(),
                {
                    'model__hidden_layer_sizes': model_cfg['neural_network']['hidden_layer_sizes'],
                    'model__activation': [model_cfg['neural_network']['activation']],
                    'model__solver': [model_cfg['neural_network']['solver']],
                }
            )
        if model_cfg['xg_boost']['is_selected']:
            add_model(
                XGBClassifier(),
                {
                    'model__n_estimators': range(50, model_cfg['xg_boost']['max_num_of_trees']),
                    'model__learning_rate': model_cfg['xg_boost']['learningRate'],
                    'model__max_depth': model_cfg['xg_boost']['max_depth_of_tree'],
                }
            )

    else:  # Regression models
        if model_cfg['RandomForestRegressor']['is_selected']:
            add_model(
                RandomForestRegressor(),
                {
                    'model__n_estimators': range(model_cfg['RandomForestRegressor']['min_trees'],
                                                model_cfg['RandomForestRegressor']['max_trees']),
                    'model__max_depth': range(model_cfg['RandomForestRegressor']['min_depth'],
                                            model_cfg['RandomForestRegressor']['max_depth']),
                }
            )
        if model_cfg['GBTRegressor']['is_selected']:
            add_model(
                GradientBoostingRegressor(),
                {
                    'model__n_estimators': model_cfg['GBTRegressor']['num_of_BoostingStages'],
                    'model__learning_rate': [model_cfg['GBTRegressor']['min_stepsize'],
                                            model_cfg['GBTRegressor']['max_stepsize']],
                    'model__max_depth': range(model_cfg['GBTRegressor']['min_depth'],
                                            model_cfg['GBTRegressor']['max_depth']),
                }
            )
        if model_cfg['DecisionTreeRegressor']['is_selected']:
            add_model(
                DecisionTreeRegressor(),
                {
                    'model__max_depth': range(model_cfg['DecisionTreeRegressor']['min_depth'],
                                            model_cfg['DecisionTreeRegressor']['max_depth']),
                    'model__min_samples_leaf': model_cfg['DecisionTreeRegressor']['min_samples_per_leaf'],
                }
            )
        if model_cfg['LinearRegression']['is_selected']:
            add_model(
                LinearRegression(),
                {
                    'model__fit_intercept': [True, False],
                }
            )
        if model_cfg['RidgeRegression']['is_selected']:
            add_model(
                Ridge(),
                {
                    'model__alpha': [model_cfg['RidgeRegression']['min_regparam'],
                                    model_cfg['RidgeRegression']['max_regparam']],
                    'model__max_iter': range(model_cfg['RidgeRegression']['min_iter'],
                                            model_cfg['RidgeRegression']['max_iter']),
                }
            )
        if model_cfg['LassoRegression']['is_selected']:
            add_model(
                Lasso(),
                {
                    'model__alpha': [model_cfg['LassoRegression']['min_regparam'],
                                    model_cfg['LassoRegression']['max_regparam']],
                }
            )
        if model_cfg['ElasticNetRegression']['is_selected']:
            add_model(
                ElasticNet(),
                {
                    'model__alpha': [model_cfg['ElasticNetRegression']['min_regparam'],
                                    model_cfg['ElasticNetRegression']['max_regparam']],
                    'model__l1_ratio': [model_cfg['ElasticNetRegression']['min_elasticnet'],
                                        model_cfg['ElasticNetRegression']['max_elasticnet']],
                }
            )
        if model_cfg['xg_boost']['is_selected']:
            add_model(
                XGBRegressor(),
                {
                    'model__n_estimators': range(50, model_cfg['xg_boost']['max_num_of_trees']),
                    'model__learning_rate': model_cfg['xg_boost']['learningRate'],
                    'model__max_depth': model_cfg['xg_boost']['max_depth_of_tree'],
                }
            )
        if model_cfg['SGD']['is_selected']:
            add_model(
                SGDRegressor(),
                {
                    'model__alpha': model_cfg['SGDRegressor']['alpha'],
                    'model__max_iter': model_cfg['SGDRegressor']['max_iter'],
                    'model__loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                }
            )
        if model_cfg['SVM']['is_selected']:
            add_model(
                SVR(),
                {
                    'model__C': model_cfg['SVM']['c_value'],
                    'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                }
            )
        if model_cfg['KNN']['is_selected']:
            add_model(
                KNeighborsRegressor(),
                {
                    'model__n_neighbors': model_cfg['KNN']['k_value'],
                }
            )
        if model_cfg['neural_network']['is_selected']:
            add_model(
                MLPRegressor(),
                {
                    'model__hidden_layer_sizes': model_cfg['neural_network']['hidden_layer_sizes'],
                    'model__activation': [model_cfg['neural_network']['activation']],
                    'model__solver': [model_cfg['neural_network']['solver']],
                }
            )

    return selected_models, param_grid


# sklearn pipeline
def run_pipeline(data, target_var, feature_reduction, selected_models, param_grid, prediction_type):
    X = data.drop(columns=[target_var])
    y = data[target_var]

    # Encoding target variable if it's categorical
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Target variable encoded: {le.classes_}")  #displaying the mapping of classes to numbers

    # Identifying categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(exclude=['object']).columns

    # Defining a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numerical_columns),
                      ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)]
    )

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for model in selected_models:
        pipeline_steps = [('preprocessor', preprocessor)]

        if feature_reduction != "passthrough":
            pipeline_steps.append(('feature_reduction', feature_reduction))

        pipeline_steps.append(('model', model))
        pipeline = Pipeline(pipeline_steps)

        model_name = type(model).__name__
        if model not in param_grid:
            print(f"No parameters defined for {model_name}. Skipping...")
            continue

        scoring_metric = 'accuracy' if prediction_type == "classification" else 'r2'

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid[model],
            cv=5,
            scoring=scoring_metric,  # Choosing metric based on prediction type
            error_score='raise'
        )

        try:
            grid_search.fit(X_train, y_train)
        except Exception as e:
            print(f"Failed to fit model {model_name}: {e}")
            continue

        y_pred = grid_search.predict(X_test)

        # Displaying metrics based on prediction type
        print(f"Model: {model_name}")
        print("Best parameters:", grid_search.best_params_)
        if prediction_type == "classification":
            print("Train Accuracy Score:", grid_search.best_score_)
            print("Test Accuracy Score:", accuracy_score(y_test, y_pred))
            print("Classification Report:\n", classification_report(y_test, y_pred))
        else:
            print("Train R2 Score:", grid_search.best_score_)
            print("Test R2 Score:", r2_score(y_test, y_pred))
            print("Test MSE:", mean_squared_error(y_test, y_pred))
        print("\n" + "=" * 50 + "\n")


# Main function
def main():
    # The path to the configuration file that contains all the algorithm settings
    json_path = './data/algoparams_from_ui.json'

    # Path to the dataset file we'll be working with
    csv_path = './data/iris.csv'

    # Load the configuration details from the JSON file
    config = load_config(json_path)

    # Get the target variable (the thing we’re trying to predict) and whether this is a classification or regression task
    target_var = config['design_state_data']['target']['target']
    prediction_type = config['design_state_data']['target']['prediction_type']  # regression or classification
    
    # Loading the dataset to identify which column is the target
    data, target_var = load_data(csv_path, target_var)

    # Applying any preprocessing or transformations to the data as needed
    data = handle_features(data, config)

    # Checking if the configuration specifies any feature reduction techniques (like PCA or skipping it altogether)
    feature_reduction = get_feature_reduction(config)

    # Picking the models and setting up their hyperparameters based on the task type and config settings
    selected_models, param_grid = select_models(config, prediction_type)

    # This is where the magic happens: it runs the pipeline to train, tune, and evaluate the models
    run_pipeline(data, target_var, feature_reduction, selected_models, param_grid, prediction_type)

# to ensures that when the script runs, it starts with the main function
if __name__ == "__main__":
    main()
