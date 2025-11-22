import os
import sys
import mlflow
import dagshub

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


# Initialize DagsHub MLflow
dagshub.init(repo_owner="spaceus9", repo_name="network-security-mlops", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/spaceus9/network-security-mlops.mlflow")
mlflow.set_experiment("network-security-mlops")


class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):

        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def track_mlflow(self, model, metrics):
        """Logs metrics + model to MLflow safely (DagsHub compatible)."""

        with mlflow.start_run():

            mlflow.log_metric("f1_score", metrics.f1_score)
            mlflow.log_metric("precision", metrics.precision_score)
            mlflow.log_metric("recall", metrics.recall_score)

            # DagsHub does NOT support log_model(), so save manually
            model_dir = f"saved_model_{mlflow.active_run().info.run_id}"
            mlflow.sklearn.save_model(model, model_dir)
            mlflow.log_artifact(model_dir)


    def train_model(self, X_train, y_train, X_test, y_test):

        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(),
            "AdaBoost": AdaBoostClassifier()
        }

        params = {
            "Random Forest": {'n_estimators': [32, 64, 128]},
            "Decision Tree": {'criterion': ['gini', 'entropy']},
            "Gradient Boosting": {'learning_rate': [.1, .01], 'n_estimators': [64, 128]},
            "Logistic Regression": {},
            "AdaBoost": {'n_estimators': [64, 128]}
        }

        model_report = evaluate_models(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            models=models, params=params
        )

        # best model name
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]

        # TRAIN METRICS
        y_train_pred = best_model.predict(X_train)
        train_metrics = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        self.track_mlflow(best_model, train_metrics)

        # TEST METRICS
        y_test_pred = best_model.predict(X_test)
        test_metrics = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        self.track_mlflow(best_model, test_metrics)

        # Save preprocessor + model
        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
        final_model = NetworkModel(preprocessor=preprocessor, model=best_model)

        os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
        save_object(self.model_trainer_config.trained_model_file_path, obj=final_model)

        return ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metrics,
            test_metric_artifact=test_metrics
        )

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
