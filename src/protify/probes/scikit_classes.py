import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Any, Tuple, Optional

try:
    from metrics import get_regression_scorer, get_classification_scorer, classification_scorer, regression_scorer
    from utils import print_message
    from seed_utils import get_global_seed
except ImportError:
    from ..metrics import get_regression_scorer, get_classification_scorer, classification_scorer, regression_scorer
    from ..utils import print_message
    from ..seed_utils import get_global_seed

from .lazy_predict import (
    LazyRegressor,
    LazyClassifier,
    CLASSIFIER_DICT,
    REGRESSOR_DICT,
    ALL_MODEL_DICT
)
from .scikit_hypers import HYPERPARAMETER_DISTRIBUTIONS


class ScikitArguments:
    """
    Combined arguments class for scikit-learn model training and tuning.
    """
    def __init__(
        self,
        # Tuning arguments
        n_iter: int = 100,
        cv: int = 3,
        random_state: Optional[int] = None,
        # Specific model arguments (optional)
        model_name: Optional[str] = None,
        model_args: Optional[Dict[str, Any]] = None,
        production_model: bool = False,
        **kwargs,
    ):
        # Tuning arguments
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state or get_global_seed()
        
        # Specific model arguments
        self.model_name = model_name
        self.model_args = model_args if model_args is not None else {}
        self.production_model = production_model


class ModelResults:
    def __init__(
        self,
        initial_scores: Optional[pd.DataFrame],
        best_model_name: str,
        best_params: Optional[Dict[str, Any]],
        final_scores: Dict[str, float],
        best_model: Any
    ):
        self.initial_scores = initial_scores
        self.best_model_name = best_model_name
        self.best_params = best_params
        self.final_scores = final_scores
        self.best_model = best_model

    def __str__(self) -> str:
        return (
            f"Best Model: {self.best_model_name}\n"
            f"Best Parameters: {self.best_params}\n"
            f"Final Scores: {self.final_scores}"
        )


class ScikitProbe:
    """
    A class for finding and tuning the best scikit-learn models for a given dataset.
    """
    def __init__(self, args: ScikitArguments):
        self.args = args
        self.n_jobs = 1
    
    def _tune_hyperparameters(
        self,
        model_class: Any,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        custom_scorer: Any,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        """
        param_distributions = HYPERPARAMETER_DISTRIBUTIONS.get(model_name, {})
        if not param_distributions:
            return model_class(), {}

        random_search = RandomizedSearchCV(
            model_class(),
            param_distributions=param_distributions,
            n_iter=self.args.n_iter,
            scoring=custom_scorer,
            cv=self.args.cv,
            random_state=self.args.random_state,
            n_jobs=self.n_jobs
        )
        
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_, random_search.best_params_

    def find_best_regressor(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ModelResults:
        """
        Find the best regression model through lazy prediction and hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            ModelResults object containing all results and the best model
        """
        # Initial lazy prediction
        print_message(f"Initial lazy prediction started")
        regressor = LazyRegressor(
            verbose=0,
            ignore_warnings=False,
            custom_metric=regression_scorer()
        )
        initial_scores = regressor.fit(X_train, X_test, y_train, y_test)
        if isinstance(initial_scores, Tuple):
            initial_scores = initial_scores[0]
        
        # Get best model name and class
        best_model_name = initial_scores.index[0]
        best_model_class = regressor.models[best_model_name].named_steps['regressor'].__class__
        print_message(f"Best model name: {best_model_name}")
        print_message(f"Best model class: {best_model_class}")
        print_message(f"Initial scores: \n{initial_scores}")

        print_message(f"Tuning hyperparameters")
        # Tune hyperparameters
        scorer = get_regression_scorer()
        best_model, best_params = self._tune_hyperparameters(
            best_model_class,
            best_model_name,
            X_train,
            y_train,
            scorer,
        )
        
        # Get final scores with tuned model
        best_model.fit(X_train, y_train)
        final_scores = scorer(best_model, X_test, y_test)
        print_message(f"Final scores: {final_scores}")
        print_message(f"Best params: \n{best_params}")

        return ModelResults(
            initial_scores=initial_scores,
            best_model_name=best_model_name,
            best_params=best_params,
            final_scores=final_scores,
            best_model=best_model
        )

    def find_best_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ModelResults:
        """
        Find the best classification model through lazy prediction and hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            ModelResults object containing all results and the best model
        """
        # Initial lazy prediction
        print_message(f"Initial lazy prediction started")
        classifier = LazyClassifier(
            verbose=0,
            ignore_warnings=False,
            custom_metric=classification_scorer()
        )
        initial_scores = classifier.fit(X_train, X_test, y_train, y_test)
        if isinstance(initial_scores, Tuple):
            initial_scores = initial_scores[0]

        # Get best model name and class
        best_model_name = initial_scores.index[0]
        best_model_class = classifier.models[best_model_name].named_steps['classifier'].__class__
        print_message(f"Best model name: {best_model_name}")
        print_message(f"Best model class: {best_model_class}")
        print_message(f"Initial scores: \n{initial_scores}")

        print_message(f"Tuning hyperparameters")
        # Tune hyperparameters
        scorer = get_classification_scorer()
        best_model, best_params = self._tune_hyperparameters(
            best_model_class,
            best_model_name,
            X_train,
            y_train,
            scorer,
        )
        
        # Get final scores with tuned model
        best_model.fit(X_train, y_train)
        final_scores = scorer(best_model, X_test, y_test)
        print_message(f"Final scores: {final_scores}")
        print_message(f"Best params: \n{best_params}")

        return ModelResults(
            initial_scores=initial_scores,
            best_model_name=best_model_name,
            best_params=best_params,
            final_scores=final_scores,
            best_model=best_model
        )

    def run_specific_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_results: Optional[ModelResults] = None,
    ) -> ModelResults:
        """
        Run a specific model with given arguments or based on a previous ModelResults.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_valid: Validation features
            y_valid: Validation targets
            X_test: Test features
            y_test: Test targets
            model_results: Optional ModelResults from find_best_classifier or find_best_regressor
                          If provided, will use the best model type and parameters from it
            
        Returns:
            ModelResults object containing results and the model
        """
        print_message("Running specific model")
        if self.args.production_model:
            print_message(f"Running in production mode, train and validation are combined")
            X_train = np.concatenate([X_train, X_valid])
            y_train = np.concatenate([y_train, y_valid])

        # If model_results is provided, use its best model type and parameters
        if model_results is not None:
            model_name = model_results.best_model_name
            model_params = model_results.best_params if model_results.best_params is not None else {}
            
            # Determine if it's a classifier or regressor
            if model_name in CLASSIFIER_DICT:
                scorer = get_classification_scorer()
            elif model_name in REGRESSOR_DICT:
                scorer = get_regression_scorer()
            else:
                raise ValueError(f"Model {model_name} not supported")
                
            # Get the model class
            model_class = ALL_MODEL_DICT[model_name]
            
            # Create and train the model with the best parameters
            cls = model_class(**model_params)
            print_message(f"Training model {cls}")
            cls.fit(X_train, y_train)
            print_message(f"Model trained")
            final_scores = scorer(cls, X_test, y_test)
            print_message(f"Final scores: {final_scores}")

            return ModelResults(
                initial_scores=None,
                best_model_name=model_name,
                best_params=model_params,
                final_scores=final_scores,
                best_model=cls
            )
        
        # Original functionality when no model_results is provided
        elif self.args.model_name is not None:
            model_name = self.args.model_name
            if model_name in CLASSIFIER_DICT:
                scorer = get_classification_scorer()
            elif model_name in REGRESSOR_DICT:
                scorer = get_regression_scorer()
            else:
                raise ValueError(f"Model {model_name} not supported")

            model_class = ALL_MODEL_DICT[model_name]
            cls = model_class(**self.args.model_args)
            cls.fit(X_train, y_train)
            final_scores = scorer(cls, X_test, y_test)
            
            return ModelResults(
                initial_scores=None,
                best_model_name=model_name,
                best_params=None,
                final_scores=final_scores,
                best_model=cls
            )
        else:
            raise ValueError("Either model_name must be specified in args or model_results must be provided")
