"""
Production-ready Feature Engineering Pipeline
for Customer Churn Prediction.

Compatible with:
- sklearn Pipeline
- Model Training
- Testing
- Deployment
"""

from dataclasses import dataclass
from typing import Optional, List, Union
import logging
import time
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """
    Immutable configuration for customer feature engineering.
    
    Attributes:
        new_customer_threshold: Tenure threshold for 'new customer' classification
        min_tenure_value: Minimum tenure value for safe division
        create_tenure_group: Whether to create tenure group categories
        create_cost_ratio: Whether to create cost-related features
        missing_value_strategy: Strategy for handling missing values
        log_feature_stats: Whether to log feature statistics
        log_performance: Whether to log execution time
    """
    new_customer_threshold: int = 12
    min_tenure_value: int = 1
    create_tenure_group: bool = True
    create_cost_ratio: bool = True
    missing_value_strategy: str = "median"
    log_feature_stats: bool = True
    log_performance: bool = True


# Default configuration
DEFAULT_CONFIG = FeatureEngineeringConfig()


# ============================================================
# Custom Exceptions
# ============================================================

class FeatureEngineeringError(Exception):
    """Base exception for feature engineering errors."""
    pass


class ValidationError(FeatureEngineeringError):
    """Raised when input validation fails."""
    pass


# ============================================================
# Main Transformer
# ============================================================

class CustomerFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Customer churn feature engineering transformer.
    
    Compatible with sklearn Pipeline.
    
    Features Created:
        1. is_new_customer: Binary flag for customers with tenure < threshold
        2. tenure_group: Categorical tenure groups (0-6, 6-12, 12-24, 24-48, 48+)
        3. avg_monthly_cost: Average monthly cost over customer lifetime
        4. monthly_to_total_ratio: Ratio of monthly to total charges (if total_charges exists)
    
    Example:
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> 
        >>> pipeline = Pipeline([
        ...     ('features', CustomerFeatureEngineer()),
        ...     ('classifier', RandomForestClassifier())
        ... ])
        >>> 
        >>> pipeline.fit(X_train, y_train)
        >>> predictions = pipeline.predict(X_test)
    """
    
    REQUIRED_COLUMNS = ["tenure", "monthly_charges"]
    
    def __init__(
        self,
        config: Optional[FeatureEngineeringConfig] = None
    ):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration object. Uses DEFAULT_CONFIG if None.
        """
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger(self.__class__.__name__)
        self.created_features: List[str] = []
        self._fit_called = False
    
    # ============================================================
    # sklearn API
    # ============================================================
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> "CustomerFeatureEngineer":
        """
        Fit method required by sklearn.
        
        Validates input and stores feature names for later use.
        
        Args:
            X: Input DataFrame
            y: Optional target variable
            
        Returns:
            self
            
        Raises:
            ValidationError: If input validation fails
        """
        self._validate_input(X)
        
        if y is not None:
            if not isinstance(y, (pd.Series, np.ndarray)):
                raise ValidationError(f"y must be Series or array, got {type(y)}")
            if len(y) != len(X):
                raise ValidationError(
                    f"X and y have different lengths: {len(X)} vs {len(y)}"
                )
        
        # Store feature names for get_feature_names_out
        self._feature_names_in = list(X.columns)
        self._fit_called = True
        
        self.logger.info("Fit completed successfully")
        return self
    
    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        Args:
            X: Input DataFrame with required columns
            
        Returns:
            Transformed DataFrame with engineered features
            
        Raises:
            ValidationError: If input validation fails
        """
        if not self._fit_called:
            self.logger.warning("fit() not called before transform(). Fitting now...")
            self.fit(X)
        
        start_time = time.perf_counter() if self.config.log_performance else None
        
        self.logger.info("Starting feature engineering pipeline...")
        
        # Reset state for each transform
        self.created_features.clear()
        df = X.copy()
        
        # Validate and handle missing values
        self._validate_input(df)
        df = self._handle_missing_values(df)
        
        # Create features
        df = self._create_customer_lifetime_features(df)
        df = self._create_cost_features(df)
        
        # Log summary statistics
        if self.config.log_feature_stats:
            self._log_feature_summary(df)
        
        # Log performance
        if self.config.log_performance and start_time is not None:
            elapsed = time.perf_counter() - start_time
            self.logger.info(
                f"Feature engineering completed in {elapsed:.4f}s. "
                f"Added {len(self.created_features)} features: {self.created_features}"
            )
        else:
            self.logger.info(
                f"Feature engineering completed. "
                f"Added {len(self.created_features)} features: {self.created_features}"
            )
        
        return df
    
    def get_feature_names_out(
        self,
        input_features: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get output feature names for sklearn compatibility.
        
        Args:
            input_features: Input feature names (ignored, used for sklearn compatibility)
            
        Returns:
            List of created feature names
        """
        return self.created_features.copy()
    
    # ============================================================
    # Validation
    # ============================================================
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame.
        
        Args:
            df: Input DataFrame
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(df, pd.DataFrame):
            raise ValidationError(f"Expected DataFrame, got {type(df)}")
        
        if df.empty:
            raise ValidationError("Input DataFrame is empty")
        
        # Check for required columns
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValidationError(
                f"Missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Check data types
        for col in self.REQUIRED_COLUMNS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValidationError(
                    f"Column '{col}' must be numeric, got {df[col].dtype}"
                )
            
            # Check for invalid values
            if (df[col] < 0).any():
                invalid_count = (df[col] < 0).sum()
                raise ValidationError(
                    f"Column '{col}' contains {invalid_count} negative values"
                )
    
    # ============================================================
    # Missing Values
    # ============================================================
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in required columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        cols = self.REQUIRED_COLUMNS
        
        if df[cols].isnull().any().any():
            self.logger.warning("Missing values detected in required columns")
            
            for col in cols:
                if df[col].isnull().all():
                    self.logger.warning(
                        f"All values in '{col}' are missing. Filling with 0."
                    )
                    df[col] = df[col].fillna(0)
                elif df[col].isnull().any():
                    if self.config.missing_value_strategy == "median":
                        fill_value = df[col].median()
                    elif self.config.missing_value_strategy == "mean":
                        fill_value = df[col].mean()
                    elif self.config.missing_value_strategy == "zero":
                        fill_value = 0
                    else:
                        fill_value = df[col].median()  # fallback
                    
                    self.logger.debug(f"Filling {df[col].isnull().sum()} missing values in '{col}' with {fill_value:.2f}")
                    df[col] = df[col].fillna(fill_value)
        
        return df
    
    # ============================================================
    # Customer Lifetime Features
    # ============================================================
    
    def _create_customer_lifetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to customer tenure and lifetime.
        
        Features:
            1. is_new_customer: Binary flag based on tenure threshold
            2. tenure_group: Categorical tenure groups
        """
        self.logger.debug("Creating customer lifetime features...")
        
        # 1. Is new customer (binary)
        feature_name = "is_new_customer"
        df[feature_name] = (
            df["tenure"] < self.config.new_customer_threshold
        ).astype("int8")
        self.created_features.append(feature_name)
        
        # 2. Tenure group (categorical)
        if self.config.create_tenure_group:
            feature_name = "tenure_group"
            df[feature_name] = pd.cut(
                df["tenure"],
                bins=[0, 6, 12, 24, 48, float("inf")],
                labels=["0-6", "6-12", "12-24", "24-48", "48+"],
                right=False
            )
            self.created_features.append(feature_name)
        
        return df
    
    # ============================================================
    # Cost Features
    # ============================================================
    
    def _create_cost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to customer costs.
        
        Features:
            1. avg_monthly_cost: Average monthly cost over lifetime
            2. monthly_to_total_ratio: Ratio of monthly to total charges (conditional)
        """
        if not self.config.create_cost_ratio:
            return df
        
        self.logger.debug("Creating cost features...")
        
        # 1. Average monthly cost over lifetime
        feature_name = "avg_monthly_cost"
        safe_tenure = df["tenure"].clip(lower=self.config.min_tenure_value)
        df[feature_name] = df["monthly_charges"] / safe_tenure
        
        # Handle edge case: tenure = 0
        zero_tenure_mask = df["tenure"] == 0
        if zero_tenure_mask.any():
            df.loc[zero_tenure_mask, feature_name] = df.loc[
                zero_tenure_mask, "monthly_charges"
            ]
        
        self.created_features.append(feature_name)
        
        # 2. Monthly to total ratio (if total_charges exists)
        if "total_charges" in df.columns:
            feature_name = "monthly_to_total_ratio"
            
            # Avoid division by zero
            denominator = df["total_charges"].replace(0, np.nan)
            df[feature_name] = df["monthly_charges"] / denominator
            
            # Replace infinite values with 0
            df[feature_name] = df[feature_name].replace([np.inf, -np.inf], 0)
            df[feature_name] = df[feature_name].fillna(0)
            
            # Clip to reasonable range
            df[feature_name] = df[feature_name].clip(0, 1)
            
            self.created_features.append(feature_name)
            self.logger.debug(
                f"Created '{feature_name}' using 'total_charges' column"
            )
        
        return df
    
    # ============================================================
    # Logging & Reporting
    # ============================================================
    
    def _log_feature_summary(self, df: pd.DataFrame) -> None:
        """
        Log summary statistics of created features.
        
        Args:
            df: DataFrame with created features
        """
        for feature in self.created_features:
            if feature not in df.columns:
                continue
            
            if pd.api.types.is_numeric_dtype(df[feature]):
                self.logger.info(
                    f"  {feature}: "
                    f"mean={df[feature].mean():.3f}, "
                    f"std={df[feature].std():.3f}, "
                    f"min={df[feature].min():.3f}, "
                    f"max={df[feature].max():.3f}, "
                    f"null={df[feature].isnull().sum()}"
                )
            else:
                self.logger.info(
                    f"  {feature}: "
                    f"unique={df[feature].nunique()}, "
                    f"null={df[feature].isnull().sum()}"
                )
    
    # ============================================================
    # Utility Methods
    # ============================================================
    
    def get_feature_names(self) -> List[str]:
        """
        Return list of created feature names.
        
        Returns:
            List of feature names
        """
        return self.created_features.copy()
    
    def get_feature_count(self) -> int:
        """
        Return number of created features.
        
        Returns:
            Number of features
        """
        return len(self.created_features)


# ============================================================
# Convenience Function
# ============================================================

def create_customer_features(
    df: pd.DataFrame,
    config: Optional[FeatureEngineeringConfig] = None
) -> pd.DataFrame:
    """
    Convenience function for feature engineering.
    
    Args:
        df: Input DataFrame with required columns
        config: Optional configuration object
        
    Returns:
        DataFrame with engineered features
        
    Example:
        >>> df = create_customer_features(df)
        >>> print(df.columns)
    """
    engineer = CustomerFeatureEngineer(config=config)
    return engineer.transform(df)


# ============================================================
# Main Execution (Testing)
# ============================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    try:
        from src.data.ingestion import load_data
        
        print("\n" + "="*70)
        print("FEATURE ENGINEERING TEST")
        print("="*70)
        
        # Load data
        df = load_data("Telco.csv")
        print(f"\n✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        print("\n" + "-"*70)
        print("BEFORE FEATURE ENGINEERING")
        print("-"*70)
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 3 rows:")
        print(df.head(3))
        
        # Apply feature engineering with default config
        engineer = CustomerFeatureEngineer()
        df_transformed = engineer.transform(df)
        
        print("\n" + "-"*70)
        print("AFTER FEATURE ENGINEERING")
        print("-"*70)
        print(f"Original columns: {len(df.columns)}")
        print(f"New columns: {len(df_transformed.columns) - len(df.columns)}")
        print(f"Total columns: {len(df_transformed.columns)}")
        
        print("\nNew features created:")
        for feat in engineer.get_feature_names():
            print(f"  - {feat}")
        
        print("\nFirst 3 rows (new features only):")
        new_features = engineer.get_feature_names()
        if new_features:
            print(df_transformed[new_features].head(3))
        
        print("\n" + "="*70)
        print("✓ Feature engineering test completed successfully")
        print("="*70)
        
    except FileNotFoundError:
        print("\n❌ ERROR: Telco.csv not found. Please check the file path.")
    except ValidationError as e:
        print(f"\n❌ Validation Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()