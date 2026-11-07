from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


TARGET = "Risk"

NUMERIC = [
    "Age",
    "Job",
    "Credit amount",
    "Duration",
]

CATEGORICAL = [
    "Sex",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Purpose",
]

FEATURES = NUMERIC + CATEGORICAL

ACTIONABLE = [
    "Credit amount",
    "Duration",
    "Saving accounts",
    "Checking account",
]


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    return model