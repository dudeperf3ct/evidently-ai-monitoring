### data preprocessing ###

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("dataset/lending_club_loan_two.csv")
target_names = ["Charged Off", "Fully Paid"]

df["loan_repaid"] = df["loan_status"].map({"Fully Paid": 1, "Charged Off": 0})
df["term"] = df["term"].apply(lambda term: int(term[:3]))
df["earliest_cr_line"] = df["earliest_cr_line"].apply(lambda date: int(date[-4:]))
df["home_ownership"] = df["home_ownership"].replace(["NONE", "ANY"], "OTHER")
df["zipcode"] = df["address"].apply(lambda adress: adress[-5:])
df["mort_acc"] = df["mort_acc"].fillna(df["mort_acc"].median())

# drop features
df = df.drop(
    ["emp_length", "emp_title", "title", "loan_status", "grade", "address", "issue_d"],
    axis=1,
)

df = df.dropna()

# handle categorical features
dummies = pd.get_dummies(
    df[
        [
            "sub_grade",
            "home_ownership",
            "verification_status",
            "application_type",
            "initial_list_status",
            "purpose",
            "zipcode",
        ]
    ],
    drop_first=True,
)
df = pd.concat(
    [
        df.drop(
            [
                "sub_grade",
                "home_ownership",
                "verification_status",
                "application_type",
                "initial_list_status",
                "purpose",
                "zipcode",
            ],
            axis=1,
        ),
        dummies,
    ],
    axis=1,
)
cols = df.columns.tolist()
cols.remove("loan_repaid")
cols.append("loan_repaid")
X = df.drop("loan_repaid", axis=1).values
y = df["loan_repaid"].values

train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)

# ### modelling ###
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_x, train_y)
predictions = (model.predict(val_x) > 0.5).astype("int32")
print(classification_report(val_y, predictions))
print(confusion_matrix(val_y, predictions))

# save training dataset and prediction
assert len(cols) == np.column_stack([train_x, train_y]).shape[1]
train_df = pd.DataFrame(
    np.column_stack([train_x, train_y]),
    columns=cols,
)
train_df["prediction"] = (model.predict(train_x) > 0.5).astype("int32")
train_df[target_names[0]] = model.predict_proba(train_x)[:, 0]
train_df[target_names[1]] = model.predict_proba(train_x)[:, 1]
train_df.to_csv("dataset/train.csv", index=False)

# save validation dataset and prediction
assert len(cols) == np.column_stack([val_x, val_y]).shape[1]
test_df = pd.DataFrame(np.column_stack([val_x, val_y]), columns=cols)
test_df["prediction"] = (model.predict(val_x) > 0.5).astype("int32")
test_df[target_names[0]] = model.predict_proba(val_x)[:, 0]
test_df[target_names[1]] = model.predict_proba(val_x)[:, 1]
test_df.to_csv("dataset/test.csv", index=False)
