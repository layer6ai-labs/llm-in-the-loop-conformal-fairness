import folktables as ft
import numpy as np
import torch
import os
import pandas as pd

from folktables import (
    ACSDataSource,
    BasicProblem,
    adult_filter,
    state_list,
)


def income_breakdowns(x):
    quantiles = np.nanquantile(x, np.linspace(0.0, 1.0, 4 + 1))[1:-1]
    val = x < -10
    for q in quantiles:
        val += (x >= q) * 1
    return val


def schl_transform(x):
    temp = x
    e = 0.000000001
    x = (
        (temp >= 1 - e) * (temp <= 15 + e)
    ) * 1  # No schooling (+ primar _ high school only)
    x += ((temp >= 16 - e) * (temp <= 16 + e)) * 2  # High School - no college
    x += ((temp >= 17 - e) * (temp <= 17 + e)) * 3  # GED - no college
    x += ((temp >= 18 - e) * (temp <= 20 + e)) * 4  # Started college/associates
    x += ((temp >= 21 - e) * (temp <= 21 + e)) * 5  # Bachelor's Degree
    x += ((temp >= 22 - e) * (temp <= 24 + e)) * 6  # Grad School/Professional Degree
    # breakpoint()
    return x - 1


def schl_filter(data):
    """Mimic the filters in place for Adult data.

    Adult documentation notes: Extraction was done by Barry Becker from
    the 1994 Census database. A set of reasonably clean records was extracted
    using the following conditions:
    ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    --Adapted from Folktable library/code
    """
    df = data
    df = df[df["AGEP"] > 16]
    df = df[df["PINCP"] > 100]
    df = df[df["WKHP"] > 0]
    df = df[df["PWGTP"] >= 1]
    df = df[df["PWGTP"] >= 1]
    df = df[df["SCHL"].notna()]  # Added this line to allow it to work
    return df


ACSEducation = BasicProblem(
    features=[
        "AGEP",
        "JWMNP",  # Travel time to work
        "MAR",
        "SEX",
        "DIS",  # Disability recode
        "ESP",  # Employment status of parents
        "MIG",  # Mobility status
        "RELSHIPP",
        "RAC1P",
        "PUMA",  # Public use microdata area code
        "STATE",  # STATE State code
        "CIT",  # Citizenship
        "OCCP",
        "COW",  # Class of worker
        "JWTRNS",  # Means of transportation to work
        "POWPUMA",  # Place of work PUMA based
        "POVPIP",  # Income-to-poverty ratio
    ],
    target="SCHL",
    target_transform=schl_transform,
    group="RAC1P",
    preprocess=schl_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSIncome = ft.BasicProblem(
    features=[
        "AGEP",
        "COW",  # Class of worker
        "SCHL",  # Educational attainment
        "MAR",  # Marital status
        "OCCP",  # Occupation recode
        "POBP",  # Place of birth
        "ESP",
        "RELSHIPP",  # RELSHIPP Relationship
        "WKHP",  # Usual hours worked
        "POWPUMA",  # Place of work PUMA based
        "SEX",
        "RAC1P",
    ],
    target="PINCP",
    target_transform=lambda x: x,
    group="RAC1P",
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

year = "2023"
horizon = "1-Year"  # or 5-Year
method = "person"  # Or 'household'


def download_acs():
    data_source = ACSDataSource(survey_year=year, horizon=horizon, survey=method)

    edu_feat_list = []
    edu_label_list = []
    edu_sens_list = []
    income_feat_list = []
    income_label_list = []
    income_sens_list = []

    for state in state_list:
        st_data = data_source.get_data(states=[state], download=True)
        edu_features, edu_labels, edu_sens = ACSEducation.df_to_numpy(st_data)
        income_features, income_labels, income_sens = ACSIncome.df_to_numpy(st_data)
        edu_feat_list.append(torch.tensor(edu_features, dtype=torch.int64))
        income_feat_list.append(torch.tensor(income_features, dtype=torch.int64))
        edu_label_list.append(torch.tensor(edu_labels, dtype=torch.int64))
        income_label_list.append(torch.tensor(income_labels, dtype=torch.int64))
        edu_sens_list.append(
            torch.tensor(edu_sens, dtype=torch.int64) - 1
        )  # subtract 1 since groups are 1,2, ..., 9
        income_sens_list.append(torch.tensor(income_sens, dtype=torch.int64) - 1)

    edu_features = torch.cat(edu_feat_list, dim=0)
    edu_labels = torch.cat(edu_label_list)
    edu_sens = torch.cat(edu_sens_list)
    income_features = torch.cat(income_feat_list, dim=0)
    income_labels = torch.cat(income_label_list)
    income_sens = torch.cat(income_sens_list)

    file_path = os.path.join("data", "acs")
    os.makedirs(file_path, exist_ok=True)

    pd.DataFrame(edu_features.numpy(), columns=ACSEducation.features).to_csv(
        os.path.join(file_path, "edu_features.csv"), index=False
    )
    pd.DataFrame(edu_labels.numpy(), columns=["edu_bracket"]).to_csv(
        os.path.join(file_path, "edu_labels.csv"), index=False
    )
    pd.DataFrame(edu_sens.numpy(), columns=["race"]).to_csv(
        os.path.join(file_path, "edu_sens.csv"), index=False
    )
    pd.DataFrame(income_features.numpy(), columns=ACSIncome.features).to_csv(
        os.path.join(file_path, "income_raw_features.csv"), index=False
    )
    pd.DataFrame(income_labels.numpy(), columns=["income_bracket"]).to_csv(
        os.path.join(file_path, "income_raw_labels.csv"), index=False
    )
    pd.DataFrame(income_sens.numpy(), columns=["race"]).to_csv(
        os.path.join(file_path, "income_raw_sens.csv"), index=False
    )

    features = pd.read_csv(os.path.join(file_path, "income_raw_features.csv"))
    sens = pd.read_csv(os.path.join(file_path, "income_raw_sens.csv"))

    raw_labels = pd.read_csv(os.path.join(file_path, "income_raw_labels.csv"))

    bins = [104, 9000, 20000, 30000, 38800, 48450, 60000, 75000, 96900, 140000, np.inf]
    raw_amounts = pd.to_numeric(raw_labels["income_bracket"], errors="coerce")
    raw_labels["income_bracket"] = pd.cut(
        raw_amounts,
        bins=bins,
        labels=list(range(10)),
        right=True,  
        include_lowest=True
    ).astype("Int64")

    if raw_labels["income_bracket"].isna().any():
        bad = raw_amounts[raw_labels["income_bracket"].isna()].head(10).tolist()
        raise ValueError(f"Some income values could not be binned. Examples: {bad}")

    raw_labels.to_csv(os.path.join(file_path, "income_labels.csv"), index=False)

    mapping = {0: 0, 8: 4, 1: 1, 5: 3, 7: 2, 6: 2, 4: 2, 3: 2, 2: 2}
    mapping_raw = {1: 0, 9: 4, 2: 1, 6: 3, 8: 2, 7: 2, 5: 2, 4: 2, 3: 2}

    sens["race"] = sens["race"].map(mapping)
    features["RAC1P"] = features["RAC1P"].map(mapping_raw)

    sens.to_csv(os.path.join(file_path, "income_sens.csv"), index=False)
    features.to_csv(os.path.join(file_path, "income_features.csv"), index=False)

if __name__ == "__main__":
    download_acs()