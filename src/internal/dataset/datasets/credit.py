import os
import random
import torch
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from internal.dataset.dataset_utils import (
    bring_examples_to_top,
    corr_ans_text_fn,
    get_loader,
    prediction_set_text_fn,
)
from internal.dataset.datasets.abstract_dataset import CustomDataset
from internal.model.networks import XGBoostClassifier
from substantive.faircp.conformity.utils import ConformalCategory

exclude_keys = {"EducationLevel", "idx", "Age"}


class Credit(CustomDataset):
    def __init__(self):
        self.uses_top_m_labels = False
        self.group_conformal_category = ConformalCategory.CLASS_CONDITIONAL

    def get_data(
        self,
        data_root,
        train_batch_size=10000,
        test_batch_size=256,
        calib_batch_size=256,
        n_calib=0.2,
        n_test=0.1,
        n_calib_val=0.15,
        n_val=0.1,
        **kwargs,
    ):
        print("Loading credit dataset")

        numeric_cols = [
            "MaxBillAmountOverLast6Months",
            "MaxPaymentAmountOverLast6Months",
            "MostRecentBillAmount",
            "MostRecentPaymentAmount",
            "TotalMonthsOverdue",
        ]

        csv_path = os.path.join(data_root, "credit.csv")
        df = pd.read_csv(csv_path)
        df = df.reset_index().rename(columns={"index": "idx"})
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.class_counts = df["EducationLevel"].value_counts().sort_index()

        targetCol = "EducationLevel"
        targetLabelCount = 4
        random_state = random.randint(0, 100)

        train_df, df_conf = train_test_split(
            df,
            test_size=(n_calib + n_test + n_calib_val),
            stratify=df[targetCol],
            random_state=random_state,
        )
        val_df = df_conf.copy()

        # step 2: split df_temp into calib + test + calib_val + val
        calib_subset_df, df_conf = train_test_split(
            df_conf,
            test_size=(n_test + n_calib_val) / (n_calib + n_test + n_calib_val),
            stratify=df_conf[targetCol],
            random_state=random_state,
        )

        calib_val_subset_df, test_subset_df = train_test_split(
            df_conf,
            test_size=(n_test) / (n_test + n_calib_val),
            stratify=df_conf[targetCol],
            random_state=random_state,
        )

        # train_df, val_df = train_test_split(
        #     train_val_df,
        #     test_size=n_val,
        #     stratify=train_val_df[targetCol],
        #     random_state=random_state,
        # )

        print(train_df["EducationLevel"].value_counts())

        # Convert to Dataset objects
        train_subset = Dataset.from_pandas(train_df)
        val_subset = Dataset.from_pandas(val_df)
        calib_subset = Dataset.from_pandas(calib_subset_df)
        test_subset = Dataset.from_pandas(test_subset_df)
        calib_val_subset = Dataset.from_pandas(calib_val_subset_df)

        # attach classes info
        for dataset in [
            train_subset,
            val_subset,
            calib_subset,
            test_subset,
            calib_val_subset,
        ]:
            dataset.classes = [i for i in range(targetLabelCount)]

        # build loaders
        train_loader = get_loader(
            train_subset,
            train_batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = get_loader(
            val_subset, len(val_subset), shuffle=False, drop_last=False
        )
        calib_loader = get_loader(
            calib_subset, calib_batch_size, shuffle=False, drop_last=False
        )
        test_loader = get_loader(
            test_subset, test_batch_size, shuffle=False, drop_last=False
        )
        calib_val_loader = get_loader(
            calib_val_subset, calib_batch_size, shuffle=False, drop_last=False
        )

        print(
            f"Dataset sizes: Train {len(train_loader.dataset)}, Validation {len(val_loader.dataset)}, "
            + f"Calib {len(calib_loader.dataset)}, Test {len(test_loader.dataset)}, "
            + f"Calib_val {len(calib_val_loader.dataset)}."
        )

        return {
            "train": train_loader,
            "val": val_loader,
            "calib": calib_loader,
            "test": test_loader,
            "calib_val": calib_val_loader,
        }

    def get_id2label(self, id=None, return_dict=False):
        label_map = {0: "0", 1: "1", 2: "2", 3: "3"}
        if return_dict:
            return label_map
        return label_map.get(id, None)

    def get_id2group(self, id=None, return_dict=False):
        group_map = {0: "0", 1: "1"}
        if return_dict:
            return group_map
        return group_map.get(id, None)

    def get_model(self, device, train_loader, val_loader, **kwargs):
        model = XGBoostClassifier()
        inputs, labels, _, _ = self.prepare_model_inputs(next(iter(train_loader)))
        model.fit(inputs, labels)

        Xt, Yt, _, _ = self.prepare_model_inputs(next(iter(val_loader)))
        # Predict labels
        predictions = model.predict(Xt)
        print("Accuracy", accuracy_score(Yt, predictions))

        return model

    def prepare_model_inputs(self, data, device=None):
        feature_tensors = []

        for key, value in data.items():
            if key not in exclude_keys:
                feature_tensors.append(
                    value.unsqueeze(1).to(torch.float32)
                )  # keep shape

        # Concatenate features along last dim -> shape [batch_size, num_features]
        input_tensor = torch.cat(feature_tensors, dim=1)

        # Labels and group info
        target = data["EducationLevel"]
        group = data["Age"]

        return input_tensor, target, group, data["idx"]

    def process_dataframe(self, df, loader_dict, k):
        id2label_fa = self.get_id2label(None, True)
        df["label_text"] = df["label"].apply(lambda label: id2label_fa[label])

        group_to_text = self.get_id2group(None, True)
        df["group_text"] = df["group"].apply(lambda group: group_to_text[group])

        df["corr_ans_text"] = df.apply(
            lambda x: corr_ans_text_fn(x["label_text"], x["label"]), axis=1
        )
        df["conformal_marginal_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["conformal_marginal_set"], id2label_fa),
            axis=1,
        )
        df["conformal_conditional_text"] = df.apply(
            lambda x: prediction_set_text_fn(
                x["conformal_conditional_set"], id2label_fa
            ),
            axis=1,
        )
        df["conformal_backward_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["conformal_backward_set"], id2label_fa),
            axis=1,
        )

        df["conformal_clustered_label_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["conformal_clustered_label_set"], id2label_fa),
            axis=1,
        )

        df["conformal_clustered_group_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["conformal_clustered_group_set"], id2label_fa),
            axis=1,
        )
        df["original_label"] = df["label"]

        # Sort dataframe, then put example instances from each class at the top
        min_num = df["label"].value_counts().min()
        df = bring_examples_to_top(df, 4, min_num)

        return df
