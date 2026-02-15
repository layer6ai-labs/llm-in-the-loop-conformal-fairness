import os
import pandas as pd
from datasets import Dataset
import torch

from internal.dataset.dataset_utils import (
    bring_examples_to_top,
    corr_ans_text_fn,
    get_loader,
    prediction_set_text_fn,
    stratified_sample_df,
)
from internal.dataset.datasets.abstract_dataset import CustomDataset
from substantive.faircp.conformity.utils import ConformalCategory
from internal.model.networks import XGBoostClassifier
from sklearn.metrics import accuracy_score


def my_collate(batch):
    features = [item["feature"] for item in batch]  # list of lists
    labels = [item["label"] for item in batch]
    groups = [item["group"] for item in batch]
    index = [item["index"] for item in batch]

    # convert to tensors
    features = torch.tensor(
        features, dtype=torch.long
    )  # shape: (batch_size, feature_length)
    labels = torch.tensor(labels, dtype=torch.long)
    groups = torch.tensor(groups, dtype=torch.long)
    index = torch.tensor(index, dtype=torch.long)

    return {"index": index, "feature": features, "label": labels, "group": groups}


class ACSEducation(CustomDataset):
    def __init__(self):
        self.uses_top_m_labels = False
        self.group_conformal_category = ConformalCategory.CLASS_CONDITIONAL

    def get_data(
        self,
        data_root,
        calib_batch_size=256,
        calib_val_batch_size=256,
        train_batch_size=10000,
        test_batch_size=256,
        n_calib=5000,
        n_train=10000,
        n_val=4000,
        n_calib_val=4000,
        n_test=2000,
        **kwargs,
    ):
        print("Loading ACS Education dataset")

        feature_path = os.path.join(data_root, "edu_features.csv")
        features = pd.read_csv(feature_path)
        label_path = os.path.join(data_root, "edu_labels.csv")
        labels = pd.read_csv(label_path)
        labels = labels.rename(columns={"edu_bracket": "label"})
        group_path = os.path.join(data_root, "edu_sens.csv")
        groups = pd.read_csv(group_path)
        groups = groups.rename(columns={"race": "group"})

        assert len(features) == len(labels) == len(groups), (
            f"ACS CSV files not aligned!{len(features)}-{len(labels)}-{len(groups)}"
        )

        features["feature"] = features.values.tolist()
        features["index"] = range(len(features))
        df = pd.concat([features[["index", "feature"]], labels, groups], axis=1)
        label_count = df["label"].nunique()

        # assume there's a column named "label"
        # stratified split (replace with your util if available)
        d_conf_df, train_val_df = stratified_sample_df(
            df,
            col="label",
            n_samples=(n_calib + n_test + n_calib_val) // label_count,
            return_remaining=True,
        )

        train_df, val_df = stratified_sample_df(
            train_val_df,
            col="label",
            n_samples=n_train // label_count,
            return_remaining=True,
        )
        val_df, _ = stratified_sample_df(
            val_df,
            col="label",
            n_samples=n_val // label_count,
            return_remaining=True,
        )

        calib_subset_df, d_conf_remain = stratified_sample_df(
            d_conf_df,
            col="label",
            n_samples=n_calib // label_count,
            return_remaining=True,
        )
        test_subset_df, d_conf_remain = stratified_sample_df(
            d_conf_remain,
            col="label",
            n_samples=n_test // label_count,
            return_remaining=True,
        )
        calib_val_subset_df, _ = stratified_sample_df(
            d_conf_remain,
            col="label",
            n_samples=n_calib_val // label_count,
            return_remaining=True,
        )

        # Convert to Dataset objects
        train_subset = Dataset.from_pandas(train_df)
        val_subset = Dataset.from_pandas(val_df)
        calib_subset = Dataset.from_pandas(calib_subset_df)
        test_subset = Dataset.from_pandas(test_subset_df)
        calib_val_subset = Dataset.from_pandas(calib_val_subset_df)

        # attach classes info
        n_classes = df["label"].nunique()
        for dataset in [
            train_subset,
            val_subset,
            calib_subset,
            test_subset,
            calib_val_subset,
        ]:
            dataset.classes = [i for i in range(n_classes)]

        # build loaders
        train_loader = get_loader(
            train_subset,
            train_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=my_collate,
        )
        val_loader = get_loader(
            val_subset,
            len(val_subset),
            shuffle=False,
            drop_last=False,
            collate_fn=my_collate,
        )
        calib_loader = get_loader(
            calib_subset,
            calib_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=my_collate,
        )
        test_loader = get_loader(
            test_subset,
            test_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=my_collate,
        )
        calib_val_loader = get_loader(
            calib_val_subset,
            calib_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=my_collate,
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
        label_map = {
            0: "No schooling (+ primar _ high school only)",
            1: "High School - no college",
            2: "GED - no college",
            3: "Started college/associates",
            4: "Bachelor's Degree",
            5: "Grad School/Professional Degree",
        }
        if return_dict:
            return label_map
        return label_map.get(id, None)

    def get_id2group(self, id=None, return_dict=False):
        group_map = {
            0: "White alone",
            4: "Two or More Races",
            1: "Black or African American alone",
            3: "Asian alone",
            2: "All Other Races (Aggregated)",
        }
        if return_dict:
            return group_map
        return group_map.get(id, None)

    def get_model(self, device, train_loader, val_loader, **kwargs):
        model = XGBoostClassifier(
            n_estimators=5000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="multi:softprob",
            eval_metric="auc",
            random_state=42,
            device=device,
            early_stopping_rounds=25,
        )
        model_checkpoint: str | bool | None = kwargs["model_checkpoint"]
        if model_checkpoint is not None:
            if model_checkpoint is True:
                model_checkpoint = "logs/model_checkpoints/AcsEducation.ubj"
            if os.path.exists(model_checkpoint):
                print(f"Loading model from {model_checkpoint}")
                model.load_model(model_checkpoint)
                return model

        inputs, labels, _, _ = self.prepare_model_inputs(next(iter(train_loader)))
        Xt, Yt, _, _ = self.prepare_model_inputs(next(iter(val_loader)))
        model.fit(inputs, labels, Xt, Yt)

        return model

    def prepare_model_inputs(self, data, device=None):
        target = data["label"]
        group = data["group"]
        input = data["feature"]
        input_data = data["index"]

        return input, target, group, input_data

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
        df = bring_examples_to_top(df, 6, min_num)

        return df
