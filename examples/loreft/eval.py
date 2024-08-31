"""Evaluate trained ReFT models
"""

import argparse
import json
import os
import shutil

from compute_metrics import compute_metrics
from dataset import LoReftGLUEDataset, LoReftSupervisedDataset
from task_config import task_config
from train import classification_tasks, device, dtype_mapping
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
)

import pyreft


def evaluate_model(
    reft_model_name_or_path: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_datasets,
    train_args,
    trigger_tokens,
    data_collator,
):
    print(f"Evaluating {reft_model_name_or_path}")
    reft_model = pyreft.ReftModel.load(
        reft_model_name_or_path,
        model,
    )
    reft_model.set_device(device)
    # ensure everything is in eval mode
    reft_model.model.eval()
    for _, v in reft_model.interventions.items():
        _ = v[0].eval()

    eval_results = {}
    for dataset_name in eval_datasets:
        # split evalset into chunks
        for split, (eval_dataset, data_items) in eval_datasets[dataset_name].items():

            _, stats = compute_metrics(
                train_args["task"],
                dataset_name,
                reft_model,
                tokenizer,
                eval_dataset,
                data_items,
                trigger_tokens,
                "foo",
                train_args["eval_batch_size"],
                data_collator if train_args["task"] in classification_tasks else None,
                split,
                train_args["greedy_decoding"],
                train_args["temperature"],
                train_args["top_p"],
                train_args["top_k"],
            )
            eval_results.update(stats)
    eval_results_filename = os.path.join(reft_model_name_or_path, "eval.json")
    with open(eval_results_filename, "w", encoding="utf-8") as f:
        json.dump(eval_results, f)


def main():
    """Run eval on trained ReFT model"""
    parser = argparse.ArgumentParser(
        description="Simple script for evaluating ReFT models at various checkpoints"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Directory to load model checkpoints from. Model name is retrieved from this",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.logdir):
        raise ValueError(f"{args.logdir} is not a valid directory")

    logdir = os.path.abspath(args.logdir)
    train_run_args_filename = os.path.join(logdir, "args.json")
    with open(train_run_args_filename, "r", encoding="utf-8") as f:
        train_args = json.load(f)

    dtype = dtype_mapping[train_args["dtype"]]

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        train_args["model"],
        model_max_length=train_args["max_length"],
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.unk_token is None and tokenizer.pad_token is None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        need_resize = True
    else:
        tokenizer.pad_token = tokenizer.unk_token
        need_resize = False

    model = AutoModelForCausalLM.from_pretrained(
        train_args["model"],
        torch_dtype=dtype if dtype != "float8" else None,  # save memory
        load_in_8bit=True if dtype == "float8" else False,
        device_map=device,
    )

    if need_resize:
        model.resize_token_embeddings(len(tokenizer))

    # which layers to intervene on
    layers = train_args["layers"]
    if layers.strip() == "":
        layers = []
    elif layers != "all":
        layers = [int(l) for l in layers.split(";")]
    else:
        temp_config = AutoConfig.from_pretrained(train_args["model"])
        layers = [l for l in range(temp_config.num_hidden_layers)]

    ReftDataset = (
        LoReftGLUEDataset if train_args["task"] == "glue" else LoReftSupervisedDataset
    )

    all_eval_datasets = {}
    for eval_dataset in task_config[train_args["task"]]["eval_datasets"]:
        test_splits = train_args["test_split"].split(";")
        all_eval_datasets[eval_dataset] = {}
        for split in test_splits:
            raw_eval = ReftDataset(
                train_args["task"],
                (
                    eval_dataset
                    if train_args["task"] == "glue"
                    else os.path.join(train_args["data_dir"], eval_dataset)
                ),
                tokenizer,
                data_split=split,
                seed=train_args["seed"],
                max_n_example=train_args["max_n_eval_example"],
                **{
                    "num_interventions": len(layers),
                    "position": train_args["position"],
                    "share_weights": train_args["share_weights"],
                },
            )
            all_eval_datasets[eval_dataset][split] = [raw_eval, raw_eval.raw_dataset]
    eval_datasets = all_eval_datasets

    # select collator based on the type
    if train_args["task"] in classification_tasks:
        data_collator_fn = DataCollatorWithPadding(
            tokenizer=tokenizer, padding="longest"
        )
    else:
        data_collator_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding="longest"
        )
    data_collator = pyreft.ReftDataCollator(data_collator=data_collator_fn)

    trigger_tokens = task_config[train_args["task"]]["trigger_tokens"]

    model_cfg_filename = os.path.join(logdir, "config.json")
    checkpoint_dirs = [d for d in os.listdir(logdir) if "checkpoint-" in d]
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda d: int(d.split("-")[1]))
    for checkpoint in checkpoint_dirs:
        print(f"Evaluating {checkpoint}")
        ckpt_dir = os.path.join(logdir, checkpoint, "intervenable_model")
        # For some reason, `Trainer` doesn't save `config.json`, so we have to port this from the
        # manual save at the top-level
        _ = shutil.copy(model_cfg_filename, ckpt_dir)
        evaluate_model(
            ckpt_dir,
            model,
            tokenizer,
            eval_datasets,
            train_args,
            trigger_tokens,
            data_collator,
        )


if __name__ == "__main__":
    main()
