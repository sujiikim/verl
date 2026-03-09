import datasets

if __name__ == "__main__":
    dataset = datasets.load_dataset("json", data_files="/c2/kangsan/verl/dataset/evocua-chat_agentnet_ubuntu_filtered_vscode_hist2.json")
    dataset = dataset["train"]
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"] # 2744
    test_dataset = dataset["test"] # 305

    def make_map_fn(split):
        def process_fn(example, idx):
            data = {
                "data_source": "evocua-chat_agentnet_ubuntu_filtered_vscode_hist2",
                "prompt": example["messages"][:-1],
                "images": example["images"],
                "ability": "nlp",
                "reward_model": {"style": "web", "ground_truth": example["messages"][-1]["content"]},
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    train_dataset.to_parquet("agentnet_vscode_train_hist2.parquet")
    test_dataset.to_parquet("agentnet_vscode_test_hist2.parquet")