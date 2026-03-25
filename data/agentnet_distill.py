import datasets
import argparse
import os

def make_map_fn(split, data_source):
    def process_fn(example, idx):
        data = {
            "data_source": data_source,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", type=str, default="/c2/kangsan/verl/dataset/evocua-chat_agentnet_ubuntu_filtered_vscode_hist2.json")
    parser.add_argument("--outputfile", type=str, default="/c2/kangsan/verl/dataset/agentnet_vscode_train_hist2.parquet")
    args = parser.parse_args()

    dataset = datasets.load_dataset("json", data_files=args.datafile)
    train_dataset = dataset["train"]
    
    print("Soruce data: ", args.datafile)
    print("Source dataset size:", len(train_dataset))

    
    dataid = os.path.basename(args.datafile).replace(".json", "")
    train_dataset = train_dataset.map(function=make_map_fn("train", dataid), with_indices=True, num_proc=8)
    # test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    train_dataset.to_parquet(args.outputfile)
    # test_dataset.to_parquet(args.output_file.replace("train", "test"))

