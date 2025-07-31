import os
from datasets import load_dataset, concatenate_datasets

def load_dataset_for_benchmark(dataset_name: str):
    dataset = load_dataset(dataset_name, token=os.getenv("HF_TOKEN"))
    
    all_splits = []
    for key in dataset.keys():
        all_splits.append(dataset[key])
    
    dataset_across_splits = concatenate_datasets(all_splits)
    
    def process_image_filename(example):
        if 'image_filename' in example:
            filename = example['image_filename']
            if filename.startswith('data/downloaded_datasets/tatdqa/'):
                example['image_filename'] = filename.split('/')[-1]
            elif filename.startswith('raw/'):
                example['image_filename'] = filename.split('/')[-1]
        return example
    
    dataset_across_splits = dataset_across_splits.map(process_image_filename)
    return dataset_across_splits

if __name__ == "__main__":
    dataset = load_dataset_for_benchmark("vidore/tatdqa_train")
    print(dataset[0])