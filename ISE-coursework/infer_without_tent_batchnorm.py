
import pandas as pd
import os
import nltk
nltk.download('stopwords')
import torch
import torch.nn
from simple_mlp import MLPWithBatchNorm
import TextDataset
import os
import pandas as pd
from torch.utils.data import  DataLoader
import helper
import TextDataset
torch.serialization.add_safe_globals([
    MLPWithBatchNorm,
    torch.nn.modules.linear.Linear,
    torch.nn.ReLU,
    torch.nn.modules.activation.ReLU,
    torch.nn.LayerNorm,
    torch.nn.BatchNorm1d
])



def write_row_to_csv(file_path, columns, values):

    assert len(columns) == len(values), "Columns and values must be the same length."
    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    df = pd.DataFrame([dict(zip(columns, values))])
    df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
basedataset="tensorflow"
datasets= ["caffe", "incubator-mxnet", "keras", "pytorch"]

for dataset in datasets:
    print(f"In dataset {dataset}")
    for i in range(20):
        print(f"In iteration {i+1}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_base = torch.load('models/baseline_batchnorm/baseline_model_tensorflow_batchnorm20_iteration.pt',map_location=device) ## model as the closest to mean
        inference_dataset=TextDataset.TextDatasetTFIDF(f"datasets/{dataset}.csv")
        inference_loader = DataLoader(inference_dataset, batch_size=32, shuffle=True)
        result=helper.evaluate_model(model_base,inference_loader,basedataset,dataset,i+1)
    #        return {
    #     "iteration":iteration,
    #     "name": name,
    #     "accuracy": accuracy, 
    #     "precision": precision,
    #     "recall": recall,
    #     "f1": f1,
    #     "auc": auc
    # }
        filepath=f"base_line_tf/base_data_tensorflow_batchnorm{dataset}.csv"
        write_row_to_csv(filepath,["iteration","name","accuracy","precision","recall","f1","auc "],[result["iteration"],result["name"],result["accuracy"],result["precision"],result["recall"],result["f1"],result["auc"]])