from huggingface_hub import HfApi
import threading

api = HfApi()

file_nbr = 7
repo_name = "TehVenom/MPT-7b-WizardLM_Uncensored-Storywriter-Merge"
safetensor = False

def upload_torchies(x, y, repo_name, is_safetensor):
    if (is_safetensor):
        api.upload_file(
            path_or_fileobj=f"./model-0000{x}-of-0000{y}.safetensors",
            path_in_repo=f"model-0000{x}-of-0000{y}.safetensors",
            repo_id=repo_name,
            repo_type="model",
        )
    else:
        api.upload_file(
            path_or_fileobj=f"./pytorch_model-0000{x}-of-0000{y}.bin",
            path_in_repo=f"pytorch_model-0000{x}-of-0000{y}.bin",
            repo_id=repo_name,
            repo_type="model",
        )

threads = []

for i = 1 to file_nbr:
    threads.append(threading.Thread(upload_torchies, [i, file_nbr, repo_name]))
  
if (is_safetensor):
    api.upload_file(
        path_or_fileobj= "model.safetensors.index.json",
        path_in_repo="model.safetensors.index.json",
        repo_id=repo_name,
        repo_type="model",
    )
else:
    api.upload_file(
        path_or_fileobj="./pytorch_model.bin.index.json",
        path_in_repo="pytorch_model.bin.index.json",
        repo_id=repo_name,
        repo_type="model",
    )
