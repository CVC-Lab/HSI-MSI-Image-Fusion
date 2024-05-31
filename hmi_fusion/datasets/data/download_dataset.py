from huggingface_hub import login
from huggingface_hub import hf_hub_download
import os
# login(token="<your_token>")

# os.environ["HF_DATASETS_CACHE"] = "./"
local_path = hf_hub_download(repo_id="cvc-lab/CAVE", filename="CAVE_segmented.zip", 
                repo_type="dataset", local_dir="./CAVE", 
                local_dir_use_symlinks=False, local_files_only=False)
print(local_path)