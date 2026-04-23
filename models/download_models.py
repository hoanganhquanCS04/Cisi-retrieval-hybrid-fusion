import os
from huggingface_hub import snapshot_download

def download():
    print("Downloading Models...")
    
    # Define cache dir
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hf_cache'))
    os.makedirs(cache_dir, exist_ok=True)
    
    ce_model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    t5_model_id = "castorini/monot5-base-msmarco"
    
    print(f"Downloading {ce_model_id}...")
    ce_local_dir = os.path.join(cache_dir, 'cross_encoder_minilm')
    snapshot_download(repo_id=ce_model_id, local_dir=ce_local_dir)
    print("Done downloading Cross-Encoder.")
    
    print(f"Downloading {t5_model_id}...")
    t5_local_dir = os.path.join(cache_dir, 'monot5_base')
    snapshot_download(repo_id=t5_model_id, local_dir=t5_local_dir)
    print("Done downloading MonoT5.")

if __name__ == "__main__":
    download()
