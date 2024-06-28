from huggingface_hub import snapshot_download

model_id = "microsoft/Phi-3-mini-128k-instruct"
snapshot_download(
    repo_id=model_id,
    local_dir="phi3-128",
    local_dir_use_symlinks=False,
    revision="main",
)
