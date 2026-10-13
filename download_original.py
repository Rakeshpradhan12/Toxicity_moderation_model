from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Horizon-Labs/multilingual-toxicity-small",
    local_dir="models/original",
)
print("done")
