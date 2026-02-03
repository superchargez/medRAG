# Configuration
$PORT = 8001
$MODEL_PATH = "C:\llama\models\embeddinggemma-300M-Q8_0.gguf" # UPDATE THIS PATH FOR EVERY SHELL

Write-Host "Starting llama-server for Embeddings on Port $PORT..." -ForegroundColor Green
Write-Host "Model: $MODEL_PATH"

# Check if file exists
if (-not (Test-Path $MODEL_PATH)) {
    Write-Error "Model file not found at: $MODEL_PATH"
    exit
}

# Start process
# Assuming llama-server is in PATH. If not, use ./path/to/llama-server
& llama-server `
    -m $MODEL_PATH `
    --port $PORT `
    --host 0.0.0.0 `
    --embedding `
    --nobrowser `
    -cb `
    -np 1