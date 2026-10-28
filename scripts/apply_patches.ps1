$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$LlamaDir = Join-Path $RootDir "third_party\llama.cpp"

Write-Host "Applying patches to third_party/llama.cpp..." -ForegroundColor Cyan

$patches = @(
    (Join-Path $RootDir "patches\llama_context_mrope_update.patch"),
    (Join-Path $RootDir "patches\ggml_cuda_multibucket_graph.patch")
)

foreach ($patch in $patches) {
    if (Test-Path $patch) {
        Write-Host "Checking $patch..."
        git -C $LlamaDir apply --check $patch 2>$null
        if ($LASTEXITCODE -eq 0) {
            git -C $LlamaDir apply $patch
            Write-Host "Applied $(Split-Path -Leaf $patch)" -ForegroundColor Green
        } else {
            Write-Host "Skipped $(Split-Path -Leaf $patch) (already applied or conflict)" -ForegroundColor Yellow
        }
    }
}
Write-Host "Patch step completed." -ForegroundColor Green
