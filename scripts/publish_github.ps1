param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Owner = "Berserk7777777",
    [string]$Name = "DSP-Fashion",
    [string]$Email = "wk17714342590@163.com",
    [switch]$PrepareOnly
)
$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$manifestPath = Join-Path $RepoRoot "UPLOAD_MANIFEST.json"
if (-not (Test-Path -LiteralPath $manifestPath)) { throw "Missing reviewed UPLOAD_MANIFEST.json." }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.repository -ne "$Owner/$Name") { throw "Manifest repository differs from requested destination." }
foreach ($entry in $manifest.files) {
    $file = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $entry.path))
    if (-not $file.StartsWith($RepoRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Path outside upload root." }
    if (-not (Test-Path -LiteralPath $file)) { throw "Missing reviewed file: $($entry.path)" }
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
        throw "Reviewed file changed: $($entry.path). Prepare a new source snapshot."
    }
}
$gitExecutable = (Get-Command git -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
# Agent-created snapshots can have a different Windows owner. Trust only this
# reviewed directory for these commands; never change global safe.directory.
$gitScope = @('-c', ('safe.directory=' + $RepoRoot.Replace('\', '/')), '-C', $RepoRoot)
foreach ($variable in @('GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE')) {
    if ([Environment]::GetEnvironmentVariable($variable)) {
        throw "Inherited $variable redirects Git. Clear it in this terminal before publishing."
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot '.git'))) {
    & $gitExecutable @gitScope init -b main
    if ($LASTEXITCODE -ne 0) { throw "Git initialization failed." }
}
$actualRoot = & $gitExecutable @gitScope rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0 -or -not $actualRoot -or
    [System.IO.Path]::GetFullPath($actualRoot) -ne $RepoRoot) {
    throw "Git did not recognize the reviewed snapshot as its repository root."
}
    & $gitExecutable @gitScope config --local user.name $Owner
    if ($LASTEXITCODE -ne 0) { throw "Cannot configure commit author." }
    & $gitExecutable @gitScope config --local user.email $Email
    if ($LASTEXITCODE -ne 0) { throw "Cannot configure commit email." }
    & $gitExecutable @gitScope config --local core.quotepath false
    if ($LASTEXITCODE -ne 0) { throw "Cannot configure Git path display." }
    $branch = & $gitExecutable @gitScope branch --show-current
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect current branch." }
    if ($branch -ne "main") { throw "Expected main branch; found $branch. No branch was overwritten." }
    if ($PrepareOnly) {
        Write-Output "Local preparation verified: $RepoRoot (branch: $branch)"
        return
    }

    # Use the official Git Credential Manager. Sign-in/device confirmation is
    # completed by the account holder; no PAT is written into files or URLs.
    & $gitExecutable @gitScope credential-manager github login --username $Owner --device --no-ui
    if ($LASTEXITCODE -ne 0) { throw "GitHub sign-in failed." }
    $credentialText = "protocol=https`nhost=github.com`nusername=$Owner`n`n" | & $gitExecutable @gitScope credential fill
    if ($LASTEXITCODE -ne 0) { throw "Could not read the signed-in GitHub credential." }
    $credential = @{}
    foreach ($line in $credentialText) {
        if ($line -match '^([^=]+)=(.*)$') { $credential[$matches[1]] = $matches[2] }
    }
    if (-not $credential.password) { throw "Credential manager returned no authentication token." }
    $headers = @{
        Authorization = "Bearer $($credential.password)"
        Accept = "application/vnd.github+json"
        "X-GitHub-Api-Version" = "2022-11-28"
        "User-Agent" = "DSP-Fashion-source-upload"
    }
    $user = Invoke-RestMethod -Uri "https://api.github.com/user" -Headers $headers
    if ($user.login -ne $Owner) { throw "Authenticated GitHub account does not match $Owner." }
    $repository = $null
    try {
        $repository = Invoke-RestMethod -Uri "https://api.github.com/repos/$Owner/$Name" -Headers $headers
    } catch {
        if ($_.Exception.Response.StatusCode -ne [System.Net.HttpStatusCode]::NotFound) { throw }
    }
    if (-not $repository) {
        $body = @{ name=$Name; private=$true; auto_init=$false;
                   description="Inference-Time Scaling for Preference Alignment in Generative Fashion Recommendation; DSP: Delayed Seed Pruning" } | ConvertTo-Json
        $repository = Invoke-RestMethod -Method Post -Uri "https://api.github.com/user/repos" -Headers $headers -ContentType "application/json" -Body $body
    }
    if (-not $repository.private) { throw "Existing repository is public; upload helper expects a private repository." }
    $credential.Clear()
    $credentialText = $null
    $headers = $null

    $remote = "https://github.com/$Owner/$Name.git"
    $remotes = @(& $gitExecutable @gitScope remote)
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect repository remotes." }
    if ($remotes -contains "origin") {
        $existingRemote = & $gitExecutable @gitScope remote get-url origin
        if ($LASTEXITCODE -ne 0) { throw "Cannot inspect origin URL." }
        if ($existingRemote -ne $remote) { throw "Existing origin differs from $remote." }
    } else {
        & $gitExecutable @gitScope remote add origin $remote
        if ($LASTEXITCODE -ne 0) { throw "Cannot add GitHub remote." }
    }
    $paths = @($manifest.files | ForEach-Object { $_.path }) + @("UPLOAD_MANIFEST.json")
    $pathspec = Join-Path ([System.IO.Path]::GetTempPath()) ("dsp-upload-" + [guid]::NewGuid().ToString("N") + ".paths")
    try {
        [System.IO.File]::WriteAllText($pathspec, (($paths -join [char]0) + [char]0), (New-Object System.Text.UTF8Encoding($false)))
        & $gitExecutable @gitScope add "--pathspec-from-file=$pathspec" --pathspec-file-nul
        if ($LASTEXITCODE -ne 0) { throw "Staging reviewed source files failed." }
    } finally { Remove-Item -LiteralPath $pathspec -ErrorAction SilentlyContinue }
    $staged = @(& $gitExecutable @gitScope diff --cached --name-only)
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect staged source." }
    foreach ($file in $staged) { if ($paths -notcontains $file) { throw "Unreviewed staged file: $file" } }
    if ($staged.Count -gt 0) {
        & $gitExecutable @gitScope commit -m "Add DSP-Fashion backbones, offline pools, and evaluation workflows"
        if ($LASTEXITCODE -ne 0) { throw "Git commit failed." }
    }
    & $gitExecutable @gitScope push -u origin main
    if ($LASTEXITCODE -ne 0) { throw "Push failed. No force push was attempted." }
    $localHead = & $gitExecutable @gitScope rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect local commit." }
    $remoteHead = & $gitExecutable @gitScope ls-remote --heads origin refs/heads/main
    if ($LASTEXITCODE -ne 0 -or -not $remoteHead -or ($remoteHead -split '\s+')[0] -ne $localHead) {
        throw "Could not verify the published main commit." }
    Write-Output "Published and verified: https://github.com/$Owner/$Name"
    Write-Output "Server clone: git clone $remote"
