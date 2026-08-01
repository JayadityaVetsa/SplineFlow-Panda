$ErrorActionPreference = "Stop"
$target = Join-Path $PSScriptRoot "..\assets\mujoco_menagerie"
$resolvedParent = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\assets"))
if (Test-Path -LiteralPath $target) {
    Copy-Item -LiteralPath (Join-Path $resolvedParent "scene.xml") `
      -Destination (Join-Path $target "franka_emika_panda\splineflow_scene.xml") -Force
    Write-Host "MuJoCo Menagerie already exists at $target"
    exit 0
}
New-Item -ItemType Directory -Force -Path $resolvedParent | Out-Null
git clone --depth 1 --filter=blob:none --sparse `
  https://github.com/google-deepmind/mujoco_menagerie.git $target
git -C $target sparse-checkout set franka_emika_panda
Copy-Item -LiteralPath (Join-Path $resolvedParent "scene.xml") `
  -Destination (Join-Path $target "franka_emika_panda\splineflow_scene.xml") -Force
Write-Host "Downloaded MuJoCo Menagerie. Review its per-model licenses before redistribution."
