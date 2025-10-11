<#
Simple helper to bootstrap vcpkg and install sqlite3 for MSVC.
Run from `cpp` directory in PowerShell: `.	ools\setup_vcpkg.ps1`
#>
param(
    [string] $VcpkgDir = "$PSScriptRoot/../vcpkg",
    [string] $Triplet = "x64-windows"
)

if (-Not (Test-Path $VcpkgDir)) {
    Write-Host "Cloning vcpkg into $VcpkgDir"
    git clone https://github.com/microsoft/vcpkg.git $VcpkgDir
}

Push-Location $VcpkgDir
if (-Not (Test-Path .\vcpkg.exe)) {
    Write-Host "Bootstrapping vcpkg"
    .\bootstrap-vcpkg.bat
}

Write-Host "Installing sqlite3 for triplet $Triplet"
.\vcpkg.exe install sqlite3:$Triplet

Write-Host "Done. To configure CMake with vcpkg use (from cpp/build):"
Write-Host "cmake .. -DCMAKE_TOOLCHAIN_FILE=$VcpkgDir/scripts/buildsystems/vcpkg.cmake -A x64 -G \"Visual Studio 17 2022\""
Pop-Location
