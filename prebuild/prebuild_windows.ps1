$Url = "https://github.com/xianyi/OpenBLAS/releases/download/v0.3.20/OpenBLAS-0.3.20-x86.zip"
$File = $env:TEMP + "OpenBLAS-0.3.20-x86.zip"
$Directory = $PSScriptRoot + "/OpenBLAS-0.3.20-x86"
Invoke-RestMethod -Uri $Url -OutFile $File
Expand-Archive $File -DestinationPath $Directory
