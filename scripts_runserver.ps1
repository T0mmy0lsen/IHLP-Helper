try {
	Write-Output "Started."
	powershell -noexit C:\PSTools\PsExec.exe -i -u SDU\gMSA_IHLPRPA -p ~ "C:\IHLP\backend\venv\Scripts\python.exe" "C:\IHLP\backend\manage.py" runserver 0.0.0.0:8000
}
catch {
	Write-Output "Failed."
}