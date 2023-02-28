for(;;) {
	powershell C:\PSTools\PsExec.exe -i -u SDU\gMSA_IHLPRPA -p ~ "C:\IHLP\backend\venv\Scripts\python.exe" "C:\IHLP\backend\manage.py" sandbox
	Start-Sleep 360
}