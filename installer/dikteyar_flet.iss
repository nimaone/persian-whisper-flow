; Inno Setup script — دیکته‌یار نسخه فلت (persian-whisper-flow)
; بیلد:
;   ۱) کلاینت فلت: packaging/flet-windows.zip (v1.0.3) — طبق توضیح dikteyar_flet.spec
;   ۲) .venv\Scripts\python.exe -m PyInstaller dikteyar_flet.spec --noconfirm
;   ۳) ISCC.exe installer\dikteyar_flet.iss
; خروجی: installer\Output\DikteYar-Flet-Setup-1.2.0-x64.exe
;
; AppId جدای نسخه CTk — نصب کنار هم مجاز است (پوشه و Uninstall مستقل).
; طراحی per-user: نصب در LocalAppData بدون نیاز به دسترسی مدیر.

#define MyAppName "DikteYarFlet"
#define MyAppNameFa "دیکته‌یار — فلت"
#define MyAppVersion "1.2.0"
#define MyAppPublisher "DikteYar contributors"
#define MyAppExeName "DikteYarFlet.exe"
#define MyAppURL "https://github.com/nimaone/persian-whisper-flow"

[Setup]
; GUID متمایز از dikteyar.iss — دو نسخه کنار هم نصب میشوند
AppId={{9D4C7B31-6A5E-4F8C-9B2D-1E47F0A3C5D8}
AppName={#MyAppName} ({#MyAppNameFa})
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={localappdata}\Programs\DikteYarFlet
DefaultGroupName=DikteYarFlet
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=Output
OutputBaseFilename=DikteYar-Flet-Setup-{#MyAppVersion}-x64
SetupIconFile=..\assets\logo.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
ChangesAssociations=no
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
RunApp=Run %1 after finishing
LaunchAfter=Launch DikteYarFlet (دیکته‌یار — فلت)

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "autostart"; Description: "اجرای خودکار با ویندوز (Run at Windows startup)"; \
    GroupDescription: "گزینه‌ها / Options"; Flags: checkedonce

; مدل عمداً حذف میشود (نصبکننده سبک) — دیالوگ اولین اجرا دانلود میکند؛
; اگر model/ را برای تست کنار exe کپی کرده باشید، Exclude مانع ورودش میشود
[Files]
Source: "..\dist\DikteYarFlet\*"; DestDir: "{app}"; Excludes: "model,model.onnx,tokens.txt"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\DikteYarFlet"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall DikteYarFlet"; Filename: "{uninstallexe}"
Name: "{autodesktop}\DikteYarFlet"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; اگر کاربر autostart را خواست — همان کلیدی که خود اپ هم مینویسد
; (WhisperFlowFarsi: مشترک با نسخه CTk — آخرین پیکربندی برنده است)
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
    ValueType: string; ValueName: "WhisperFlowFarsi"; \
    ValueData: """{app}\{#MyAppExeName}"""; \
    Flags: uninsdeletevalue; Tasks: autostart

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchAfter}"; \
    Flags: nowait postinstall skipifsilent

[UninstallRun]
; خروج از اپ در حال اجرا قبل از حذف (+ کلاینت فلت ممکن است اجرا باشد)
Filename: "{cmd}"; Parameters: "/C taskkill /F /IM DikteYarFlet.exe & taskkill /F /IM flet.exe"; Flags: runhidden; RunOnceId: "KillApp"

[UninstallDelete]
; تنظیمات کاربر حذف نمیشود (%APPDATA%\WhisperFlowFarsi) — عمدی
; مدل دانلودشده در {app}\model با پوشه حذف میشود
Type: filesandordirs; Name: "{app}"
