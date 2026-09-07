; Inno Setup script — دیکته‌یار (persian-whisper-flow)
; بیلد:
;   ۱) .venv\Scripts\python.exe -m PyInstaller dikteyar.spec --noconfirm
;   ۲) ISCC.exe installer\dikteyar.iss
; خروجی: installer\Output\DikteYar-Setup-1.1.0.exe
;
; طراحی per-user: نصب در LocalAppData بدون نیاز به دسترسی مدیر —
; (دانلود مدل در اولین اجرا و رجیستری autostart با admin دردسر می‌شوند)

#define MyAppName "DikteYar"
#define MyAppNameFa "دیکته‌یار"
#define MyAppVersion "1.1.0"
#define MyAppPublisher "DikteYar contributors"
#define MyAppExeName "DikteYar.exe"
#define MyAppURL "https://github.com/nimaone/persian-whisper-flow"

[Setup]
AppId={{7E1B9C42-53A8-4F0E-9D6B-8C25A1F04D9E}
AppName={#MyAppName} ({#MyAppNameFa})
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={localappdata}\Programs\DikteYar
DefaultGroupName=DikteYar
DisableProgramGroupPage=yes
; بدون دسترسی مدیر — نصب per-user
PrivilegesRequired=lowest
OutputDir=Output
OutputBaseFilename=DikteYar-Setup-{#MyAppVersion}
SetupIconFile=..\assets\logo.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; اپ tray است — بعد از نصب اجرا نکن (کاربر خودش باز می‌کند)
CloseApplications=yes
ChangesAssociations=no
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
; دوزبانه در خود ویزارد (کاربر فارسی‌زبان)
RunApp=Run %1 after finishing (recommended: finish, launch, pick a mic)
LaunchAfter=Launch DikteYar (دیکته‌یار)

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "autostart"; Description: "اجرای خودکار با ویندوز (Run at Windows startup)"; \
    GroupDescription: "گزینه‌ها / Options"; Flags: checkedonce

; مدل عمداً حذف می‌شود (نصب‌کننده سبک) — اگر برای تست، model/ را کنار exe
; کپی کرده باشید، این Exclude مانع ورود ۴۳۸MB به نصب‌کننده می‌شود
[Files]
Source: "..\dist\DikteYar\*"; DestDir: "{app}"; Excludes: "model,model.onnx,tokens.txt"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\DikteYar"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall DikteYar"; Filename: "{uninstallexe}"
Name: "{autodesktop}\DikteYar"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; اگر کاربر autostart را خواست — همان کلیدی که خود اپ هم می‌نویسد
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
    ValueType: string; ValueName: "WhisperFlowFarsi"; \
    ValueData: """{app}\{#MyAppExeName}"""; \
    Flags: uninsdeletevalue; Tasks: autostart

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchAfter}"; \
    Flags: nowait postinstall skipifsilent

[UninstallRun]
; خروج از اپ در حال اجرا قبل از حذف
Filename: "{cmd}"; Parameters: "/C taskkill /F /IM DikteYar.exe"; Flags: runhidden; RunOnceId: "KillApp"

[UninstallDelete]
; تنظیمات کاربر حذف نمی‌شود (%APPDATA%\WhisperFlowFarsi) — عمدی
; مدل دانلودشده در {app}\model با پوشه حذف می‌شود
Type: filesandordirs; Name: "{app}"
