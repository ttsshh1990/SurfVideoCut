DOCX/HTML Compare App

Files needed to copy to another Windows machine:
- compare_ui_server.py
- compare_ui.html
- generate_diff_pdf.py
- launch_compare_app.py
- setup_windows.py
- Setup DOCX Compare App.bat
- Start DOCX Compare App.bat
- requirements.txt

Optional files:
- Open DOCX Compare App.command
  macOS launcher only, not needed on Windows.
- legacy/docx-html-qa-compare-v2.html
  Legacy browser-only comparator, not needed for the current PDF-annotation app.

Do not copy these local-only folders/files unless you want sample data or old outputs:
- local_test/
- .venv/
- __pycache__/
- ui_runs/
- ui_server.log

Windows setup:
1. Install Python 3.11+.
2. Double-click:
   Setup DOCX Compare App.bat
3. If setup fails, check:
   setup_windows.log
4. Start the app:
   Start DOCX Compare App.bat

What the dependencies are for:
- playwright
  Renders the input HTML in Chromium and exports the PDF.
- pypdf
  Adds PDF comment annotations for the detected differences.
- Python standard library modules are used for everything else.

Runtime output:
- ui_runs\
  The app writes each compare job here.
- ui_server.log
  Server startup/runtime log.
- setup_windows.log
  Windows setup/install log.
