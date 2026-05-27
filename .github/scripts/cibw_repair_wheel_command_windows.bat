set DEST_DIR=%1
set WHEEL=%2

delvewheel repair -w %DEST_DIR %WHEEL
