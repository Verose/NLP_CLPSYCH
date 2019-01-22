export GOPATH="C:\Users\Verose\go\yapproj"

INPUT_FILE="$1"

if [ ! -f $INPUT_FILE ]; then
    echo "File not found!"
    exit(0)
fi

"C:\Users\Verose\go\yapproj\bin\yap.exe"
