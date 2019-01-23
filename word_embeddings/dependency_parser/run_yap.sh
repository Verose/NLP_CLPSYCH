#!/usr/bin/env bash
export GOPATH="C:\Users\Verose\go\yapproj"

INPUT_FILE="$1"

if [[ ! -f ${INPUT_FILE} ]]; then
    echo "File not found!"
    exit 0
fi

cd "C:\Users\Verose\go\yapproj\bin\yap.exe"
# write sentence to file5.txt which ends with newline
# output to bla.txt
./yap.exe hebma -raw file5.txt -out bla.txt
./yap.exe md -in bla.txt -om bla_new.txt
./yap.exe dep -inl bla_new.txt -oc output.txt


# shorter way:
./yap.exe hebma -raw file5.txt -out bla.txt
#./yap.exe joint -f conf/jointzeager.yaml -in bla.txt -l conf/hebtb.labels.conf -m data/joint_arc_zeager_model_temp_i33.b64 -nolemma -oc output_dep.txt -om output_on.txt-os output_os.txt
./yap.exe joint -f conf/jointzeager.yaml -in bla.txt -l conf/hebtb.labels.conf -m data/joint_arc_zeager_model_temp_i33.b64 -nolemma -oc output_dep.txt -om output_on.txt-os output_os.txt -jointstr ArcGreedy -oraclestr ArcGreedy