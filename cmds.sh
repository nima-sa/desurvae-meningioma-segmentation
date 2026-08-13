#! /bin/bash 

python3 -m pip install -U pip virtualenv
python3 -m virtualenv venv3
source venv3/bin/activate
pip install scikit-build

unzip MONAI.zip
cd MONAI/
pip install -e ".[all]"
cd ..

pip install -r requirements.txt


apt-get update
apt-get install -y software-properties-common
add-apt-repository universe -y
apt-get update
apt install libfuse2 -y


chmod +x CaPTk_1.9.0.bin
printf "Y\n" | ./CaPTk_1.9.0.bin --accept --appimage-extract --target ./CaPTk_1.9.0
cp -r saved_models CaPTk_1.9.0/saved_models
echo "$(ls -lah)"