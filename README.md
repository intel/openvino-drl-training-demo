# openVino_Blog

## Installing Software
Step 1: Check to see if the correct version of openMPI is installed (should be 2.1.1)

```console
mpirun --version
```

Step 2: install stable baselines dependencies

```console
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

Step 3: Git clone this library 

Step 4: Install all the python packages (reccomended to this in a python virtual environment)
```console
pip3 install -r requirements.txt
```

Step 5: Install Intel's OpenVino Tool https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_apt.html

Step 6: After intstalling openvino, enable it by running 
```console
source /opt/intel/openvino_2021/bin/setupvars.sh
```

