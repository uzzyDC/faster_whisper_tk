## How to recreate the environment in the future
Create the env using the following Python version:
py -3.12.4-m venv _whisperEnv

Then:
& ".\_whisperEnv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r ".\requirements.txt"


Finally, verify that the model can use the GPU when running your script.
Note: While `requirements.txt` records the Python dependencies, it does not, by itself, guarantee the reproduction of the entire GPU stack.
For an RTX card, you must also ensure that the NVIDIA driver and the CUDA libraries required by CTranslate2 remain compatible.