python -m venv .venv
source .venv/bin/activate
pip3 install -e ".[model_worker,webui]"
for cudnn_so in /usr/lib/python3/dist-packages/tensorflow/libcudnn*; do
  sudo ln -s "$cudnn_so" /usr/lib/x86_64-linux-gnu/
done
pip3 install -e ".[train]"
pip3 install trl bitsandbytes scipy deepspeed