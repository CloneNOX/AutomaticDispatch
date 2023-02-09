# 请确保先进入了工作的conda环境，使用--yes以自动确认
conda create -n pt-py3.9 python=3.9 --yes
conda install numpy --yes
conda install tqdm --yes
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia --yes
pip install jieba