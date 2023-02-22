# 请确保先进入了工作的conda环境，使用--yes以自动确认
conda create -n fasttext python=3.9 --yes
conda activate fasttext
conda install numpy --yes
conda install tqdm --yes
conda install pandas --yes
conda install scipy --yes
conda install flask --yes
conda install xlrd --yes
pip install fasttext
pip install jieba