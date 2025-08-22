# :gear: Installation

### General Requirement

This codebase is tested with `torch==2.1.0+cu118`, `torchvision==0.16.0+cu118`, `mmcv-full==1.6.0`, with `CUDA 11.8`. In order to successfully reproduce the results reported in our paper, we recommend you **follow the exact same configuration** with us. 

However, similar versions that came out lately should be good as well.

<hr>

### Step 1: Create Environment
```Shell
conda create -n openess python=3.10
```

### Step 2: Activate Environment
```Shell
conda activate openess
```

### Step 3: Install PyTorch
```Shell
conda install pytorch==2.1.0 torchvision==0.16.0 cudatoolkit=11.8 -c pytorch
```

### Step 4: Install MMCV
```Shell
pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```


## Environment Summary
We provide the list of all packages and their corresponding versions installed in this codebase:
```Shell
absl-py                    2.1.0
addict                     2.4.0
aiofiles                   23.2.1
aiohappyeyeballs           2.6.1
aiohttp                    3.11.14
aiosignal                  1.3.2
albucore                   0.0.24
albumentations             2.0.8
aliyun-python-sdk-core     2.16.0
aliyun-python-sdk-kms      2.16.5
annotated-types            0.7.0
anyio                      4.9.0
argon2-cffi                23.1.0
argon2-cffi-bindings       21.2.0
arrow                      1.3.0
asttokens                  2.4.1
async-lru                  2.0.5
async-timeout              5.0.1
attrs                      25.3.0
babel                      2.17.0
beautifulsoup4             4.13.3
black                      24.10.0
bleach                     6.2.0
blinker                    1.9.0
cachetools                 5.5.0
ccimport                   0.4.4
certifi                    2024.8.30
cffi                       1.17.1
charset-normalizer         3.4.0
click                      8.1.7
colorama                   0.4.6
comm                       0.2.2
ConfigArgParse             1.7
contourpy                  1.3.0
crcmod                     1.7
cryptography               43.0.3
cumm-cu118                 0.7.11
cycler                     0.12.1
dash                       3.0.3
debugpy                    1.8.13
decorator                  4.4.2
defusedxml                 0.7.1
descartes                  1.1.0
diffusers                  0.33.1
easydict                   1.13
einops                     0.8.0
eval_type_backport         0.2.2
exceptiongroup             1.2.2
executing                  2.1.0
fastapi                    0.115.12
fastjsonschema             2.21.1
ffmpy                      0.5.0
filelock                   3.14.0
fire                       0.7.0
flake8                     7.1.1
flash-attn                 2.7.3
Flask                      3.0.3
fonttools                  4.54.1
fqdn                       1.5.1
frozenlist                 1.5.0
fsspec                     2024.10.0
fvcore                     0.1.5.post20221221
gradio                     4.44.1
gradio_client              1.3.0
grpcio                     1.67.0
h11                        0.14.0
h5py                       3.14.0
hdf5plugin                 5.1.0
httpcore                   1.0.7
httpx                      0.28.1
huggingface-hub            0.29.3
idna                       3.10
imageio                    2.37.0
imageio-ffmpeg             0.6.0
importlib_metadata         8.5.0
importlib_resources        6.4.5
iniconfig                  2.0.0
iopath                     0.1.10
ipykernel                  6.29.5
ipython                    8.18.1
ipywidgets                 8.1.5
isoduration                20.11.0
itsdangerous               2.2.0
jedi                       0.19.1
Jinja2                     3.1.4
jmespath                   0.10.0
joblib                     1.4.2
json5                      0.12.0
jsonpointer                3.0.0
jsonschema                 4.23.0
jsonschema-specifications  2024.10.1
jupyter                    1.1.1
jupyter_client             8.6.3
jupyter-console            6.6.3
jupyter_core               5.7.2
jupyter-events             0.12.0
jupyter-lsp                2.2.5
jupyter_server             2.15.0
jupyter_server_terminals   0.5.3
jupyterlab                 4.3.6
jupyterlab_pygments        0.3.0
jupyterlab_server          2.27.3
jupyterlab_widgets         3.0.13
kiwisolver                 1.4.7
kornia                     0.8.0
kornia_rs                  0.1.8
lark                       1.2.2
llvmlite                   0.38.1
loguru                     0.7.3
lyft-dataset-sdk           0.0.8
Markdown                   3.7
markdown-it-py             3.0.0
MarkupSafe                 2.1.5
matplotlib                 3.5.2
matplotlib-inline          0.1.7
mccabe                     0.7.0
mdurl                      0.1.2
mistune                    3.1.3
mmcls                      0.25.0
mmcv-full                  1.6.0 
mmdet                      2.28.2
mmsegmentation             0.30.0 
model-index                0.1.11
motmetrics                 1.1.3
moviepy                    1.0.3
mpmath                     1.3.0
multidict                  6.2.0
mypy-extensions            1.0.0
nbclient                   0.10.2
nbconvert                  7.16.6
nbformat                   5.10.4
nest-asyncio               1.6.0
networkx                   2.4
ninja                      1.11.1.4
notebook                   7.3.3
notebook_shim              0.2.4
numba                      0.55.0
numpy                      1.21.0
nuscenes-devkit            1.1.10
open3d                     0.19.0
opencv-python              4.10.0.84
opencv-python-headless     4.12.0.88
opendatalab                0.0.10
openmim                    0.3.9
openxlab                   0.1.2
ordered-set                4.1.0
orjson                     3.10.16
oss2                       2.17.0
overrides                  7.7.0
packaging                  24.1
pandas                     1.5.3
pandocfilters              1.5.1
parso                      0.8.4
pathspec                   0.12.1
pccm                       0.4.16
pexpect                    4.9.0
pillow                     10.4.0
Pillow-SIMD                9.0.0.post1
pip                        24.2
platformdirs               4.3.6
plotly                     5.24.1
pluggy                     1.5.0
plyfile                    1.1
portalocker                2.10.1
prettytable                3.11.0
proglog                    0.1.12
prometheus_client          0.21.1
prompt_toolkit             3.0.48
propcache                  0.3.1
protobuf                   5.28.3
psutil                     7.0.0
ptyprocess                 0.7.0
pure_eval                  0.2.3
pybind11                   2.13.6
pycocotools                2.0.8
pycodestyle                2.12.1
pycparser                  2.22
pycryptodome               3.21.0
pydantic                   2.10.6
pydantic_core              2.27.2
pydub                      0.25.1
pyflakes                   3.2.0
Pygments                   2.18.0
pyparsing                  3.2.0
pyquaternion               0.9.9
pytest                     8.3.3
python-dateutil            2.9.0.post0
python-dotenv              1.1.0
python-json-logger         3.3.0
python-multipart           0.0.20
PyTurboJPEG                1.7.7
pytz                       2023.4
PyWavelets                 1.4.1
PyYAML                     6.0.2
pyzmq                      26.4.0
referencing                0.36.2
regex                      2024.11.6
requests                   2.32.3
retrying                   1.3.4
rfc3339-validator          0.1.4
rfc3986-validator          0.1.1
rich                       13.4.2
rpds-py                    0.24.0
ruff                       0.11.2
safetensors                0.5.3
scikit-image               0.19.3
scikit-learn               1.5.2
scipy                      1.13.1
semantic-version           2.10.0
Send2Trash                 1.8.3
setuptools                 59.5.0
Shapely                    1.8.5
shellingham                1.5.4
simsimd                    6.5.1
six                        1.16.0
sniffio                    1.3.1
soupsieve                  2.6
spconv-cu118               2.3.8
stack-data                 0.6.3
starlette                  0.46.1
stringzilla                3.12.6
sympy                      1.13.3
tabulate                   0.9.0
tenacity                   9.0.0
tensorboard                2.18.0
tensorboard-data-server    0.7.2
termcolor                  2.5.0
terminado                  0.18.1
terminaltables             3.1.10
threadpoolctl              3.5.0
tifffile                   2024.8.30
timm                       0.5.4
tinycss2                   1.4.0
tomli                      2.0.2
tomlkit                    0.12.0
torch                      2.1.0+cu118
torch-geometric            2.1.0
torch-scatter              2.1.2+pt21cu118
torch-sparse               0.6.18+pt21cu118
torchaudio                 2.1.0+cu118
torchmetrics               0.9.0
torchvision                0.16.0+cu118
tornado                    6.4.2
tqdm                       4.65.2
traitlets                  5.14.3
trimesh                    2.35.39
triton                     2.1.0
typer                      0.15.2
types-python-dateutil      2.9.0.20241206
typing_extensions          4.12.2
tzdata                     2024.2
uri-template               1.3.0
urllib3                    2.3.0
uvicorn                    0.34.0
wcwidth                    0.2.13
webcolors                  24.11.1
webencodings               0.5.1
websocket-client           1.8.0
websockets                 12.0
Werkzeug                   3.0.5
wheel                      0.44.0
widgetsnbextension         4.0.13
xmltodict                  0.14.2
yacs                       0.1.8
yapf                       0.40.1
yarl                       1.18.3
zipp                       3.20.2

```
