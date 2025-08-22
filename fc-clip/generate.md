### Example conda environment setup
```bash
conda create --name fcclip python=3.8 -y
conda activate fcclip
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

cd fc-clip
python -m pip install -e detectron2
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
pip install -r requirements.txt
cd fcclip/modeling/pixel_decoder/ops
sh make.sh
```
### Generation
```bash
# generate DDD17 pseudo label
python fc-clip/demo/generate_pl_ddd17.py --opts MODEL.WEIGHTS fc-clip/demo/fcclip_cocopan.pth
# generate DSEC pseudo label
python fc-clip/demo/generate_pl_dsec.py --opts MODEL.WEIGHTS fc-clip/demo/fcclip_cocopan.pth
```