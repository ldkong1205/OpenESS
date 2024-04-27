<p align="right">English | <a href="./README_CN.md">简体中文</a></p>  


<p align="center">
  <img src="docs/figs/logo.png" align="center" width="23%">
  
  <h3 align="center"><strong>OpenESS: Event-Based Semantic Scene Understanding with Open Vocabularies</strong></h3>

  <p align="center">
      <a href="https://ldkong.com/" target='_blank'>Lingdong Kong</a><sup>1,2</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://github.com/youquanl" target='_blank'>Youquan Liu</a><sup>3</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://ipal.cnrs.fr/lai-xing-ng/" target='_blank'>Lai Xing Ng</a><sup>4</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://ipal.cnrs.fr/benoit-cottereau-personal-page/" target='_blank'>Benoit R. Cottereau</a><sup>5,6</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://www.comp.nus.edu.sg/cs/people/ooiwt/" target='_blank'>Wei Tsang Ooi</a><sup>1</sup>
    </br>
  <sup>1</sup>National University of Singapore&nbsp;&nbsp;&nbsp;
  <sup>2</sup>CNRS@CREATE&nbsp;&nbsp;&nbsp;
  <sup>3</sup>Hochschule Bremerhaven&nbsp;&nbsp;&nbsp;
  <sup>4</sup>Institute for Infocomm Research, A*STAR&nbsp;&nbsp;&nbsp;
  <sup>5</sup>IPAL, CNRS IRL 2955, Singapore&nbsp;&nbsp;&nbsp;
  <sup>6</sup>CerCo, CNRS UMR 5549, Universite Toulouse III
  </p>

</p>

<p align="center">
  <a href="https://ldkong.com/PDF/2024_cvpr_OpenESS.pdf" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-lightblue">
  </a>
  
  <a href="https://ldkong.com/OpenESS" target='_blank'>
    <img src="https://img.shields.io/badge/Project-%F0%9F%94%97-blue">
  </a>
  
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/Demo-%F0%9F%8E%AC-pink">
  </a>
  
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/%E4%B8%AD%E8%AF%91%E7%89%88-%F0%9F%90%BC-red">
  </a>
  
  <a href="https://hits.seeyoufarm.com">
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fldkong1205%2FOpenESS&count_bg=%2300B48B&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitors&edge_flat=false"/>
  </a>
</p>



## About

`OpenESS` is an open-vocabulary event-based semantic segmentation (ESS) framework that synergizes information from image, text, and event-data domains to enable scalable ESS in an open-world, annotation-efficient manner. 

| <img width="169" src="docs/figs/teaser_1.png"> | <img width="169" src="docs/figs/teaser_2.png"> | <img width="169" src="docs/figs/teaser_3.png"> | <img width="169" src="docs/figs/teaser_4.png"> |
| :-: | :-: | :-: | :-: |
| Input Event Stream | “Driveable” | “Car” | “Manmade” |
| <img width="169" src="docs/figs/teaser_5.png"> | <img width="169" src="docs/figs/teaser_6.png"> | <img width="169" src="docs/figs/teaser_7.png"> | <img width="169" src="docs/figs/teaser_8.png"> |
| Zero-Shot ESS | “Walkable” | “Barrier” | “Flat” |



## Updates

- \[2024.05\] - Our paper is available on arXiv, click [here](https://ldkong.com/PDF/2024_cvpr_OpenESS.pdf) to check it out. The code will be available later.
- \[2024.04\] - [OpenESS](https://ldkong.com/PDF/2024_cvpr_OpenESS.pdf) was selected as a :sparkles: highlight :sparkles: at [CVPR 2024](https://cvpr.thecvf.com/Conferences/2024) (2.8% = 324/11532). 
- \[2024.02\] - [OpenESS](https://ldkong.com/PDF/2024_cvpr_OpenESS.pdf) was accepted to [CVPR 2024](https://cvpr.thecvf.com/Conferences/2024)! :tada:


## Outline
- [Installation](#gear-installation)
- [Data Preparation](#hotsprings-data-preparation)
- [Getting Started](#rocket-getting-started)
- [Benchmark](#bar_chart-benchmark)
- [TODO List](#memo-todo-list)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## :gear: Installation

Kindly refer to [INSTALL.md](docs/INSTALL.md) for the installation details.


## :hotsprings: Data Preparation

Kindly refer to [DATA_PREPARE.md](docs/DATA_PREPARE.md) for the details to prepare the [DDD17-Seg]() and [DSEC-Semantic]() datasets.


## :rocket: Getting Started

Please refer to [GET_STARTED.md](docs/GET_STARTED.md) to learn more about how to use this codebase.


## :bar_chart: Benchmark

### OpenESS Framework

| <img src="docs/figs/framework.png"> |
| :-: |


### Annotation-Free ESS

To be updated.


### Fully-Supervised ESS

To be updated.


### Open-Vocabulary ESS

To be updated.


### Qualitative Assessment

| <img src="docs/figs/qualitative.png"> |
| :-: |


## :memo: TODO List

To be updated.


## Citation
If you find this work helpful, please kindly consider citing our paper:
```bibtex
@inproceedings{kong2024openess,
  title = {OpenESS: Event-Based Semantic Scene Understanding with Open Vocabularies},
  author = {Kong, Lingdong and Liu, Youquan and Ng, Lai Xing and Cottereau, Benoit R. and Ooi, Wei Tsang},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024},
}
```

## License

This work is under the [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0), while some specific implementations in this codebase might be with other licenses. Kindly refer to [LICENSE.md](https://github.com/ldkong1205/Calib3D/blob/main/docs/LICENSE.md) for a more careful check, if you are using our code for commercial matters.


## Acknowledgements

To be updated.
