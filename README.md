# Monocular Depth Estimation for Tilted Images via Gravity Rectifier
**Authors:** [Yuki Saito], [Hideo Saito], [Vincent Fremont]  


This is a repository of Paper: "Monocular Depth Estimation for Tilted Images via Gravity Rectifier".

<a href="http://hvrl.ics.keio.ac.jp/saito_y/images/IW-FCV/system_overview.png" target="_blank"><img src="https://user-images.githubusercontent.com/52692327/183321849-1bf5794c-fdd4-4c4e-8f82-36d5b94aaa25.gif" 
alt="TeaserVideo" width="526" height="132" border="0" /></a>



### Related Publications:

[Our paper3] Yuki Saito, Hideo Saito，Vincent Fremont, **Monocular Depth Estimation for Tilted Images via Gravity Rectifier**，第25回画像の認識・理解シンポジウム(Miru2022)，2022年7月26日発表済． ** [site](https://sites.google.com/view/miru2022)**

[Our Paper2] 

# 1. License

If you use our scripts  in an academic work, please cite:

    @article{TiltedDepth2022,
      title={Monocular Depth Estimation for Tilted Images via Gravity Rectifier},
      author={Yuki Saito, Hideo Saito, and Vincent Fremont},
      journal={},
      pages={},
      publisher={},
      year={2022}
     }

# 2. Prerequisites

This code was tested with Pytorch 11.24, CUDA 11.0 and Ubuntu 20.04.
Training takes about 30 hours with the default parameters on the ScanNet standard split on a single NVIDIA 3090 GPU.
You can tune the training parameter in config/XXX.yml files (e.g.: batch_size, epochs, pre-trained models, etc...)


# 3. How to Run

## Training

1. Download a ScanNet dataset from author's link: http://www.scan-net.org/

2. Run the bash script. Our training is composed of 3 phase: training of gravity recitifiernetwork, depth prediction network, and concat training of both networks.  

  ```
  $ ./train.sh
  ```

## Testing

Run the following scipt to evaluate the model.

  ```
  $ python test.py --cfg XXX --dataset XXX
  ```

There is several optional parameters for evaluation in command line. 

  ```
    --cfg : setting file of the model. It contains model_name/ckpt_path/batch_size/dataset/training mode etc...

    --dataset: dataset name(scannet/NYUv2/TUMrgbd_frei*rpy/OurDataset_roll_seq*/OurDataset_pitch_seq*)

    --bs: batch size.

    --imshow: if show predicted depth map and rectified input

    --save_video: if save precicted depth map video in .mp4 format

    --bin_eval: evaluate accuracies at every XX degree of rotation angles(e.g.: -15~0, 0~15, 15~30...)

    --time: evaluate inference time
  ```
  
# 4. Demo Results

## Monocular Depth Estimation Module

<a href="http://hvrl.ics.keio.ac.jp/saito_y/images/TiltedDepthEstimation/Qualitative_OurDataset_OtherBaselines_v3.pdf" target="_blank"><img src="http://hvrl.ics.keio.ac.jp/saito_y/images/TiltedDepthEstimation/Qualitative_OurDataset_OtherBaselines_v3.pdf"
alt="OtherBaseline" width="916" height="197" border="30" /></a>



## Dense Reconstruction Demo
<a href="http://hvrl.ics.keio.ac.jp/saito_y/images/IW-FCV/DenseReconstruction.png" target="_blank"><img src="http://hvrl.ics.keio.ac.jp/saito_y/images/IW-FCV/DenseReconstruction.png"
alt="CNN-MonoFusion-Demo" width="648" height="512" border="30" /></a>
