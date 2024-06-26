# Model parameter transfer 

## Table of Contents
  * [License](#license)
  * [Description](#description)
  * [Using the code](#using-the-code)
    * [Data](#data)
    * [Steps](#steps)
    * [SMPL to SMPL-X](#smpl-to-smpl-x)
    * [SMPL-X to SMPL](#smpl-x-to-smpl)
    * [SMPL+H to SMPL](#smpl%2Bh-to-smpl)
    * [SMPL to SMPL+H](#smpl-to-smpl%2Bh)
    * [SMPL+H to SMPL-X](#smpl%2Bh-to-smpl-x)
    * [SMPL-X to SMPL+H](#smpl-x-to-smpl%2Bh)
  * [Visualize correspondences](visualize-correspondences)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)
  * [Contact](#contact)

## License

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and conditions](https://github.com/vchoutas/smplx/blob/master/LICENSE) and any accompanying documentation before you download and/or use the SMPL-X/SMPLify-X model, data and software, (the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this [License](./LICENSE).

## Description

The repository contains code for converting model parameters of one model to
another. **Never** copy parameters between the models. You will not get the
same poses. SMPL, SMPL+H and SMPL-X shape spaces are **NOT** compatible, since
each model is the result of a different training process.
A more detailed explanation on how we extract correspondences
between the models and the loss function used to estimate the parameters can be
found [here](./docs/transfer.md).

## Requirements

1. Install [mesh](https://github.com/MPI-IS/mesh)
2. Start by cloning the SMPL-X repo:
```Shell 
git clone https://github.com/vchoutas/smplx.git
```
3. Run the following command to install all necessary requirements
```Shell
    pip install -r requirements.txt
```
4. Install the Torch Trust Region optimizer by following the instructions [here](https://github.com/vchoutas/torch-trust-ncg)
5. Install loguru
6. Install open3d
7. Install omegaconf

## Using the code

### Data

Register on the [SMPL-X website](http://smpl-x.is.tue.mpg.de/), go to the
downloads section to get the correspondences and sample data,
by clicking on the *Model correspondences* button.
Create a folder
named `transfer_data` and extract the downloaded zip there. You should have the
following folder structure now:

```bash
transfer_data
├── meshes
│   ├── smpl
│   ├── smplx
├── smpl2smplh_def_transfer.pkl
├── smpl2smplx_deftrafo_setup.pkl
├── smplh2smpl_def_transfer.pkl
├── smplh2smplx_deftrafo_setup.pkl
├── smplx2smpl_deftrafo_setup.pkl
├── smplx2smplh_deftrafo_setup.pkl
├── smplx_mask_ids.npy
```

### Steps

First, break the motion into a set of pose `.obj` files. Depending on how the
SMPL-* parameters are stored this code will differ. For the example AMASS data
in this repository you can use the example code here:

```
python write_obj.py --model-folder ../models/ --motion-file ../transfer_data/support_data/github_data/amass_sample.npz --output-folder ../transfer_data/meshes/amass_sample/
```

To run the `transfer_model` utility you will require a `.yaml` config file,
which can point to the location the output `.obj` files have been saved. Use the
templates in `config_files` in the root of this repository. To convert the
sample AMASS code to SMPL-X:

```
python -m transfer_model --exp-cfg config_files/smplh2smplx_as.yaml
```

Finally, the output `.obj` files have to be merged into a single motion
sequence. Example code to do this in a way that matches `SMPL-X` AMASS archives
can be found in `merge_output.py` and run as follows:

```
python merge_output.py --gender neutral ../output
```

Debug notes describing common problems encountered during this can be found
[here](https://github.com/gngdb/smplx/blob/debug/transfer_model/DEBUG_NOTES.md).
Problems are also discussed in
[two](https://github.com/vchoutas/smplx/issues/82)
[issues](https://github.com/vchoutas/smplx/issues/75).


## My debug note:
- [x] Env setup, I just build upon the soma env
- [x] Step 1, working locally
  - [x] Added wandb, and parrallel codes
  - [x] Working on slurm, GPU is way faster than CPU only, est 5 hours each for 10 subjects
- [x] Step 2, working locally
  - [x] `vertices_frame` need to convert to `vertices` in multiple places, it was updated in the upstream SMPLX repo but not here for some reason. I have fixed those that are throwing errors but not all. 
  - [x] Need gpu? - step 2 conversion need
- [x] Speed issue, the conversion now takes 4 years on my gpu, try slurm and get time estimate
  - [x] If slurm time is reasonable, maybe consider downsampling
    - [x] Update: got response from [issue](https://github.com/vchoutas/smplx/issues/190), added parameter in config file to dramatically speed up, now takes 115h for one activity, so for 90, it will take 10,350h, which is 431 days. If I downsample by 5, and run on 10 instances, it will take 8.6 days
      - [x] Testing blender visualization, exactly the same with SMPLX and SMPL-slow
      - [x] Testing beta similarity, still okayish
      - [x] Moving forward with the new parameter
      - [x] Also increasing batch size to 128 dramatically speed up the process 
    - [x] Tested Slurm speed, downsample 5 --> 9 hours each for 10 subjects, can run parrell, pretty fast
    - [ ] Full will take 9*5 = 45 hours, 1.8 days per subject. If no redundant calculation, it will be 9*4 = 36 hours, 1.5 days per subject
      - [ ] First back up
      - [ ] Minor issue, each current activity is 56gb, so 90 will be 5tb, when no downsample it will be 25tb, need to find a way to store this
  - [ ] If slurm time is unreasonable, get beta for each subject, then find a way to convert SMPLX-pose to SMPL-pose, the body pose (3:66) should be similar, only that SMPL have an extra r and l hand pose, which need to find from SMPLX-handpose
    - [ ] First, see the pkl file for smplx and smpl on the same frame, confirm 3:66 is the same
    - [ ] Also confirm :3 is the same, because, why not, just assert similar
    - [ ] Then, look for the hand pose
- [ ] Step 3
  - [x] Need GPU
  - [x] Merge output now returning `torch.cat(): expected a non-empty list of Tensors`, it might be due to early stopping on the conversion step
    - [x] This is caused by transl set to None, described in [issue](https://github.com/vchoutas/smplx/issues/168), applied temp fix by ignoring the transl since MB don't use it.  
  - [ ] High beta errors
- [ ] Rotate based on camera parameters
- [ ] Format to MotionBert format



#### beta similarity
```
slow/original:
 -0.835 +/- 0.036
 1.530 +/- 0.068
 2.037 +/- 0.222
 0.884 +/- 0.150
 0.802 +/- 0.312
 -0.781 +/- 0.295
 0.132 +/- 0.138
 -0.107 +/- 0.264
 -0.948 +/- 0.478
 1.678 +/- 0.297
```
```
fast/added issue parameter:
-0.834 +/- 0.040
1.532 +/- 0.074
2.042 +/- 0.241
0.874 +/- 0.157
0.835 +/- 0.330
-0.749 +/- 0.319
0.143 +/- 0.155
-0.097 +/- 0.314
-0.910 +/- 0.490
1.661 +/- 0.297
```
- running on local GPU, 
  - batch_size = 1, num_workers=0 --> 20/3381 [06:42<18:48:27, 20.15s/it]
  - batch_size = 8, num_workers=0 --> 5/423 [09:03<12:20:24, 106.28s/it]
  - batch_size = 8, num_workers=16 --> 10/423 [16:52<10:57:32, 95.53s/it]
  - batch_size = 32, num_workers=16, downsample 5 --> 5/22 [23:10<1:18:47, 278.10s/it]
    - 23/60/795*22000*9/24 --> 3.97 days per subject
- running on slurm
  - batch_size = 32, num_workers=16, downsample 5 --> 7/31 [17:56<1:00:14, 150.59s/it]59<00:07,  8.64it/s]
    - 1/4475*22000*9/24 --> 1.8 days per subject
### Full workflow

```bash
cd transfer_model
python write_obj.py --model-folder ../models/ --motion-file ../transfer_data/support_data/S05/Activity04_stageii.pkl --output-folder ../transfer_data/meshes/VEHS_test/ --model-type smplx --batch-moshpp

cd ..
python -m transfer_model  --exp-cfg config_files/smplx2smpl.yaml

cd transfer_model
python merge_output.py --gender neutral ../transfer_data/output/VEHS_test/
```



### SMPL to SMPL-X

To run the code to convert SMPL meshes to SMPL-X parameters use the following command:
  ```Shell
  python -m transfer_model --exp-cfg config_files/smpl2smplx.yaml
  ```
This should be run from the top directory of the repository.

The file *smpl2smplx.yaml* contains a sample configuration that reads meshes from a folder,
processes them and returns pkl files with SMPL-X parameters. To run on your own data create a folder
with SMPL meshes, in either ply or obj format, change the path in the config file and run the code.

### SMPL-X to SMPL

To run the code to convert SMPL-X meshes to SMPL parameters use the following command:
  ```Shell
  python -m transfer_model  --exp-cfg config_files/smplx2smpl.yaml
  ```

The file *smplx2smpl.yaml* contains a sample configuration that reads meshes from a folder,
processes them and returns pkl files with SMPL parameters. To run on your own data create a folder
with SMPL-X meshes, in either ply or obj format, change the path in the config file and run the code.
When creating the SMPL-X meshes, do not use the hand and face parameters. 
Naturally, you will lose all hand and face information if you choose this, since
SMPL cannot model them.


### SMPL+H to SMPL

To run the code to convert SMPL+H meshes to SMPL parameters use the following command from the root `smplx` directory:
  ```Shell
  python -m transfer_model --exp-cfg config_files/smplh2smpl.yaml
  ```
This should be run from the top directory of the repository.

The file *smplh2smpl.yaml* contains a sample configuration that reads meshes from a folder,
processes them and returns pkl files with SMPL parameters. To run on your own data create a folder
with SMPL+H meshes, in either ply or obj format, change the path in the config file and run the code.
Note that using this direction means that you will lose information on the
hands.


### SMPL to SMPL+H

To run the code to convert SMPL meshes to SMPL+H parameters use the following command:
  ```Shell
  python -m transfer_model --exp-cfg config_files/smpl2smplh.yaml
  ```
This should be run from the top directory of the repository.

The file *smpl2smplh.yaml* contains a sample configuration that reads meshes from a folder,
processes them and returns pkl files with SMPL parameters. To run on your own data create a folder
with SMPL meshes, in either ply or obj format, change the path in the config file and run the code.

### SMPL+H to SMPL-X

To run the code to convert SMPL+H meshes to SMPL-X parameters use the following command:
  ```Shell
  python -m transfer_model --exp-cfg config_files/smplh2smplx.yaml
  ```
This should be run from the top directory of the repository.

The file *smplh2smplx.yaml* contains a sample configuration that reads meshes from a folder,
processes them and returns pkl files with SMPL-X parameters. To run on your own data create a folder
with SMPL+H meshes, in either ply or obj format, change the path in the config file and run the code.


### SMPL-X to SMPL+H

To run the code to convert SMPL-X meshes to SMPL+H parameters use the following command:
  ```Shell
  python -m transfer_model --exp-cfg config_files/smplx2smplh.yaml
  ```
This should be run from the top directory of the repository.

The file *smplx2smpl.yaml* contains a sample configuration that reads meshes from a folder,
processes them and returns pkl files with SMPL+H parameters. To run on your own data create a folder
with SMPL-X meshes, in either ply or obj format, change the path in the config file and run the code.
Make sure that you do not use the jaw pose and expression parameters to generate
the meshes.


## Visualize correspondences

To visualize correspondences:
```Shell
python vis_correspondences.py --exp-cfg configs/smpl2smplx.yaml --exp-opts colors_path PATH_TO_SMPL_COLORS
```
You should then see the following image. Points with similar color are in
correspondence.
![Correspondence example](./docs/images/smpl_smplx_correspondence.png)

## Citation

Depending on which model is loaded for your project, i.e. SMPL-X or SMPL+H or SMPL, please cite the most relevant work:

```
@article{SMPL:2015,
    author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
    title = {{SMPL}: A Skinned Multi-Person Linear Model},
    journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
    month = oct,
    number = {6},
    pages = {248:1--248:16},
    publisher = {ACM},
    volume = {34},
    year = {2015}
}
```

```
@article{MANO:SIGGRAPHASIA:2017,
          title = {Embodied Hands: Modeling and Capturing Hands and Bodies Together},
          author = {Romero, Javier and Tzionas, Dimitrios and Black, Michael J.},
          journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
          volume = {36},
          number = {6},
          pages = {245:1--245:17},
          series = {245:1--245:17},
          publisher = {ACM},
          month = nov,
          year = {2017},
          url = {http://doi.acm.org/10.1145/3130800.3130883},
          month_numeric = {11}
        }
```


```
@inproceedings{SMPL-X:2019,
    title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
    author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019}
}
```


## Acknowledgments
The code of this repository was implemented by [Vassilis Choutas](vassilis.choutas@tuebingen.mpg.de),
based on a Chumpy implementation from [Timo Bolkart](timo.bolkart@tuebingen.mpg.de).

## Contact

For questions, please contact [smplx@tue.mpg.de](smplx@tue.mpg.de).
