# Fetal-brain-transient-segmentation

An automatic model to segment T2w MRI of the fetal brain into 31 ROIs (of left and right hemispheres) from 21-36 weeks of gestational age, including transient regions: Subplate, Ventricular Zone and Ganglionic Eminence (see [^4] for segmentation details) and Periventricular White Matter (subsegmented in 3 crossroad portions: C1, C2+C5 and C4 - see [^5] for segmentation details). 

This model was trained with a large and heteregeneous cohort with different aquisition parameters (FOV:1.5-3T, TE=80-250ms), including the developing human connectome proejct (dHCP) dataset [^1] and clinical cohorts from St. Thomas Hospital, London, UK.

Please refer to [^2] and [^3] for the repository and paper regarding the reconstruction/regional segmentation pipeline. Further details on methods for training and validating the model can be found on our paper [^4] (winner of best paper in PIPPI workshop part of the MICCAI conference in 2023). 

## Instructions to run BOUNTI-TR:

Preprocessing of the T2w images (in ".nii" format) needs to be done following the tools on this docker [^6] (please read instructions for using the docker):

1) T2w MRI images for input to BOUNTI-TR should be brain extracted - in case you need to do skull stripping you can use the 3D CNN tool in https://hub.docker.com/r/fetalsvrtk/segmentation - and use the following command on the docker: 

bash /home/auto-proc-svrtk/sctipts/auto-brain-bounti-segmentation-fetal.sh /home/data/your_folder_with_brain_svr_t2_files  /home/data/output_folder_for_segmentations

2) Images need to be preprocessed using the command in the docker:

bash /home/auto-proc-svrtk/sctipts/auto-brain-bounti-segmentation-fetal.sh /home/data/your_folder_with_brain_svr_t2_files  /home/data/output_folder_for_segmentations

- This command resamples images to the desired image size (256x256x256) using pad and reorients to the standard radiological atlas space.

3) Once preprocess is done you can either train from scratch or test the trained model on your machine (you can also use the environement shared here "bounti-tr.yml" to run it):

# Train from scratch example: 
python ./run_bounti_tr.py ./train-imgs-folder ./train-labels-folder ./test-imgs-folder ./checkpoint-folder ./results-folder 1 0 200000 


# Test trained model example: 
python ./run_bounti_tr.py ./train-imgs-folder ./train-labels-folder ./test-imgs-folder ./checkpoint-folder ./results-folder 0 1 1


[^1]: https://www.developingconnectome.org/ 
[^2]: https://github.com/SVRTK/auto-proc-svrtk
[^3]: https://www.biorxiv.org/content/10.1101/2023.04.18.537347v2 
[^4]: https://link.springer.com/chapter/10.1007/978-3-031-45544-5_2 
[^5]: https://dl.acm.org/doi/abs/10.1007/978-3-032-05997-0_10
[^6]: https://hub.docker.com/r/fetalsvrtk/segmentation
