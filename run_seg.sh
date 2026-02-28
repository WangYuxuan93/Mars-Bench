python -m marsbench.main task=segmentation model_name=mask2former data_name=conequest_segmentation load_from_hf=true repo_id="Mirali33/mb-conequest_seg" training.trainer.max_epochs=5


python -m marsbench.main task=segmentation model_name=unet data_name=conequest_segmentation load_from_hf=true repo_id="Mirali33/mb-conequest_seg" training.trainer.max_epochs=5 training.num_workers=0



python -m marsbench.main task=segmentation model_name=ViTMAE data_name=conequest_segmentation load_from_hf=true repo_id="Mirali33/mb-conequest_seg" training.trainer.max_epochs=5 training.num_workers=0


python -m marsbench.main task=segmentation model_name=transformersDPT data_name=conequest_segmentation load_from_hf=true repo_id="Mirali33/mb-conequest_seg" training.trainer.max_epochs=5 training.num_workers=0