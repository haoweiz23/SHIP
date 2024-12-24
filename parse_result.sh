

for DATASET in caltech101 cifar100 dtd oxford_flowers102 oxford_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
do
    python parse_result.py saves/${DATASET}
    # sleep 10
    wait
done
