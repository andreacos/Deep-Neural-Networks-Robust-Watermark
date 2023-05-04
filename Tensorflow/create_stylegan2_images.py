import os
from glob import glob
from distutils.dir_util import copy_tree

# The model used to generate images. Make sure this model is inside the Docker image in 'WORKDIR\networks'
model = 'stylegan2-ffhq-config-b.pkl'
total_images = 100000

# Output directory for the generated images
output_dir = 'stylegan2faces'
merged_dir = 'Datasets/Stylegan2ModelBfaces'

# Mount output to Docker, which inside stores images in stylegan2/results
mounted_volume = f'-v {output_dir}/:/home/stylegan2/results'

# List of truncation parameters
truncation_psis = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1]


def create_images():

    os.makedirs(output_dir, exist_ok=True)

    for idx, psi in enumerate(truncation_psis):

        interval = [int(idx*total_images/len(truncation_psis)), int((idx+1)*total_images/len(truncation_psis))-1]

        cmd = f"docker run --gpus all {mounted_volume} stylegan2 python run_generator.py generate-images " \
              f"--seeds={interval[0]}-{interval[1]} --truncation-psi={psi} --network=networks/{model}"

        print(cmd)
        os.system(cmd)

    return


def merge_images():

    os.makedirs(merged_dir, exist_ok=True)

    directories = sorted(glob(os.path.join(output_dir, '*')))

    for source_dir in directories:
        print(f"Merging {source_dir} to {merged_dir}")
        copy_tree(source_dir, merged_dir, update=0)

    # Delete TXT and PKL files produced by styleGan
    txt_files = glob(os.path.join(merged_dir, '*.txt'))
    for f in txt_files:
        os.remove(f)

    pkl_files = glob(os.path.join(merged_dir, '*.pkl'))
    for f in pkl_files:
        os.remove(f)
    return


if __name__ == '__main__':

    print(f"***** CREATING STYLEGAN 2 FACE DATASET ({total_images} images over {len(truncation_psis)} truncation psis) *****")
    create_images()

    print(f"***** MERGING STYLEGAN 2 FACE DATASET *****")
    merge_images()

