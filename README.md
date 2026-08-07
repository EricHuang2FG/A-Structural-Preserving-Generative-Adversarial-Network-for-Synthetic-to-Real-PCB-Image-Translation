# A Structure-Preserving GAN (SPresGAN) for Synthetic-to-Real PCB Image Translation

## Setting up the Environment
Create a Python 3.12 virtual environment:
```bash
python -m venv .venv
```
Install all dependencies listed in `requirements.txt` by running:
```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```
Furthermore, download [KiCad 10.0](https://www.kicad.org/download/).

## Data Processing

Obtain the raw Domain A data from [open-schematics](https://huggingface.co/datasets/bshada/open-schematics) by running

```bash
python -m src.data_processing.fetch
```

Then, process the schematic files and generate synthetic renders and annotations by running

```bash
export PATH="/Applications/KiCad/KiCad.app/Contents/MacOS:$PATH"
export PYTHONPATH="$(pwd):$PYTHONPATH"

/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3 src/data_processing/process_synthetic.py
```
with the specific paths where your KiCad and its Python package are located. Note that this process is expected to take 12 to 24 hours.

Obtain the Domain B data from [PCB-DSLR](https://zenodo.org/records/3886553?preview_file=cvl_pcb_dslr_1.zip) and unzip the files into the `data/PCB-DSLR` folder. Also, download and unzip the [pcb_data](https://www.kaggle.com/datasets/pkompally/pcb-data) Kaggle dataset, place it within the `data/` directory, and name the folder `pcb_data`. Then, process these data by running:
```bash
python -m src.data_processing.process_real
```

## Model Training

To train CycleGAN with the processed PCB data, first run
```bash
python -m src.data_processing.process_cyclegan
```

to prepare the data for CycleGAN. Then, clone the [official CycleGAN repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) into the project root and run the following command:

```bash
python pytorch-CycleGAN-and-pix2pix/train.py \
    --dataroot ./data/cyclegan_data \
    --name checkpoints \
    --model cycle_gan \
    --checkpoints_dir ./models/cyclegan \
    --batch_size 2 \
    --preprocess resize_and_crop \
    --load_size 256 \
    --crop_size 256 \
    --save_latest_freq 5000 \
    --save_epoch_freq 5 \
    --pool_size 50 \
    --no_dropout
```
This will train CycleGAN for 200 epochs, and its model checkpoints will be saved in `models/cyclegan/checkpoints`

To train the Structure-Preserving GAN (SPresGAN), you must have the trained U-Net segmentor model. You could either train the model yourself by running:
```bash
python -m src.train.train_segmentor
```
or it could be downloaded [here](https://drive.google.com/drive/folders/1yBO6M3fhkbEYkH131Y0aOJ9g91qMiaIB?usp=drive_link). Place the downloaded model file into `models/segmentor/best`. Finally, train the SPresGAN by running:
```bash
python -m src.train.train_spresgan
```
The trained 

## Producing Image Translations

To translate the test images using CycleGAN, run

```bash
python src/inference/inference_cyclegan.py --dataroot ./data/cyclegan_data/testA \
    --checkpoints_dir ./models/cyclegan --name checkpoints --model test --dataset_mode single \
    --no_dropout --results_dir ./outputs/cyclegan/images --model_suffix _A --epoch 200 --num_test 223
```
This command will use the model checkpoint at epoch 200 (the final epoch). To run the model at a different epoch value, simply change the `--epoch` flag. It will translate all 223 test images in the `data/cyclegan_data/testA` directory.

To translate the test images using SPresGAN, run
```bash
python -m src.inference.inference_spresgan
```

## Evaluating Models

To produce FID and mean IoU metrics for the raw synthetic images, CycleGAN and SPresGAN using test data from open-schematics, run:
```bash
python -m src.inference.metrics
```
To evaluate the models on external datasets sourced from [sethhillbrand/kicad_templates](https://github.com/sethhillbrand/kicad_templates.git) and [tscircuit/dataset-srj18](https://github.com/tscircuit/dataset-srj18.git), these datasets need to be additionally fetched and processed. To do so, run:
```bash
python -m src.data_processing.fetch_external_test_datasets
```
Then, process these files and generate synthetic renders and annotations by running:
```bash
export PATH="/Applications/KiCad/KiCad.app/Contents/MacOS:$PATH"
export PYTHONPATH="$(pwd):$PYTHONPATH"

/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3 src/data_processing/process_synthetic.py --external
```
with the specific paths where your KiCad and its Python package are located. Then, prepare the data for CycleGAN by running:
```bash
python -m src.data_processing.process_cyclegan --external
```
Translate images for CycleGAN by running:
```bash
python src/inference/inference_cyclegan.py --dataroot ./data/external_test_datasets/cyclegan/testA \
    --checkpoints_dir ./models/cyclegan --name checkpoints --model test --dataset_mode single \
    --no_dropout --results_dir ./outputs/cyclegan/images --model_suffix _A --epoch 200 --num_test 15
```
Translate images for SPresGAN by running:
```bash
python -m src.inference.inference_spresgan --external
```
Finally, produce the FID and IoU metrics for the external dataset on the synthetic images, the CycleGAN translated images, and the SPresGAN translated images by running:
```bash
python -m src.inference.metrics --external
```