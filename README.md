Ensure that all dependences listed in `requirements.txt` are installed and that KiCad 10.0 is available.

***Data Processing***

Obtain the raw Domain A data from [open-schematics](https://huggingface.co/datasets/bshada/open-schematics) by running

```bash
python -m src.data_processing.fetch
```

Then, process the schematic files by running

```bash
export PATH="/Applications/KiCad/KiCad.app/Contents/MacOS:$PATH"
export PYTHONPATH="$(pwd):$PYTHONPATH"

/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3 src/data_processing/process_synthetic.py
```

with the path to your KiCad and its Python are located. Note that this process is expected to take 12 to 24 hours.

Obtain the Domain B data from [PCB-DSLR](https://zenodo.org/records/3886553?preview_file=cvl_pcb_dslr_1.zip) and unzip the files into the `data/PCB-DSLR` folder. Then, run

```bash
python -m src.data_processing.process_real
```

to process the real PCB images.

***Model Training***

To train CycleGAN with the obtained PCB data, first run

```bash
python -m src.data_processing.process_cyclegan
```

to prepare the data for CycleGAN. Then, clone the [official CycleGAN repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and run the following command:

```bash
python pytorch-CycleGAN-and-pix2pix/train.py \
    --dataroot ./data/cyclegan_data \
    --name checkpoints \
    --model cycle_gan \
    --batch_size 2 \
    --preprocess resize_and_crop \
    --load_size 256 \
    --crop_size 256 \
    --save_latest_freq 5000 \
    --save_epoch_freq 5 \
    --pool_size 50 \
    --no_dropout
```

To train the U-Net segmentor, run

```bash
python -m src.train.train_segmentor
```

To train the Structure-Preserving GAN (SPresGAN), run

```bash
python -m src.train.train_spresgan
```

***Producing Image Translations***

To translate the test images using CycleGAN, run

```bash
python src/inference/inference_cyclegan.py --dataroot ./data/cyclegan_data/testA \
    --checkpoints_dir ./models/cyclegan --name checkpoints --model test --dataset_mode single \
    --no_dropout --num_test 162 --results_dir ./outputs/cyclegan/images --model_suffix _A --epoch 30
```

This command will use the model checkpoint at epoch 30. To run the model at a different epoch value, simply change the `--epoch` flag.

To translate the test images using SPresGAN, run

```bash
python -m src.inference.inference_spresgan
```

***Evaluating Models***

To produce FID and mean IoU metrics for CycleGAN and SPresGAN, run

```bash
python -m src.inference.metrics
```
