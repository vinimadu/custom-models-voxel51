# Custom Faster RCNN

Wrapper for Faster RCNN model for the FiftyOne Model Zoo.

This model was trained on the [German Traffic Sign Detection Benchmark](https://benchmark.ini.rub.de/gtsdb_dataset.html) with a modification to detect a single class: 'traffic sign'.

You can follow the tutorial in [fiftyone-pytorch_detection_training](https://github.com/voxel51/fiftyone-examples/blob/master/examples/pytorch_detection_training.ipynb) and the files provided in this repository to train your custom Faster RCNN and add it to the FiftyOne Zoo. 

For that, you may need to adapt the following fields in `manifest.json`:
- `"base_name"`: Set this to a descriptive name for your model, e.g., `"faster-rcnn-gtsdb"`.
- `"base_filename"`: Specify the filename of your model, e.g., `"faster_rcnn_gtsdb.pt"`.
- `"google_drive_id"`: Provide the Google Drive ID where your model is hosted.

## Example Usage

``` python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    max_samples=50,
    shuffle=True,
    label_types=['detections'],
    classes=['Traffic sign']
)

foz.register_zoo_model_source("https://github.com/vinimadu/custom-models-voxel51")
model = foz.load_zoo_model("faster-rcnn-gtsdb-single-class",classes=['background','traffic sign'])

dataset.apply_model(model, label_field="predictions")
```
