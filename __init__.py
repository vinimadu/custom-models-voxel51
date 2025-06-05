import torch
import fiftyone.utils.torch as fout

def download_model(model_name, model_path):
    pass

def load_model(model_name, model_path, classes, **kwargs):
    """Loads the model.

    Args:
        model_name: the name of the model to load, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which the model was
            donwloaded, as declared by the ``base_filename`` field of the
            manifest
        **kwargs: optional keyword arguments that configure how the model
            is loaded

    Returns:
        a :class:`fiftyone.core.models.Model`
    """

    # Consturct the specified `Model` instance, generally by importing
    # other modules in `model_dir`

    config_dict = MODEL_TYPES[model_name]

    d = dict(model_path=model_path, classes=classes)

    config = CustomModelConfig(d, config_dict)

    return CustomModel(config)

class CustomModelConfig(fout.TorchImageModelConfig):
    def __init__(self, d, config_dict):
        super().__init__(d)

        self.model_path = self.parse_string(d, "model_path")

        for k, v in config_dict.items():
            setattr(self, k, v)

class CustomModel(fout.TorchImageModel):
    def __init__(self, config):
        super().__init__(config)

        self._model = torch.load(config.model_path,weights_only=False)

MODEL_TYPES = {
    "faster-rcnn-gtsdb-single-class": {
        'type': 'detection',
        'entrypoint_fcn': "torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn",
        'output_processor_cls': "fiftyone.utils.torch.DetectorOutputProcessor"
    },
    "mobilenet_v2-torch-day-night": {
        'type': 'classification',
        'entrypoint_fcn': "torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn",
        'output_processor_cls': "fiftyone.utils.torch.ClassifierOutputProcessor",
        "image_min_dim": 224,
        "image_max_dim": 2048,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "embeddings_layer": "<classifier.1"
    }
}