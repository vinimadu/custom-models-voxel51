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

    d = dict(model_path=model_path, classes=classes)

    config = CustomFrcnnModelConfig(d)

    return CustomFrcnnModel(config)

class CustomFrcnnModelConfig(fout.TorchImageModelConfig):
    def __init__(self, d):
        super().__init__(d)

        self.model_path = self.parse_string(d, "model_path")

        self.entrypoint_fcn = "torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn"
        self.output_processor_cls = "fiftyone.utils.torch.DetectorOutputProcessor"

class CustomFrcnnModel(fout.TorchImageModel):
    def __init__(self, config):
        super().__init__(config)

        self._model = torch.load(config.model_path,weights_only=False)