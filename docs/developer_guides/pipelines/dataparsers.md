# DataParsers

```{image} imgs/pipeline_parser-light.png
:align: center
:class: only-light
:width: 600
```

```{image} imgs/pipeline_parser-dark.png
:align: center
:class: only-dark
:width: 600
```

## What is a DataParser?

The dataparser returns `DataparserOutputs`, which puts all the various datasets into a common format. The DataparserOutputs should be lightweight, containing filenames or other meta information which can later be processed by actual PyTorch Datasets and Dataloaders. The common format makes it easy to add another DataParser. All you have to do is implement the private method `_generate_dataparser_outputs` shown below.

```python
@dataclass
class DataparserOutputs:
    """Dataparser outputs for the which will be used by the DataManager
    for creating RayBundle and RayGT objects."""

    image_filenames: List[Path]
    """Filenames for the images."""
    cameras: Cameras
    """Camera object storing collection of camera information in dataset."""
    alpha_color: Optional[TensorType[3]] = None
    """Color of dataset background."""
    scene_box: SceneBox = SceneBox()
    """Scene box of dataset. Used to bound the scene or provide the scene scale depending on model."""
    semantics: Optional[Semantics] = None
    """Semantics information."""

@dataclass
class DataParser:

    @abstractmethod
    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        """Abstract method that returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """
```

## Example

Here is an example where we implement a DataParser for our Nerfstudio data format.

```python
@dataclass
class NerfstudioDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Nerfstudio)
    """target class to instantiate"""
    data: Path = Path("data/sdfstudio/poster")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "vertical"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""

@dataclass
class Nerfstudio(DataParser):
    """Nerfstudio DatasetParser"""

    config: NerfstudioDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        meta = load_from_json(self.config.data / "transforms.json")
        image_filenames = []
        poses = []
        ...
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
        )
        return dataparser_outputs
```

## Train and Eval Logic

The DataParser will generate a train and eval DataparserOutputs depending on the `split` argument. For example, here is how you'd initialize some `InputDataset` classes that live in the DataManager. Because our DataparserOutputs maintain a common form, our Datasets should be plug-and-play. These datasets will load images needed to supervise the model with `RayGT` objects.

```python
config = NerfstudioDataParserConfig()
dataparser = config.setup()
# train dataparser
dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
input_dataset = InputDataset(dataparser_outputs)
```

You can also pull out information from the DataParserOutputs for other DataMangager componenets, such as the RayGenerator. The RayGenerator generates RayBundle objects from camera and pixel indices.

```python
ray_generator = RayGenerator(dataparser_outputs.cameras)
```

## Our Implementations

Below we enumerate the various dataparsers that we have implemented in our codebase. Feel free to use ours or add your own. Also any contributions are welcome and appreciated!

###### Nerfstudio

This is our custom dataparser. We have a script to convert images or videos with COLMAP to this format.

```{button-link} https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/data/dataparsers/nerfstudio_dataparser.py
:color: primary
:outline:
See the code!
```

###### Blender

We support the synthetic Blender dataset from the original NeRF paper.

```{button-link} https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/data/dataparsers/blender_dataparser.py
:color: primary
:outline:
See the code!
```

###### Instant NGP

This supports the Instant NGP dataset.

```{button-link} https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/data/dataparsers/instant_ngp_dataparser.py
:color: primary
:outline:
See the code!
```

###### Record3D

This dataparser can use recorded data from a >= iPhone 12 Pro using the [Record3D app](https://record3d.app/) . Record a video and export with the `EXR + JPG sequence` format. Unzip export and `rgb` folder before training.

For more information on capturing with Record3D, see the [Custom Dataset Docs](/quickstart/custom_dataset.md).

```{button-link} https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/data/dataparsers/record3d_dataparser.py
:color: primary
:outline:
See the code!
```
