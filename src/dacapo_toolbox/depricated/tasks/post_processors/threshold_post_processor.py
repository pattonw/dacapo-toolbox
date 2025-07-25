from .threshold_post_processor_parameters import ThresholdPostProcessorParameters
from dacapo_toolbox.tmp import LocalArrayIdentifier
from .post_processor import PostProcessor
import numpy as np
import daisy
from daisy import Roi, Coordinate
from funlib.persistence import open_ds

from dacapo_toolbox.tmp import (
    open_from_identifier,
    create_from_identifier,
    num_channels_from_array,
)
from funlib.persistence import Array

from typing import Iterable
import logging

logger = logging.getLogger(__name__)

"array > threshold"


class ThresholdPostProcessor(PostProcessor):
    """
    A post-processor that applies a threshold to the prediction.

    Attributes:
        prediction_array_identifier: The identifier of the prediction array.
        prediction_array: The prediction array.
    """

    def __init__(self):
        pass

    def enumerate_parameters(self) -> Iterable["ThresholdPostProcessorParameters"]:
        for i, threshold in enumerate([127]):
            yield ThresholdPostProcessorParameters(id=i, threshold=threshold)

    def set_prediction(self, prediction_array_identifier):
        self.prediction_array_identifier = prediction_array_identifier
        self.prediction_array = open_from_identifier(prediction_array_identifier)

    def process(
        self,
        parameters: "ThresholdPostProcessorParameters",  # type: ignore[override]
        output_array_identifier: "LocalArrayIdentifier",
        num_workers: int = 12,
        block_size: Coordinate = Coordinate((256, 256, 256)),
    ) -> Array:
        """
        Process the prediction with the given parameters.

        Args:
            parameters (ThresholdPostProcessorParameters): The parameters to use for processing.
            output_array_identifier (LocalArrayIdentifier): The identifier of the output array.
            num_workers (int): The number of workers to use for processing.
            block_size (Coordinate): The block size to use for processing.
        """
        # TODO: Investigate Liskov substitution princple and whether it is a problem here
        # OOP theory states the super class should always be replaceable with its subclasses
        # meaning the input arguments to methods on the subclass can only be more loosely
        # constrained and the outputs can only be more highly constrained. In this case
        # we know our parameters will be a `ThresholdPostProcessorParameters` class,
        # which is more specific than the `PostProcessorParameters` parent class.
        # Seems unrelated to me since just because all `PostProcessors` use some
        # `PostProcessorParameters` doesn't mean they can use any `PostProcessorParameters`
        # so our subclasses aren't directly replaceable anyway.
        # Might be missing something since I only did a quick google, leaving this here
        # for me or someone else to investigate further in the future.
        if self.prediction_array._source_data.chunks is not None:
            block_size = self.prediction_array._source_data.chunks

        write_size = Coordinate(
            [
                b * v
                for b, v in zip(
                    block_size[-self.prediction_array.dims :],
                    self.prediction_array.voxel_size,
                )
            ]
        )
        output_array = create_from_identifier(
            output_array_identifier,
            self.prediction_array.axis_names,
            self.prediction_array.roi,
            num_channels_from_array(self.prediction_array),
            self.prediction_array.voxel_size,
            np.uint8,
            overwrite=True,
        )

        read_roi = Roi(write_size * 0, write_size)
        input_array = open_ds(
            f"{self.prediction_array_identifier.container.path}/{self.prediction_array_identifier.dataset}"
        )

        def process_block(block):
            write_roi = block.write_roi.intersect(input_array.roi)
            data = input_array[write_roi] > parameters.threshold
            data = data.astype(np.uint8)
            if int(data.max()) == 0:
                logger.debug("No data in block", write_roi)
                return
            output_array[write_roi] = data

        task = daisy.Task(
            f"threshold_{output_array_identifier.dataset}",
            total_roi=self.prediction_array.roi,
            read_roi=read_roi,
            write_roi=read_roi,
            process_function=process_block,
            check_function=None,
            read_write_conflict=False,
            fit="overhang",
            max_retries=0,
            timeout=None,
        )

        return daisy.run_blockwise([task], multiprocessing=False)
