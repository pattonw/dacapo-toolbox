from typing import List


from .evaluator import Evaluator
from .instance_evaluation_scores import InstanceEvaluationScores
from dacapo_toolbox.utils.voi import voi as _voi
from dacapo_toolbox.tmp import open_from_identifier

import numpy as np

import logging

logger = logging.getLogger(__name__)


class InstanceEvaluator(Evaluator):
    """
    A class representing an evaluator for instance segmentation tasks.

    Attributes:
        criteria : List[str]
            the evaluation criteria
    Methods:
        evaluate(output_array_identifier, evaluation_array)
            Evaluate the output array against the evaluation array.
        score
            Return the evaluation scores.
    Note:
        The InstanceEvaluator class is used to evaluate the performance of an instance segmentation task.

    """

    criteria: List[str] = ["voi_merge", "voi_split", "voi"]

    def evaluate(self, output_array_identifier, evaluation_array):
        """
        Evaluate the output array against the evaluation array.

        Args:
            output_array_identifier : str
                the identifier of the output array
            evaluation_array : Zarr Array
                the evaluation array
        Returns:
            InstanceEvaluationScores
                the evaluation scores
        Raises:
            ValueError: if the output array identifier is not valid
        Examples:
            >>> instance_evaluator = InstanceEvaluator()
            >>> output_array_identifier = "output_array"
            >>> evaluation_array = open_from_identifier("evaluation_array")
            >>> instance_evaluator.evaluate(output_array_identifier, evaluation_array)
            InstanceEvaluationScores(voi_merge=0.0, voi_split=0.0)
        Note:
            This function is used to evaluate the output array against the evaluation array.

        """
        output_array = open_from_identifier(output_array_identifier)
        evaluation_data = evaluation_array[evaluation_array.roi].astype(np.uint64)
        output_data = output_array[output_array.roi].astype(np.uint64)
        results = voi(evaluation_data, output_data)

        return InstanceEvaluationScores(
            voi_merge=results["voi_merge"],
            voi_split=results["voi_split"],
        )

    @property
    def score(self) -> InstanceEvaluationScores:
        """
        Return the evaluation scores.

        Returns:
            InstanceEvaluationScores
                the evaluation scores
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> instance_evaluator = InstanceEvaluator()
            >>> instance_evaluator.score
            InstanceEvaluationScores(voi_merge=0.0, voi_split=0.0)
        Note:
            This function is used to return the evaluation scores.

        """
        return InstanceEvaluationScores()


def voi(truth, test):
    """
    Calculate the variation of information (VOI) between two segmentations.

    Args:
        truth : ndarray
            the ground truth segmentation
        test : ndarray
            the test segmentation
    Returns:
        dict
            the variation of information (VOI) scores
    Raises:
        ValueError: if the truth and test arrays are not of type np.ndarray
    Examples:
        >>> truth = np.array([[1, 1, 0], [0, 2, 2]])
        >>> test = np.array([[1, 1, 0], [0, 2, 2]])
        >>> voi(truth, test)
        {'voi_split': 0.0, 'voi_merge': 0.0}
    Note:
        This function is used to calculate the variation of information (VOI) between two segmentations.

    """
    voi_split, voi_merge = _voi(test + 1, truth + 1, ignore_groundtruth=[])
    return {"voi_split": voi_split, "voi_merge": voi_merge}
