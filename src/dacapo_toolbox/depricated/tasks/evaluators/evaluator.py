import xarray as xr

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import math
import itertools
from funlib.persistence import Array

if TYPE_CHECKING:
    from dacapo_toolbox.tasks.evaluators.evaluation_scores import EvaluationScores
    from dacapo_toolbox.datasplits.datasets import DatasetConfig
    from dacapo_toolbox.tmp import LocalArrayIdentifier
    from dacapo_toolbox.tasks.post_processors import PostProcessorParameters

# Custom types for improved readability
OutputIdentifier = tuple["DatasetConfig", "PostProcessorParameters", str]
Iteration = int
Score = float
BestScore = tuple[Iteration, Score] | None


class Evaluator(ABC):
    """
    Base class of all evaluators: An abstract class representing an evaluator that compares and evaluates the output array against the evaluation array.

    An evaluator takes a post-processor's output and compares it against
    ground-truth. It then returns a set of scores that can be used to
    determine the quality of the post-processor's output.

    Attributes:
        best_scores : dict[OutputIdentifier, BestScore]
            the best scores for each dataset/post-processing parameter/criterion combination
    Methods:
        evaluate(output_array_identifier, evaluation_array)
            Compare and evaluate the output array against the evaluation array.
        is_best(dataset, parameter, criterion, score)
            Check if the provided score is the best for this dataset/parameter/criterion combo.
        get_overall_best(dataset, criterion)
            Return the best score for the given dataset and criterion.
        get_overall_best_parameters(dataset, criterion)
            Return the best parameters for the given dataset and criterion.
        compare(score_1, score_2, criterion)
            Compare two scores for the given criterion.
        set_best(validation_scores)
            Find the best iteration for each dataset/post_processing_parameter/criterion.
        higher_is_better(criterion)
            Return whether higher is better for the given criterion.
        bounds(criterion)
            Return the bounds for the given criterion.
        store_best(criterion)
            Return whether to store the best score for the given criterion.
    Note:
        The Evaluator class is used to compare and evaluate the output array against the evaluation array.

    """

    @abstractmethod
    def evaluate(
        self, output_array_identifier: "LocalArrayIdentifier", evaluation_array: Array
    ) -> "EvaluationScores":
        """
        Compares and evaluates the output array against the evaluation array.

        Args:
            output_array_identifier : LocalArrayIdentifier
                The identifier of the output array.
            evaluation_array : Array
                The evaluation array.
        Returns:
            EvaluationScores
                The evaluation scores.
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> evaluator = Evaluator()
            >>> output_array_identifier = LocalArrayIdentifier("output_array")
            >>> evaluation_array = Array()
            >>> evaluator.evaluate(output_array_identifier, evaluation_array)
            EvaluationScores()
        Note:
            This function is used to compare and evaluate the output array against the evaluation array.
        """
        pass

    @property
    def best_scores(
        self,
    ) -> dict[OutputIdentifier, BestScore]:
        """
        The best scores for each dataset/post-processing parameter/criterion combination.

        Returns:
            dict[OutputIdentifier, BestScore]
                the best scores for each dataset/post-processing parameter/criterion combination
        Raises:
            AttributeError: if the best scores are not set
        Examples:
            >>> evaluator = Evaluator()
            >>> evaluator.best_scores
            {}
        Note:
            This function is used to return the best scores for each dataset/post-processing parameter/criterion combination.
        """
        if not hasattr(self, "_best_scores"):
            self._best_scores: dict[OutputIdentifier, BestScore] = {}
        return self._best_scores

    def is_best(
        self,
        dataset: "DatasetConfig",
        parameter: "PostProcessorParameters",
        criterion: str,
        score: "EvaluationScores",
    ) -> bool:
        """
        Check if the provided score is the best for this dataset/parameter/criterion combo.

        Args:
            dataset : DatasetConfig
                the dataset
            parameter : PostProcessorParameters
                the post-processor parameters
            criterion : str
                the criterion
            score : EvaluationScores
                the evaluation scores
        Returns:
            bool
                whether the provided score is the best for this dataset/parameter/criterion combo
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> evaluator = Evaluator()
            >>> dataset = DatasetConfig()
            >>> parameter = PostProcessorParameters()
            >>> criterion = "criterion"
            >>> score = EvaluationScores()
            >>> evaluator.is_best(dataset, parameter, criterion, score)
            False
        Note:
            This function is used to check if the provided score is the best for this dataset/parameter/criterion combo.
        """
        if not self.store_best(criterion) or math.isnan(getattr(score, criterion)):
            return False
        previous_best = self.best_scores[(dataset, parameter, criterion)]
        if previous_best is None:
            return True
        else:
            _, previous_best_score = previous_best
            return self.compare(
                getattr(score, criterion), previous_best_score, criterion
            )

    def get_overall_best(self, dataset: "DatasetConfig", criterion: str):
        """
        Return the best score for the given dataset and criterion.

        Args:
            dataset : DatasetConfig
                the dataset
            criterion : str
                the criterion
        Returns:
            float | None
                the best score for the given dataset and criterion
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> evaluator = Evaluator()
            >>> dataset = DatasetConfig()
            >>> criterion = "criterion"
            >>> evaluator.get_overall_best(dataset, criterion)
            None
        Note:
            This function is used to return the best score for the given dataset and criterion.
        """
        overall_best = None
        if self.best_scores:
            for _, parameter, _ in self.best_scores.keys():
                if (dataset, parameter, criterion) not in self.best_scores:
                    continue
                score = self.best_scores[(dataset, parameter, criterion)]
                if score is None:
                    overall_best = None
                else:
                    _, current_parameter_score = score
                    if overall_best is None:
                        overall_best = current_parameter_score
                    else:
                        if current_parameter_score:
                            if self.compare(
                                current_parameter_score, overall_best, criterion
                            ):
                                overall_best = current_parameter_score
        return overall_best

    def get_overall_best_parameters(self, dataset: "DatasetConfig", criterion: str):
        """
        Return the best parameters for the given dataset and criterion.

        Args:
            dataset : DatasetConfig
                the dataset
            criterion : str
                the criterion
        Returns:
            PostProcessorParameters | None
                the best parameters for the given dataset and criterion
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> evaluator = Evaluator()
            >>> dataset = DatasetConfig()
            >>> criterion = "criterion"
            >>> evaluator.get_overall_best_parameters(dataset, criterion)
            None
        Note:
            This function is used to return the best parameters for the given dataset and criterion.
        """
        overall_best = None
        overall_best_parameters = None
        if self.best_scores:
            for _, parameter, _ in self.best_scores.keys():
                score = self.best_scores[(dataset, parameter, criterion)]
                if score is None:
                    overall_best = None
                else:
                    _, current_parameter_score = score
                    if overall_best is None:
                        overall_best = current_parameter_score
                        overall_best_parameters = parameter
                    else:
                        if current_parameter_score:
                            if self.compare(
                                current_parameter_score, overall_best, criterion
                            ):
                                overall_best = current_parameter_score
                                overall_best_parameters = parameter
        return overall_best_parameters

    def compare(self, score_1, score_2, criterion):
        """
        Compare two scores for the given criterion.

        Args:
            score_1 : float
                the first score
            score_2 : float
                the second score
            criterion : str
                the criterion
        Returns:
            bool
                whether the first score is better than the second score for the given criterion
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> evaluator = Evaluator()
            >>> score_1 = 0.0
            >>> score_2 = 0.0
            >>> criterion = "criterion"
            >>> evaluator.compare(score_1, score_2, criterion)
            False
        Note:
            This function is used to compare two scores for the given criterion.
        """
        if self.higher_is_better(criterion):
            return score_1 > score_2
        else:
            return score_1 < score_2

    def set_best(self, validation_scores) -> None:
        """
        Find the best iteration for each dataset/post_processing_parameter/criterion.

        Args:
            validation_scores : ValidationScores
                the validation scores
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> evaluator = Evaluator()
            >>> validation_scores = ValidationScores()
            >>> evaluator.set_best(validation_scores)
            None
        Note:
            This function is used to find the best iteration for each dataset/post_processing_parameter/criterion.
            Typically, this function is called after the validation scores have been computed.
        """
        scores = validation_scores.to_xarray()

        # type these variables for mypy
        best_indexes: xr.DataArray | None
        best_scores: xr.DataArray | None

        if len(validation_scores.scores) > 0:
            best_indexes, best_scores = validation_scores.get_best(
                scores, dim="iterations"
            )
        else:
            best_indexes, best_scores = None, None
        for dataset, parameters, criterion in itertools.product(
            scores.coords["datasets"].values,
            scores.coords["parameters"].values,
            scores.coords["criteria"].values,
        ):
            if not self.store_best(criterion):
                continue
            if best_scores is None or best_indexes is None:
                self.best_scores[(dataset, parameters, criterion)] = None
            else:
                winner_index, winner_score = (
                    best_indexes.sel(
                        datasets=dataset, parameters=parameters, criteria=criterion
                    ),
                    best_scores.sel(
                        datasets=dataset, parameters=parameters, criteria=criterion
                    ),
                )
                if math.isnan(winner_score.item()):
                    self.best_scores[
                        (
                            dataset,
                            parameters,
                            criterion,
                        )
                    ] = None
                else:
                    self.best_scores[
                        (
                            dataset,
                            parameters,
                            criterion,
                        )
                    ] = (winner_index.item(), winner_score.item())

    @property
    @abstractmethod
    def criteria(self) -> list[str]:
        """
        A list of all criteria for which a model might be "best". i.e. your
        criteria might be "precision", "recall", and "jaccard". It is unlikely
        that the best iteration/post processing parameters will be the same
        for all 3 of these criteria

        Returns:
            list[str]
                the evaluation criteria
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> evaluator = Evaluator()
            >>> evaluator.criteria
            []
        Note:
            This function is used to return the evaluation criteria.
        """
        pass

    def higher_is_better(self, criterion: str) -> bool:
        """
        Wether or not higher is better for this criterion.

        Args:
            criterion : str
                the criterion
        Returns:
            bool
                whether higher is better for the given criterion
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> evaluator = Evaluator()
            >>> criterion = "criterion"
            >>> evaluator.higher_is_better(criterion)
            False
        Note:
            This function is used to determine whether higher is better for the given criterion.
        """
        return self.score.higher_is_better(criterion)

    def bounds(self, criterion: str) -> tuple[int | float | None, int | float | None]:
        """
        The bounds for this criterion

        Args:
            criterion : str
                the criterion
        Returns:
            tuple[int | float | None, int | float | None]
                the bounds for the given criterion
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> evaluator = Evaluator()
            >>> criterion = "criterion"
            >>> evaluator.bounds(criterion)
            (0, 1)
        Note:
            This function is used to return the bounds for the given criterion.
        """
        return self.score.bounds(criterion)

    def store_best(self, criterion: str) -> bool:
        """
        The bounds for this criterion

        Args:
            criterion : str
                the criterion
        Returns:
            bool
                whether to store the best score for the given criterion
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> evaluator = Evaluator()
            >>> criterion = "criterion"
            >>> evaluator.store_best(criterion)
            False
        Note:
            This function is used to return whether to store the best score for the given criterion.
        """
        return self.score.store_best(criterion)

    @property
    @abstractmethod
    def score(self) -> "EvaluationScores":
        """
        The evaluation scores.

        Returns:
            EvaluationScores
                the evaluation scores
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> evaluator = Evaluator()
            >>> evaluator.score
            EvaluationScores()
        Note:
            This function is used to return the evaluation scores.
        """
        pass
