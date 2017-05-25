from autosklearn import (
    classification,
)

from modules.AbstractLearner import AbstractLearner


class AutoLearner(AbstractLearner):

    def _train(self):
        x = self._train_features
        y = self._train_outputs

        classifier = classification.AutoSklearnClassifier(
            time_left_for_this_task=24000,
            per_run_time_limit=120,
            initial_configurations_via_metalearning=0,
            tmp_folder='./astmp',
            delete_tmp_folder_after_terminate=False,
            output_folder='./asout',
            delete_output_folder_after_terminate=False,
            seed=1742,
        )
        classifier.fit(x, y, dataset_name='LIStask4-labeled')
        print(classifier.show_models())
        self._model = classifier.predict
