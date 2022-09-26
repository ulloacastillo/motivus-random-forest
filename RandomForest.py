import pickle
from motivus.client import Client
import asyncio
import time


async def main(task_definition):
    motivus = await Client.connect()
    task_id = motivus.call_async(task_definition)
    task = motivus.select_task(task_id)
    return await task


class RandomForestClassifier:
    def __init__(self, model_path=None, n_estimators=100, min_samples_split=2, max_depth=3, n_features=4, seed=41) -> None:
        if model_path:
            objects = []
            with (open(model_path, "rb")) as f:
                while True:
                    try:
                        objects.append(pickle.load(f))
                    except EOFError:
                        break

            self.random_forest = objects[0]
        else:
            self.random_forest = None
            self.n_estimators = n_estimators
            self.min_samples_split = min_samples_split
            self.max_depth = max_depth
            self.n_features = n_features
            self.seed = seed

    def predict(self, X):
        if self.random_forest:
            task_definition = {
                # local build
                # 'wasm_path': "build/random-forest-0.1.0.wasm",
                # 'loader_path': "build/random-forest-0.1.0.js",
                # published build
                'algorithm': "random-forest",
                'algorithm_version': "0.1.0",
                'params': [self.random_forest, X, [], "predict"]
            }
            result = asyncio.run(main(task_definition))
            print(result)
        else:
            raise Exception('You cannot predidict without a trained forest')

    def fit(self, X, Y):
        if not self.random_forest:
            task_definition = {
                # local build
                # 'wasm_path': "build/random-forest-0.1.0.wasm",
                # 'loader_path': "build/random-forest-0.1.0.js",
                # published build
                'algorithm': "random-forest",
                'algorithm_version': "0.1.0",
                'params': [{'n_trees': self.n_estimators, 'min_samples_split': self.min_samples_split, 'max_depth': self.max_depth, 'n_feats': self.n_features, 'seed': self.seed}, X, Y, "train"]
            }

            result = asyncio.run(main(task_definition))
            print(result)
            self.random_forest = result

        else:
            raise Exception('Hyperparameters not found')
