import asyncio

class RandomForest:
    # n_trees: usize, min_samples_split: usize, max_depth: usize, n_feats: usize, seed: u64)
    def __init__(self, n_estimators=100, max_depth=3, min_samples_split=2, n_feats=1, seed=0):
        self.n_estimators = n_estimators
        self.max_depth=max_depth
        self.min_samples_split = min_samples_split
        self.n_feats = n_feats
        self.seed = seed
    
    def fit(self, X, Y):

        result = asyncio.run(main())