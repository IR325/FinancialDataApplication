class AdvancedKfold:
    def __init__(self, n_splits, random_state, shuffle):
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.past_train_indicies = []
        self.past_eval_indicies = []
        self.past_valid_indicies = []

    def split(self, X: pd.DataFrame):
        indicies = list(X.index)
        n_sample = int(len(indicies) / self.n_splits)
        for i in range(self.n_splits - 1):
            # 抽出
            eval_indicies = random.sample(list(set(indicies) - set(self.past_eval_indicies)), n_sample)
            valid_indicies = random.sample(list(set(indicies) - set(eval_indicies) - set(self.past_valid_indicies)), n_sample)
            train_indicies = list(set(indicies) - set(eval_indicies) - set(valid_indicies))
            yield train_indicies, eval_indicies, valid_indicies
        eval_indicies = list(set(indicies) - set(self.past_eval_indicies))
        valid_indicies = list(set(indicies) - set(eval_indicies) - set(self.past_valid_indicies))
        train_indicies = list(set(indicies) - set(eval_indicies) - set(valid_indicies))
        yield train_indicies, eval_indicies, valid_indicies
