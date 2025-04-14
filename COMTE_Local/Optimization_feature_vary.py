import logging
import multiprocessing
import numbers
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from skopt import gbrt_minimize, gp_minimize
from TSInterpret.InterpretabilityModels.counterfactual.COMTE.Problem import (
    LossDiscreteState,
    Problem,
)
from TSInterpret.InterpretabilityModels.counterfactual.COMTE.Optimization_helpers import (
    random_hill_climb,
)


class BaseExplanation:
    def __init__(
        self,
        clf,
        timeseries,
        labels,
        silent=True,
        num_distractors=2,
        dont_stop=False,
        threads=multiprocessing.cpu_count(),
        features_to_vary="all",  # 新添加的参数,
    ):
        self.clf = clf
        self.timeseries = timeseries
        self.labels = pd.DataFrame(labels, columns=["label"])
        self.silent = silent
        self.num_distractors = num_distractors
        self.dont_stop = dont_stop
        self.window_size = timeseries.shape[-1]
        self.channels = timeseries.shape[-2]
        self.ts_min = np.repeat(timeseries.min(), self.window_size)
        self.ts_max = np.repeat(timeseries.max(), self.window_size)
        self.ts_std = np.repeat(timeseries.std(), self.window_size)
        self.tree = None
        self.per_class_trees = None
        self.threads = threads
        self.features_to_vary = features_to_vary  # 存储允许修改的特征

    def explain(self, x_test, **kwargs):
        raise NotImplementedError("Please don't use the base class directly")

    def construct_per_class_trees(self):
        if self.per_class_trees is not None:
            return
        self.per_class_trees = {}
        self.per_class_node_indices = {c: [] for c in np.unique(self.labels)} #每个类别label生成[]，例如 {0: [], 1: [], 2: [], 3: []}
        input_ = self.timeseries #输入序列

        preds = np.argmax(self.clf(input_), axis=1) #预测结果，与labels有区别
        true_positive_node_ids = {c: [] for c in np.unique(self.labels)}#存储#每个类别label下的TP
        for pred, (idx, row) in zip(preds, self.labels.iterrows()):
            if row["label"] == pred:
                true_positive_node_ids[pred].append(idx)
        for c in np.unique(self.labels): #遍历所有类别
            dataset = []
            for node_id in true_positive_node_ids[c]: #每一类中TP的样本编号
                dataset.append(self.timeseries[[node_id], :, :].T.flatten()) #1，T，C展平为T*C
                self.per_class_node_indices[c].append(node_id)
            if len(dataset) != 0:
                self.per_class_trees[c] = KDTree(np.stack(dataset)) #针对第c类，构建KD树
            else:
                self.per_class_trees[c] = []
                logging.warning(
                    f"Due to lack of true postitives for class {c} no kd-tree could be build."
                )

    def construct_tree(self):
        if self.tree is not None:
            return
        train_set = []
        self.node_indices = []
        for node_id in self.timeseries.index.get_level_values("node_id").unique():
            train_set.append(self.timeseries[[node_id], :, :].T.flatten())
            self.node_indices.append(node_id)
        self.tree = KDTree(np.stack(train_set))

    def _get_distractors(self, x_test, to_maximize, n_distractors=2):
        self.construct_per_class_trees()
        # to_maximize can be int, string or np.int64
        if isinstance(to_maximize, numbers.Integral):#to_maximize：原始预测中概率第二大的类别，用于生成反事实
            to_maximize = np.unique(self.labels)[to_maximize]
        distractors = []
        for idx in (
            self.per_class_trees[to_maximize] #第二大类的KD树
            .query(x_test.T.flatten().reshape(1, -1), k=n_distractors)[1] #通过KD树在to_maximize这类中找出最类似的k个distractors
            .flatten()
        ):
            distractors.append( #idx为最相似样本的编号
                self.timeseries[[self.per_class_node_indices[to_maximize][idx]], :, :]
            )

        return distractors

    def local_lipschitz_estimate(
        self,
        x,
        optim="gp",
        eps=None,
        bound_type="box",
        clip=True,
        n_calls=100,
        njobs=-1,
        verbose=False,
        exp_kwargs=None,
        n_neighbors=None,
    ):
        np_x = x.T.flatten()
        if n_neighbors is not None and self.tree is None:
            self.construct_tree()

        # Compute bounds for optimization
        if eps is None:
            # If want to find global lipzhitz ratio maximizer
            # search over "all space" - use max min bounds of dataset
            # fold of interest
            lwr = self.ts_min.flatten()
            upr = self.ts_max.flatten()
        elif bound_type == "box":
            lwr = (np_x - eps).flatten()
            upr = (np_x + eps).flatten()
        elif bound_type == "box_std":
            lwr = (np_x - eps * self.ts_std).flatten()
            upr = (np_x + eps * self.ts_std).flatten()
        if clip:
            lwr = lwr.clip(min=self.ts_min.min())
            upr = upr.clip(max=self.ts_max.max())
        if exp_kwargs is None:
            exp_kwargs = {}

        consts = []
        bounds = []
        variable_indices = []
        for idx, (l, u) in enumerate(zip(*[lwr, upr])):
            if u == l:
                consts.append(l)
            else:
                consts.append(None)
                bounds.append((l, u))
                variable_indices.append(idx)
        consts = np.array(consts)
        variable_indices = np.array(variable_indices)

        orig_explanation = set(self.explain(x, **exp_kwargs))
        if verbose:
            logging.info("Original explanation: %s", orig_explanation)

        def lipschitz_ratio(y):
            nonlocal self
            nonlocal consts
            nonlocal variable_indices
            nonlocal orig_explanation
            nonlocal np_x
            nonlocal exp_kwargs

            if len(y) == len(consts):
                # For random search
                consts = y
            else:
                # Only search in variables that vary
                np.put_along_axis(consts, variable_indices, y, axis=0)
            df_y = pd.DataFrame(
                np.array(consts).reshape((len(self.metrics), self.window_size)).T,
                columns=self.metrics,
            )
            df_y = pd.concat([df_y], keys=["y"], names=["node_id"])
            new_explanation = set(self.explain(df_y, **exp_kwargs))
            # Hamming distance
            exp_distance = len(orig_explanation.difference(new_explanation)) + len(
                new_explanation.difference(orig_explanation)
            )
            # Multiply by 1B to get a sensible number
            ratio = exp_distance * -1e9 / np.linalg.norm(np_x - consts)
            if verbose:
                logging.info("Ratio: %f", ratio)
            return ratio

        # Run optimization
        min_ratio = 0
        worst_case = np_x
        if n_neighbors is not None:
            for idx in self.tree.query(np_x.reshape(1, -1), k=n_neighbors)[1].flatten():
                y = self.timeseries[[self.node_indices[idx]], :, :].T.flatten()
                ratio = lipschitz_ratio(y)
                if ratio < min_ratio:
                    min_ratio = ratio
                    worst_case = y
            if verbose:
                logging.info("The worst case explanation was for %s", idx)
        elif optim == "gp":
            logging.info("Running BlackBox Minimization with Bayesian Opt")
            # Need minus because gp only has minimize method
            res = gp_minimize(
                lipschitz_ratio, bounds, n_calls=n_calls, verbose=verbose, n_jobs=njobs
            )
            min_ratio, worst_case = res["fun"], np.array(res["x"])
        elif optim == "gbrt":
            logging.info("Running BlackBox Minimization with GBT")
            res = gbrt_minimize(
                lipschitz_ratio, bounds, n_calls=n_calls, verbose=verbose, n_jobs=njobs
            )
            min_ratio, worst_case = res["fun"], np.array(res["x"])
        elif optim == "random":
            for i in range(n_calls):
                y = (upr - lwr) * np.random.random(len(np_x)) + lwr
                ratio = lipschitz_ratio(y)
                if ratio < min_ratio:
                    min_ratio = ratio
                    worst_case = y

        if len(worst_case) != len(consts):
            np.put_along_axis(consts, variable_indices, worst_case, axis=0)

        return min_ratio, consts


CLASSIFIER = None
X_TEST = None
DISTRACTOR = None


def _eval_one(tup):
    column, label_idx = tup
    global CLASSIFIER
    global X_TEST
    global DISTRACTOR
    x_test = X_TEST.copy()
    x_test[0][column] = DISTRACTOR[0][column] #将x_test对应列进行修改
    input_ = x_test.reshape(1, -1, x_test.shape[-1]) #变为正常序列格式

    return CLASSIFIER(input_)[0][
        label_idx
    ]  # CLASSIFIER.predict_proba(x_test)[0][label_idx]


class BruteForceSearch(BaseExplanation):
    def _find_best(self, x_test, distractor, label_idx):
        global CLASSIFIER
        global X_TEST
        global DISTRACTOR
        CLASSIFIER = self.clf
        X_TEST = x_test
        DISTRACTOR = distractor
        input_ = x_test
        best_case = self.clf(input_)[0][label_idx] #best_case为原本x_test在第二大类上的概率#label_idx为原始对x_test预测中概率第二大的类别，用于生成反事实
        best_column = None
        tuples = []
        # for c in range(0, self.channels):
        #     if np.any(distractor[0][c] != x_test[0][c]):
        #         tuples.append((c, label_idx))
        
        # 仅考虑features_to_vary中的特征
        if self.features_to_vary == "all":
            columns = range(0, self.channels)
        else:
            columns = self.features_to_vary
        
        for c in columns:
            if np.any(distractor[0][c] != x_test[0][c]): #记录允许修改的特征中,(t,)长度的时间序列，x_test与distractor不同的是哪些feature
                tuples.append((c, label_idx)) #记录对应的特征和distractor对应的类别
        
        
        
        if self.threads == 1: #单线程，windows默认情况
            results = []
            for t in tuples:
                results.append(_eval_one(t)) #每次修改x_test的一列为distractor的对应列，并计算成为第二大类的概率
        else:
            pool = multiprocessing.Pool(self.threads)
            results = pool.map(_eval_one, tuples)
            pool.close()
            pool.join()
        for (c, _), pred in zip(tuples, results):
            if pred > best_case: #如果修改对应列成为第二大类的概率，大于了原始的best_case，则替换，类似冒泡，选出更改使得概率最大的一列
                best_column = c
                best_case = pred
        if not self.silent:
            logging.info("Best column: %s, best case: %s", best_column, best_case)
        return best_column, best_case

    def explain(self, x_test, to_maximize=None, num_features=10):
        input_ = x_test #np.array,要解释的样本
        orig_preds = self.clf(input_)
        if to_maximize is None:
            to_maximize = np.argsort(orig_preds)[0][-2:-1][0] #原始对x_test预测中概率第二大的类别，用于生成反事实
        print('\n')   
        print("orig_preds: ", orig_preds)
        print("cf target: ", to_maximize)
        orig_label = np.argmax(self.clf(input_)) #原始预测概率最大的实际类别
        if orig_label == to_maximize:
            print("Original and Target Label are identical !")
            return None, None
        distractors = self._get_distractors(
            x_test, to_maximize, n_distractors=self.num_distractors
        ) #在第二大类别中，通过KD树选取与x_test最相似的k个样本作为distractors
        # print('distractores',distractors)
        best_explanation = set()
        best_change_feats = np.inf#最优特征数量改动，类似loss，从无穷开始降低
        best_explanation_score = 0 
        final_modified = None #类似冒泡，记录中场最佳用于比较
        
        for count, dist in enumerate(distractors):#从所有distractors中选出最合适的，count为第几个distractor，dist为distractors的缩写
            print('\n')    
            print('This is the distractor: ', count)    
            explanation = []
            modified = x_test.copy()
            prev_best = 0
            # best_dist = dist
            while True:
                input_ = modified #需要一次次修改的x_test
                probas = self.clf(input_) 
                if np.argmax(probas) == to_maximize:
                    #如果修改后的modified预测已经改变了类别，成为了第二大类的
                    current_best = np.max(probas)
                    print('current len(explanation): ', len(explanation))
                    print('current_best: ', current_best)
                    if (current_best > best_explanation_score and len(explanation) <= best_change_feats):
                        #作为这个distractor提供的改动最少列能得到的反事实概率大于之前的，且当前distractors要求修改的特征数量小于等于之前最优的
                        print('Update final_modified')
                        print('Feature to change: ', explanation)
                        best_explanation = explanation #不断更新的全局最优反事实修改的特征
                        best_explanation_score = current_best #不断更新的全局最优反事实概率
                        final_modified = modified.copy() #不断更新的全局最优反事实改动
                        best_change_feats = len(explanation)#不断更新的全局最优改动特征数量
                    if current_best <= prev_best:
                        break
                    prev_best = current_best #上一个distractor带来的最优概率
                    if not self.dont_stop:
                        break
                if (
                    not self.dont_stop
                    and len(best_explanation) != 0
                    and len(explanation) >= len(best_explanation)#如果当前distractors要求修改的特征数量比之前的还要大，就废除
                ):
                    break
                best_column, _ = self._find_best(modified, dist, to_maximize)#一列一列的换，找出反事实至第二大类概率最大的特征
                if best_column is None:
                    break

                modified[0][best_column] = dist[0][best_column]#将modified这个x_test的替身，对应列进行修改，直接把对应特征的(t,)的列更换为distractor的
                explanation.append(best_column)
        other = final_modified
        # other = modified
        target = np.argmax(self.clf(other), axis=1)
        return other, target


class OptimizedSearch(BaseExplanation):
    def __init__(
        self,
        clf,
        timeseries,
        labels,
        silent,
        threads,
        num_distractors,
        max_attempts,
        maxiter,
        **kwargs,
    ):
        super().__init__(clf, timeseries, labels, **kwargs)
        self.discrete_state = False
        self.backup = BruteForceSearch(clf, timeseries, labels, **kwargs)
        self.max_attemps = max_attempts
        self.maxiter = maxiter

    def opt_Discrete(self, to_maximize, x_test, dist, columns, init, num_features=None):
        fitness_fn = LossDiscreteState(
            to_maximize,
            self.clf,
            x_test,
            dist,
            columns,
            reg=0.8,
            max_features=num_features,
            maximize=False,
        )
        problem = Problem(length=len(columns), loss=fitness_fn, max_val=2)
        best_state, best_fitness = random_hill_climb(
            problem,
            max_attempts=self.max_attemps,
            max_iters=self.maxiter,
            init_state=init,
            restarts=5,
        )

        self.discrete_state = True
        return best_state

    def _prune_explanation(
        self, explanation, x_test, dist, to_maximize, max_features=None
    ):
        if max_features is None:
            max_features = len(explanation)
        short_explanation = set()
        while len(short_explanation) < max_features:
            modified = x_test.copy()
            for c in short_explanation:
                modified[0][c] = dist[0][c]
            input_ = modified
            prev_proba = self.clf(input_)[0][to_maximize]
            best_col = None
            best_diff = 0
            for c in explanation:
                tmp = modified.copy()

                tmp[0][c] = dist[0][c]
                input_ = tmp
                cur_proba = self.clf(input_)[0][to_maximize]
                if cur_proba - prev_proba > best_diff:
                    best_col = c
                    best_diff = cur_proba - prev_proba
            if best_col is None:
                break
            else:
                short_explanation.add(best_col)
        return short_explanation

    def explain(
        self, x_test, num_features=None, to_maximize=None
    ) -> Tuple[np.array, int]:
        input_ = x_test
        orig_preds = self.clf(input_)

        orig_label = np.argmax(orig_preds)

        if to_maximize is None:
            to_maximize = np.argsort(orig_preds)[0][-2:-1][0]

        if orig_label == to_maximize:
            print("Original and Target Label are identical !")
            return None, None

        explanation = self._get_explanation(x_test, to_maximize, num_features)
        tr, _ = explanation
        if tr is None:
            print("Run Brute Force as Backup.")
            explanation = self.backup.explain(
                x_test, num_features=num_features, to_maximize=to_maximize
            )
        best, other = explanation
        target = np.argmax(self.clf(best), axis=1)

        return best, target

    def _get_explanation(self, x_test, to_maximize, num_features):
        distractors = self._get_distractors(
            x_test, to_maximize, n_distractors=self.num_distractors
        )

        # Avoid constructing KDtrees twice
        self.backup.per_class_trees = self.per_class_trees
        self.backup.per_class_node_indices = self.per_class_node_indices

        best_explanation = set()
        best_explanation_score = 0

        for count, dist in enumerate(distractors):
            columns = [
                c for c in range(0, self.channels) if np.any(dist[0][c] != x_test[0][c])
            ]

            # Init options
            init = [0] * len(columns)

            result = self.opt_Discrete(
                to_maximize, x_test, dist, columns, init=init, num_features=num_features
            )

            if not self.discrete_state:
                explanation = {
                    x for idx, x in enumerate(columns) if idx in np.nonzero(result.x)[0]
                }
            else:
                explanation = {
                    x for idx, x in enumerate(columns) if idx in np.nonzero(result)[0]
                }

            explanation = self._prune_explanation(
                explanation, x_test, dist, to_maximize, max_features=num_features
            )

            modified = x_test.copy()

            for c in columns:
                if c in explanation:
                    modified[0][c] = dist[0][c]
            input_ = modified  # .reshape(1, -1, self.window_size)
            probas = self.clf(input_)

            if not self.silent:
                logging.info("Current probas: %s", probas)
            if np.argmax(probas) == to_maximize:
                current_best = np.max(probas)
                if current_best > best_explanation_score:
                    best_explanation = explanation
                    best_explanation_score = current_best
                    best_modified = modified

        if len(best_explanation) == 0:
            return None, None

        return best_modified, best_explanation
