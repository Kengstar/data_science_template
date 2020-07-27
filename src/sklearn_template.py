import sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def build_pipeline(steps, ml_component):
    """[summary]

    Args:
        step ([type]): [description]
        ml_component ([type]): [description]

    Returns:
        [type]: [description]
    """
    processing_steps = list(zip(steps, ml_component))
    pipeline = None
    return pipeline


x,y = None, None ### load dataset in src/datasets
###just use numpy format, pf.values or as matrix 

### add preprocessing in Pipeline, write sklearn api, 
model = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
model = LogisticRegression(solver='liblinear', class_weight={0:1,1:8})

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

kfold = model_selection.KFold(n_splits=5, random_state=42)

results = model_selection.cross_validate(estimator=model,
                                          X=x,
                                          y=y,
                                          cv=kfold,
                                          scoring=scoring)
np.mean(results['test_recall']), np.mean(results['test_precision']), np.mean(results['test_f1_score'])
print(results)

def run_experiment():
    pass


if __name__ == "__main__":
    run_experiment()