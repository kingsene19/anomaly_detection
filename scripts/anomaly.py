from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, balanced_accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Traitement des features
# Traitement des features
def handle_features(dataframe, target_name):
    """
    Transforme un DataFrame en séparant les caractéristiques numériques et catégoriques. 
    Applique RobustScaler pour les colonnes numériques et OrdinalEncoder pour les colonnes catégoriques. 
    Divise les données en ensembles d'entraînement et de test.

    Args:
        dataframe (pd.DataFrame): Le DataFrame à traiter.
        target_name (str): Le nom de la colonne cible.

    Returns:
        tuple: Contient les ensembles (X_train, X_test), (y_train, y_test) et les transformateurs (numerical_transformer, categorical_transformer).
    """
    feats = dataframe.drop(columns=target_name)
    y = dataframe[target_name]

    numerical_transformer = RobustScaler()
    categorical_transformer = OrdinalEncoder()

    num_feats = numerical_transformer.fit_transform(feats.select_dtypes(include=['number']))
    cat_feats = categorical_transformer.fit_transform(feats.select_dtypes(include=['category','object','bool']))
    if num_feats.shape[1] == 0 and cat_feats.shape[1] > 0:
        X = cat_feats
    elif cat_feats.shape[1] == 0 and num_feats.shape[1] > 0:
        X = num_feats
    else:
        X = np.hstack((num_feats, cat_feats))
    
    y = np.array([1 if value==0 else -1 for value in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.2, stratify=y)

    return (X_train, X_test), (y_train, y_test), (numerical_transformer, categorical_transformer)

# Fonction pour oversampling
def create_oversampled(X_train, X_test, y_train, y_test):
    """
    Applique la méthode SMOTE pour augmenter la classe minoritaire dans l'ensemble d'entraînement.

    Args:
        X_train (np.ndarray): Caractéristiques d'entraînement.
        X_test (np.ndarray): Caractéristiques de test.
        y_train (np.ndarray): Labels d'entraînement.
        y_test (np.ndarray): Labels de test.

    Returns:
        tuple: Contient les ensembles sur-échantillonnés (X_train, X_test) et (y_train, y_test).
    """
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    return (X_train, X_test), (y_train, y_test)

# Fonction pour undersampling
def create_undersampled(X_train, X_test, y_train, y_test):
    """
    Applique RandomUnderSampler pour réduire la classe majoritaire dans l'ensemble d'entraînement.

    Args:
        X_train (np.ndarray): Caractéristiques d'entraînement.
        X_test (np.ndarray): Caractéristiques de test.
        y_train (np.ndarray): Labels d'entraînement.
        y_test (np.ndarray): Labels de test.

    Returns:
        tuple: Contient les ensembles sous-échantillonnés (X_train, X_test) et (y_train, y_test).
    """
    under = RandomUnderSampler(random_state=42)
    X_train, y_train = under.fit_resample(X_train, y_train)
    return (X_train, X_test), (y_train, y_test)


# Recherche des meilleurs hyperparamètres
def calculate_f1(estimator, X, y):
    """
    Calcule le score F1 en utilisant le modèle fourni.

    Args:
        estimator: Modèle pour lequel calculer le F1 score.
        X (np.ndarray): Caractéristiques.
        y (np.ndarray): Labels.

    Returns:
        float: Le score F1.
    """
    if hasattr(estimator, "fit_predict"):
        return f1_score(y, estimator.fit_predict(X))
    else:
        return f1_score(y, estimator.predict(X))


def get_best_params(estimator, params, X, y, n_iter=10, cv=5, is_supervised=False, random_state=42):
    """
    Recherche les meilleurs hyperparamètres pour un modèle donné à l'aide de RandomizedSearchCV.

    Args:
        estimator: Le modèle à optimiser.
        params (dict): Distribution des hyperparamètres.
        X (np.ndarray): Caractéristiques.
        y (np.ndarray): Labels.
        n_iter (int): Nombre d'itérations pour la recherche.
        cv (int): Nombre de folds pour la validation croisée.
        is_supervised (bool): Indique si la tâche est supervisée.
        random_state (int): Graine pour la reproductibilité.

    Returns:
        dict: Les meilleurs hyperparamètres trouvés.
    """
    if is_supervised and y is None:
        raise ValueError("For supervised learning, 'y' must be provided.")
    
    model_instance = estimator()
    random_search = RandomizedSearchCV(
        estimator=model_instance,
        param_distributions=params,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv),
        random_state=random_state,
        n_jobs=-1,
        scoring=calculate_f1
    )
    print(f"Demarrage recherche hyperparametres pour {estimator.__name__}")
    random_search.fit(X,y)
    best_params = random_search.best_params_
    print(f"Done! Meilleurs hyperparamètres pour {estimator.__name__}", best_params)
    return best_params


# Evaluation avec stratified KFold
def run_stratified_kflod(model, X, y, n_splits=5, is_supervised=False, random_state=42):
    """
    Effectue une validation croisée stratifiée pour évaluer un modèle en termes de performances AUC ROC, AUC PR, F1, et Balanced Accuracy.

    Args:
        model: Modèle à évaluer.
        X (np.ndarray): Caractéristiques.
        y (np.ndarray): Labels.
        n_splits (int): Nombre de folds.
        is_supervised (bool): Indique si la tâche est supervisée.
        random_state (int): Graine pour la reproductibilité.

    Returns:
        tuple: Moyennes des scores AUC ROC, AUC PR, F1, et Balanced Accuracy.
    """

    if is_supervised and y is None:
        raise ValueError("For supervised learning, 'y' must be provided.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    auc_roc_scores = []
    auc_pr_scores = []
    f1_scores = []
    balanced_accs = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        if is_supervised:
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            preds_proba = model.predict_proba(X_val)[:, 1]
        else:
            preds = model.fit_predict(X_train)


        if is_supervised:
            auc_roc = roc_auc_score(y_val, preds_proba)
            auc_pr = average_precision_score(y_val, preds_proba)
            f1 = f1_score(y_val, preds)
            balanced_acc = balanced_accuracy_score(y_val, preds)
        else:
            if model.__class__.__name__ == "IsolationForest":
                scores = model.decision_function(X_train)
            else:
                scores = model.negative_outlier_factor_
            probas = 1 / (1 + np.exp(-scores))
            auc_roc = roc_auc_score(y_train, probas)
            auc_pr = average_precision_score(y_train, probas)
            f1 = f1_score(y_train, preds)
            balanced_acc = balanced_accuracy_score(y_train, preds)

        auc_roc_scores.append(auc_roc)
        auc_pr_scores.append(auc_pr)
        f1_scores.append(f1)
        balanced_accs.append(balanced_acc)

    mean_auc_roc = np.mean(auc_roc_scores)
    mean_auc_pr = np.mean(auc_pr_scores)
    mean_f1 = np.mean(f1_scores)
    mean_balanced_acc = np.mean(balanced_accs)

    return mean_auc_roc, mean_auc_pr, mean_f1, mean_balanced_acc

# Calculer les metriques du meilleur modèle (roc_auc, average_precision_f1) et afficher les courbes ROC et Precision/Recall
def get_best_model_performance(model, X_test, y_test):
    """
    Calcule les métriques de performance pour un modèle donné et affiche les courbes ROC et Precision-Recall.

    Args:
        model: Modèle à évaluer.
        X_train (np.ndarray): Données d'entraînement.
        y_train (np.ndarray): Labels d'entraînement.
        X_test (np.ndarray): Données de test.
        y_test (np.ndarray): Labels de test.

    Returns:
        dict: Contient les scores ROC AUC, PR AUC, F1 et Balanced Accuracy.
    """

    if model.__class__.__name__ not in ["LocalOutlierFactor", "IsolationForest"]:
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    else:
        y_pred = model.fit_predict(X_test)
        if model.__class__.__name__ == "IsolationForest":
            scores = model.decision_function(X_test)
        else:
            scores = model.negative_outlier_factor_
        y_pred_prob = 1 / (1 + np.exp(-scores))
    
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    pr_auc = average_precision_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"ROC AUC Score: {roc_auc:.2f}")
    print(f"Average Precision Score: {pr_auc:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Balanced Acc: {balanced_acc}")
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))    

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    disp.plot()
    plt.tight_layout()
    plt.show()
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1_score': f1,
        'balanced_acc': balanced_acc
    }

# Automatisation de la comparaison des modèles
def compare_models_anomaly(dataframe, target_name, estimator_params, n_iter=10, cv=5, n_splits=5, compare_with='f1'):
    """
    Compare plusieurs modèles pour une tâche d'anomalie en utilisant différentes métriques comme le F1 score, AUC ROC ou PR AUC.

    Args:
        dataframe (pd.DataFrame): Le DataFrame contenant les données.
        target_name (str): Nom de la colonne cible.
        estimator_params (dict): Dictionnaire contenant les modèles et hyperparamètres.
        n_iter (int): Nombre d'itérations pour RandomizedSearchCV.
        cv (int): Nombre de folds pour la validation croisée.
        n_splits (int): Nombre de splits pour Stratified KFold.
        compare_with (str): Métrique de comparaison ("f1", "auc_roc", "auc_pr" ou "balanced_acc").

    Returns:
        None: Affiche les résultats et le meilleur modèle.
    """
    (X_train, X_test), (y_train, y_test), transformers = handle_features(dataframe, target_name)
    best_model = None
    best_score = float('-inf')

    for model_name, config in estimator_params.items():
        model = config['estimator']
        params = config['params']
        is_supervised = config['is_supervised']
        undersampling = config['undersampling']
        oversampling = config['oversampling']

        if undersampling:
            (X_train, X_test), (y_train, y_test) = create_undersampled(X_train, X_test, y_train, y_test)
        if oversampling:
            (X_train, X_test), (y_train, y_test) = create_oversampled(X_train, X_test, y_train, y_test)

        best_params = get_best_params(model, params, X_train, y_train, n_iter, cv, is_supervised=is_supervised)
        model = model(**best_params)

        mean_auc_roc, mean_auc_pr, mean_f1, mean_balanced_acc = run_stratified_kflod(model, X_train, y_train, n_splits=n_splits, is_supervised=is_supervised)

        print(f"Performance du meilleur modele: {model_name}")
        print(f"Mean AUC ROC: {mean_auc_roc:.2f}")
        print(f"Mean AUC PR: {mean_auc_pr:.2f}")
        print(f"Mean F1 Score: {mean_f1:.2f}")
        print(f"Mean Balanced Acc: {mean_balanced_acc:.2f}")
        
        if compare_with == "f1":
            if mean_f1 > best_score:
                best_score = mean_f1
                best_model = model
        elif compare_with == "auc_roc":
            if mean_auc_roc > best_score:
                best_score = mean_auc_roc
                best_model = model
        elif compare_with == "auc_pr":
            if mean_auc_pr > best_score:
                best_score = mean_auc_pr
                best_model = model
        else:
            if mean_balanced_acc > best_score:
                best_score = mean_balanced_acc
                best_model = model

    if best_model is not None:
        print("---------------------------------- FINAL RESULTS ------------------------------------------")
        print(f"Meilleur modele après comparaison de toutes les approches: {best_model.__class__.__name__}")
        get_best_model_performance(best_model, X_test, y_test)