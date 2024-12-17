from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, balanced_accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Traitement des features
def handle_features(dataframe, target_name):
    """
    Traitement des features du jeu de données.

    Args:
        dataframe (pd.DataFrame): Le jeu de données contenant les features et la cible.
        target_name (str): Le nom de la colonne cible.
        is_lof (bool, optional): Indique si le traitement concerne un modèle LOF. Par défaut False.

    Returns:
        tuple: Contient (X_train, X_test), (y_train, y_test) et les transformateurs numériques et catégoriels.
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

# Recherche des meilleurs hyperparams
def get_best_params(estimator, params, X, y, n_iter=10, validation_size=0.2, random_state=42):
    """
    Recherche les meilleurs hyperparamètres pour un modèle en utilisant une recherche aléatoire.
    Adapte le comportement pour LocalOutlierFactor (entraînement uniquement sur y==1) 
    et IsolationForest (entraînement normal).

    Args:
        estimator (class): La classe du modèle pour laquelle effectuer la recherche.
        params (dict): Le dictionnaire des distributions de paramètres.
        X (array-like): Données d'entrée pour l'entraînement.
        y (array-like): Vraies étiquettes des données (1 = normal, 0 = anormal).
        n_iter (int, optional): Nombre d'itérations de la recherche aléatoire. Par défaut 10.
        validation_size (float, optional): Taille de l'ensemble de validation. Par défaut 0.3.
        random_state (int, optional): Graine aléatoire. Par défaut 42.

    Returns:
        dict: Les meilleurs paramètres trouvés.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_size, random_state=random_state, stratify=y
    )
    
    print(f"Demarrage recherche hyperparametres pour {estimator.__name__}")
    if estimator.__name__ == "LocalOutlierFactor":
        normal_indices = np.where(y_train == 1)[0]
        X_train_filtered = X_train[normal_indices]
        y_train_filtered = y_train[normal_indices]
    else:
        X_train_filtered = X_train
        y_train_filtered = y_train

    best_params = None
    best_score = -np.inf
    for _ in range(n_iter):
        current_params = {key: np.random.choice(values) if hasattr(values, '__iter__') else values.rvs()
                          for key, values in params.items()}

        model = estimator(**current_params)

        model.fit(X_train_filtered, y_train_filtered)

        scores = model.decision_function(X_val)
        
        current_score = average_precision_score(y_val, scores)

        if current_score > best_score:
            best_score = current_score
            best_params = current_params
    print(f"Meilleurs hyperparamètres pour {estimator.__name__} : {best_params}")
    return best_params

# Evaluer chaque approche
def run_stratified_kflod(model, X, y, n_splits=5, random_state=42):
    """
    Évalue un modèle en utilisant Stratified K-Fold et retourne les métriques moyennes.

    Args:
        model (object): Le modèle à évaluer.
        X (array-like): Données d'entrée.
        y (array-like): Étiquettes des données.
        n_splits (int, optional): Nombre de splits pour la validation croisée. Par défaut 5.
        random_state (int, optional): La graine aléatoire. Par défaut 42.

    Returns:
        tuple: Contient les moyennes de ROC AUC, PR AUC, F1 Score et précision équilibrée.
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    auc_roc_scores = []
    auc_pr_scores = []
    f1_scores = []
    balanced_accs = []

    for train_index, val_index in skf.split(X, y):

        if model.__class__.__name__ == "LocalOutlierFactor":
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            normal_indices = np.where(y_train == 1)[0]
            X_train_normal = X_train[normal_indices]
            model.fit(X_train_normal)
            preds = model.predict(X_val)
        else:
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            model.fit(X_train)
            preds = model.predict(X_val)

        scores = model.decision_function(X_val)
        probas = 1 / (1 + np.exp(-scores))
        auc_roc = roc_auc_score(y_val, probas)
        auc_pr = average_precision_score(y_val, probas)
        f1 = f1_score(y_val, preds)
        balanced_acc = balanced_accuracy_score(y_val, preds)

        auc_roc_scores.append(auc_roc)
        auc_pr_scores.append(auc_pr)
        f1_scores.append(f1)
        balanced_accs.append(balanced_acc)

    mean_auc_roc = np.mean(auc_roc_scores)
    mean_auc_pr = np.mean(auc_pr_scores)
    mean_f1 = np.mean(f1_scores)
    mean_balanced_acc = np.mean(balanced_accs)

    return mean_auc_roc, mean_auc_pr, mean_f1, mean_balanced_acc

# Evaluer le meilleur modèle trouvé
def get_best_model_performance(model, X_test, y_test):
    """
    Calcule les métriques pour le meilleur modèle et affiche les courbes ROC et Precision-Recall.

    Args:
        model (object): Le modèle à évaluer.
        X_test (array-like): Données de test.
        y_test (array-like): Étiquettes des données de test.

    Returns:
        dict: Contient les métriques (roc_auc, pr_auc, f1_score, balanced_acc).
    """
    y_pred = model.predict(X_test)
    scores = model.decision_function(X_test)
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
def compare_models_novelty(dataframe, target_name, estimator_params, n_iter=10, cv=5, n_splits=5, compare_with='auc_pr'):
    """
    Automatise la comparaison de plusieurs modèles pour des tâches de détection de nouveauté (novelty detection).

    Args:
        dataframe (pd.DataFrame): Le jeu de données contenant les features et la cible.
        target_name (str): Le nom de la colonne cible.
        estimator_params (dict): Dictionnaire de configuration pour les modèles (estimator, params).
        n_iter (int, optional): Nombre d'itérations pour la recherche des hyperparamètres. Par défaut 10.
        cv (int, optional): Nombre de validations croisées. Par défaut 5.
        n_splits (int, optional): Nombre de splits pour Stratified K-Fold. Par défaut 5.
        compare_with (str, optional): La métrique utilisée pour la comparaison ('auc_pr', 'auc_roc', 'f1', 'balanced_acc'). Par défaut 'auc_pr'.

    Returns:
        None: Affiche les résultats et les courbes du meilleur modèle.
    """
    (X_train, X_test), (y_train, y_test), transformers = handle_features(dataframe, target_name)
    
    best_model = None
    best_score = float('-inf')

    for model_name, config in estimator_params.items():
        model = config['estimator']
        params = config['params']


        best_params = get_best_params(model, params, X_train, y_train, n_iter, cv)
        model = model(**best_params)

        auc_roc, auc_pr, f1, balanced_acc = run_stratified_kflod(model, X_train, y_train, n_splits=n_splits)

        print(f"Performance du meilleur modele: {model_name}")
        print(f"AUC ROC: {auc_roc:.2f}")
        print(f"AUC PR: {auc_pr:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Balanced Acc: {balanced_acc:.2f}")

        
        if compare_with == "f1":
            if f1 > best_score:
                best_score = f1
                best_model = model
        elif compare_with == "auc_roc":
            if auc_roc > best_score:
                best_score = auc_roc
                best_model = model
        elif compare_with == "auc_pr":
            if auc_pr > best_score:
                best_score = auc_pr
                best_model = model
        else:
            if balanced_acc > best_score:
                best_score = balanced_acc
                best_model = model

    if best_model is not None:
        print("---------------------------------- FINAL RESULTS ------------------------------------------")
        print(f"Meilleur modele trouvé après comparaison des approches: {best_model.__class__.__name__}")
        get_best_model_performance(best_model,X_test, y_test)