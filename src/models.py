from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

def train_binary_model(X_train, y_train, model_type='rf'):
    """训练二分类模型（支持 rf, xgb, lgb)"""
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    elif model_type == 'xgb':
        model = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))
    elif model_type == 'lgb':
        model = LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    else:
        raise ValueError("model_type must be 'rf', 'xgb' or 'lgb'")
    model.fit(X_train, y_train)
    return model

def train_multiclass_model(X_train, y_train, model_type='rf'):
    """训练多分类模型"""
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'xgb':
        model = XGBClassifier(n_estimators=100, random_state=42)
    elif model_type == 'lgb':
        model = LGBMClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("model_type must be 'rf', 'xgb' or 'lgb'")
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test, labels=None):
    """评估模型，打印报告并返回准确率"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=labels))
    return acc, y_pred, confusion_matrix(y_test, y_pred)

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)