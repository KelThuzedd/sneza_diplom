import nni
import pandas as pd
import xgboost as xgb
import logging
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

# Загружаем данные из формата parquet
X_train = pd.read_parquet('../data/X_train.parquet')
X_test = pd.read_parquet('../data/X_test.parquet')
y_train = pd.read_parquet('../data/y_train.parquet')
y_test = pd.read_parquet('../data/y_test.parquet')


LOG = logging.getLogger('xgboost_regression')
LOG.setLevel(logging.DEBUG)

# Создайте обработчик для записи логов в файл nni_log.txt
file_handler = logging.FileHandler('nni_log.txt')
file_handler.setLevel(logging.DEBUG)

# Определите формат записи логов
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Добавьте обработчик к логгеру
LOG.addHandler(file_handler)
file_handler.flush()


def load_data():
    return X_train, X_test, y_train, y_test


def get_default_parameters():
    '''Получите параметры по умолчанию'''
    params = {
        'max_depth': 3,
        'n_estimators': 100,  # Пример значения по умолчанию
        'learning_rate': 0.1,  # Пример значения по умолчанию
        'subsample': 1,  # Пример значения по умолчанию
        'gamma': 0,  # Пример значения по умолчанию
        'min_child_weight': 1,  # Пример значения по умолчанию
        'max_delta_step': 0,  # Пример значения по умолчанию
        'colsample_bytree': 1,  # Пример значения по умолчанию
        'reg_alpha': 0,  # Пример значения по умолчанию
        'reg_lambda': 1,  # Пример значения по умолчанию
        'scale_pos_weight': 1  # Пример значения по умолчанию
    }
    return params



def get_model(PARAMS):
    '''Получите модель XGBoost с параметрами из NNI'''
    model = xgb.XGBClassifier(
        max_depth=PARAMS['max_depth'],
        n_estimators=PARAMS['n_estimators'],
        learning_rate=PARAMS['learning_rate'],
        subsample=PARAMS['subsample'],
        gamma=PARAMS['gamma'],  # Добавленный гиперпараметр
        min_child_weight=PARAMS['min_child_weight'],  # Добавленный гиперпараметр
        max_delta_step=PARAMS['max_delta_step'],  # Добавленный гиперпараметр
        colsample_bytree=PARAMS['colsample_bytree'],  # Добавленный гиперпараметр
        reg_alpha=PARAMS['reg_alpha'],  # Добавленный гиперпараметр
        reg_lambda=PARAMS['reg_lambda'],  # Добавленный гиперпараметр
        scale_pos_weight=PARAMS['scale_pos_weight'],  # Добавленный гиперпараметр
        # objective='reg:squarederror',
        tree_method='gpu_hist'
    )

    return model


def run(X_train, X_test, y_train, y_test, model):
    '''Обучите модель и выполните предсказание'''
    try:
        model.fit(X_train, y_train)
        predict_y = model.predict(X_test)
        score = accuracy_score(y_test, predict_y)
        LOG.debug('accuracy score: %s', score)
        LOG.debug('Функция run')
        nni.report_final_result(score)
    except Exception as e:
        # Запишите исключение в лог
        LOG.exception(e)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()

    try:
        # Получите параметры из tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS)
        LOG.debug('Параметры получены')
        run(X_train, X_test, y_train, y_test, model)
        LOG.debug('Модель запущена')
    except Exception as exception:
        LOG.exception(exception)
        LOG.debug('Ошибка')
        raise
