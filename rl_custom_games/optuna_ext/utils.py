import os


def get_optuna_storage():
    OPTUNA_MYSQL_DATABASE = os.getenv("OPTUNA_MYSQL_DATABASE")

    if OPTUNA_MYSQL_DATABASE is None:
        return None
    OPTUNA_MYSQL_USER = os.getenv("OPTUNA_MYSQL_USER")
    OPTUNA_MYSQL_PASSWORD = os.getenv("OPTUNA_MYSQL_PASSWORD")
    OPTUNA_MYSQL_HOST = os.getenv("OPTUNA_MYSQL_HOST")
    OPTUNA_MYSQL_PORT = os.getenv("OPTUNA_MYSQL_PORT")


    strstorage= f"mysql+pymysql://{OPTUNA_MYSQL_USER}:{OPTUNA_MYSQL_PASSWORD}@{OPTUNA_MYSQL_HOST}:{OPTUNA_MYSQL_PORT}/{OPTUNA_MYSQL_DATABASE}"
    # print(strstorage)

    return strstorage