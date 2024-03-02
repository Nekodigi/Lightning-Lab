import os


def listdir_by_date(path):
    return sorted(os.listdir(path), key=lambda x: os.path.getctime(f"{path}/{x}"))
