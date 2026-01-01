import os
import pandas as pd

from typing import List

from src.caesar import dec_addr, dec_email
from src.scoring import best_k_by_addr, best_k_joint

INPUT_PATH = "../data/task-3/Задание-3-данные.xlsx"


def identify(hashes: List[str]) -> None:
    """
    Brute-force атака для деанонимизации телефонных номеров.

    :param hashes: исходные хеши телефонных номеров
    :type hashes: List[str] 
    """
    with open('hashes.txt', 'w') as f:
        for hash in hashes:
            f.write(hash + "\n")
    os.system(
        "hashcat.exe -a 3 -m 100 -o output.txt hashes.txt ?d?d?d?d?d?d?d?d?d?d?d")


def deanon_data(input_path: str = INPUT_PATH) -> pd.DataFrame:
    """
    Основная функция для выполнения

    :param input_path: путь для входного файла
    :type input_path: str
    :return: датафрейм с деобезличенными данными и ключами шифрования
    :rtype: DataFrame
    """
    raw = pd.read_excel(input_path)

    raw.columns = ["_c0", "Телефон", "email", "Адрес"]
    raw = raw.iloc[1:].reset_index(drop=True)
    df = raw[["Телефон", "email", "Адрес"]].copy()

    addr_keys = []
    addr_scores = []
    for a in df["Адрес"].astype(str):
        k, sc = best_k_by_addr(a)
        addr_keys.append(k)
        addr_scores.append(sc)

    df_addr = df.copy()
    df_addr["k_addr"] = addr_keys
    df_addr["addr_score"] = addr_scores

    keys = []
    sum_scores = []
    addr_part = []
    email_part = []

    for e, a in zip(df["email"].astype(str), df["Адрес"].astype(str)):
        k, ssum, sa, se = best_k_joint(e, a)
        keys.append(k)
        sum_scores.append(ssum)
        addr_part.append(sa)
        email_part.append(se)

    out = df.copy()
    out["Ключ_шифрования"] = keys
    out["email_деобезличен"] = [
        dec_email(e, k) for e, k in zip(df["email"].astype(str), keys)]
    out["Адрес_деобезличен"] = [
        dec_addr(a, k) for a, k in zip(df["Адрес"].astype(str), keys)]
    out["score_addr"] = addr_part
    out["score_email"] = email_part
    out["score_total"] = sum_scores

    phones = df["Телефон"].astype(str).str.strip()

    identify(phones.tolist())

    with open("output.txt", "r") as f:
        lines = f.readlines()
    hash_to_phone = {}
    for line in lines:
        parts = line.strip().split(":")
        if len(parts) == 2:
            hash_val, phone = parts
            hash_to_phone[hash_val] = phone

    out['Телефон_деобезличен'] = out['Телефон'].astype(str).map(hash_to_phone)
    return out


if __name__ == "__main__":
    out = deanon_data()
    out[['Телефон_деобезличен', 'email_деобезличен',
         'Адрес_деобезличен', 'Ключ_шифрования']].to_excel(
        'Деобезличенные_данные+ключ_шифрования.xlsx', index=False)
