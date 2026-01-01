import re
from src.caesar import dec_addr, dec_email, RUS_ALPHA


ADDR_TOKENS = ["ул.", "пер.", "пр.", "пл.",
               "наб.", "кв.", "д.", "дом", "корп", "стр"]
EMAIL_RE = re.compile(r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$")
EMAIL_HINTS = [
    "gmail.com", "hotmail.com", "yandex.ru", "mail.ru", "outlook.com", "icloud.com",
    ".com", ".ru", ".net", ".org", ".biz", ".info"
]


def score_addr(addr_plain: str) -> int:
    """
    Оценивает правдоподобие расшифрованного адреса по наличию типичных «якорей» адресного формата.

    Используется как эвристика для выбора ключа (сдвига) в шифре Цезаря:
    чем больше в строке характерных сокращений/токенов адреса, тем выше score.

    Логика скоринга:
      - За каждый токен из `ADDR_TOKENS` (например, "ул.", "д.", "кв.") добавляем +3.
      - Если одновременно встречаются "кв." и ("д." или "дом") → +2,
        т.к. связка «дом + квартира» сильно указывает на корректный адрес.

    Параметры
    ----------
    addr_plain : str
        Адрес в предполагаемом «расшифрованном» виде (после применения кандидата-ключа).

    Возвращает
    ----------
    int
        Целочисленный скор. Чем больше, тем более вероятно, что адрес расшифрован правильно.
    """
    s = str(addr_plain).lower()
    score = 0
    for tok in ADDR_TOKENS:
        if tok in s:
            score += 3
    if re.search(r"\bул\.\s", s):
        score += 2
    if re.search(r"\bпер\.\s", s):
        score += 2
    if re.search(r"\bпр\.\s", s):
        score += 2
    if "кв." in s and ("д." in s or "дом" in s):
        score += 2
    return score


def best_k_by_addr(addr_enc: str) -> tuple[int, int]:
    """
    Подбирает ключ (сдвиг) шифра Цезаря для адреса перебором всех возможных значений.

    Перебираем k от 0 до len(RUS_ALPHA)-1:
      1) дешифруем `addr_enc` функцией `dec_addr(addr_enc, k)`,
      2) считаем score через `score_addr`,
      3) выбираем k, который даёт максимальный score.

    Параметры
    ----------
    addr_enc : str
        Зашифрованный адрес (обезличенное значение).

    Возвращает
    ----------
    tuple[int, int]
        (best_k, best_score), где:
          - best_k — найденный ключ (сдвиг),
          - best_score — скор для этого ключа.
    """
    best_k, best_sc = 0, -10**9
    for k in range(len(RUS_ALPHA)):
        sc = score_addr(dec_addr(addr_enc, k))
        if sc > best_sc:
            best_sc = sc
            best_k = k
    return best_k, best_sc


def score_email(email_plain: str) -> int:
    """
    Оценивает правдоподобие расшифрованного email по формату и доменным «подсказкам».

    Используется как эвристика для выбора ключа (сдвига) в шифре Цезаря для email:
    правильная расшифровка должна соответствовать базовому формату email и часто
    содержать типичные домены/суффиксы.

    Логика скоринга:
      - Если строка матчится на `EMAIL_RE` (валидный формат email) → +4.
      - За каждую подстроку из `EMAIL_HINTS` (gmail.com, .com, .ru и т.п.) → +2.

    Параметры
    ----------
    email_plain : str
        Email в предполагаемом «расшифрованном» виде (после применения кандидата-ключа).

    Возвращает
    ----------
    int
        Целочисленный скор. Чем больше, тем более вероятно, что email расшифрован правильно.
    """
    s = str(email_plain).lower()
    score = 0
    if EMAIL_RE.match(s):
        score += 4
    for h in EMAIL_HINTS:
        if h in s:
            score += 2
    if "@" in s and "." in s.split("@")[0]:
        score += 1
    return score


def best_k_joint(email_enc: str, addr_enc: str) -> tuple[int, int, int, int]:
    """
    Совместно подбирает ключ (сдвиг) для пары полей (email, адрес) в одной строке датасета.

    Метод:
      Для каждого k от 0 до len(RUS_ALPHA)-1:
        1) a_plain = dec_addr(addr_enc, k)
        2) e_plain = dec_email(email_enc, k)
        3) sa = score_addr(a_plain)
        4) se = score_email(e_plain)
        5) ssum = sa + se
      Выбираем k, максимизирующий ssum.

    Параметры
    ----------
    email_enc : str
        Зашифрованный email (обезличенное значение).
    addr_enc : str
        Зашифрованный адрес (обезличенное значение).

    Возвращает
    ----------
    tuple[int, int, int, int]
        (best_k, best_sum_score, best_addr_score, best_email_score), где:
          - best_k — выбранный ключ,
          - best_sum_score — максимальная сумма score_addr + score_email,
          - best_addr_score — вклад адреса при best_k,
          - best_email_score — вклад email при best_k.
    """
    best_k, best_sum = 0, -10**9
    best_a, best_e = 0, 0
    for k in range(len(RUS_ALPHA)):
        a_plain = dec_addr(addr_enc, k)
        e_plain = dec_email(email_enc, k)
        sa = score_addr(a_plain)
        se = score_email(e_plain)
        ssum = sa + se
        if ssum > best_sum:
            best_sum = ssum
            best_k = k
            best_a = sa
            best_e = se
    return best_k, best_sum, best_a, best_e
