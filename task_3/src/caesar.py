RUS_ALPHA = "абвгдежзийклмнопрстуфхцчшщъыьэюя"  # без "ё"
ENG_ALPHA = "abcdefghijklmnopqrstuvwxyz"


def caesar_shift(text: str, shift: int, alphabet: str) -> str:
    """
    Дешифрует строку, зашифрованную шифром Цезаря (циклическим сдвигом) по заданному алфавиту.

    Идея:
      - Каждая буква из `alphabet` заменяется на букву, сдвинутую на `shift` позиций назад по циклу.
      - Сдвиг выполняется по модулю длины алфавита.
      - Регистр букв сохраняется (верхний/нижний).
      - Символы, которых нет в `alphabet` (цифры, пробелы, точки, '@', дефисы и т.п.),
        остаются без изменений.

    Параметры
    ----------
    text : str
        Входной текст (зашифрованная строка). Любой тип будет приведён к строке.
    shift : int
        Величина сдвига (ключ). Для дешифрования используется правило:
        new_index = (old_index - shift) mod len(alphabet).
    alphabet : str
        Алфавит, по которому выполняется сдвиг

    Возвращает
    ----------
    str
        Дешифрованная строка.
    """
    text = str(text)
    n = len(alphabet)
    idx = {ch: i for i, ch in enumerate(alphabet)}
    out = []
    for ch in text:
        lo = ch.lower()
        if lo in idx:
            i = idx[lo]
            new = alphabet[(i - shift) % n]
            out.append(new.upper() if ch.isupper() else new)
        else:
            out.append(ch)
    return "".join(out)


def dec_addr(addr_enc: str, k: int) -> str:
    """
    Дешифрует обезличенный адрес, предполагая шифр Цезаря по русскому алфавиту.

    Параметры
    ----------
    addr_enc : str
        Зашифрованный (обезличенный) адрес.
    k : int
        Ключ (сдвиг) для текущей строки.

    Возвращает
    ----------
    str
        Дешифрованный адрес.
    """
    return caesar_shift(addr_enc, k % len(RUS_ALPHA), RUS_ALPHA)


def dec_email(email_enc: str, k: int) -> str:
    """
    Дешифрует обезличенный email, предполагая шифр Цезаря по латинскому алфавиту.

    Параметры
    ----------
    email_enc : str
        Зашифрованный (обезличенный) email.
    k : int
        Ключ (сдвиг) для текущей строки.

    Возвращает
    ----------
    str
        Дешифрованный email.
    """
    return caesar_shift(email_enc, k % 26, ENG_ALPHA)
