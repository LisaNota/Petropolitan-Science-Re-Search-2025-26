#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Задание 1. Подсчёт количества уникальных IPv6-адресов в большом файле.

Ограничения и требования:
- Входной файл может содержать до 10^9 строк; каждая строка — валидный IPv6 (полная/сокращённая форма, произвольный регистр, возможен '::').
- Нужно посчитать количество уникальных IPv6-адресов.
- Память ограничена (~1 ГБ), допускаются временные файлы.
- Использовать только стандартную библиотеку Python.

Идея решения: внешняя сортировка (external merge sort) по каноническому бинарному виду IPv6 (16 байт)
и подсчёт уникальных при финальном k-way merge.

Запуск:
    python main.py <input.txt> <output.txt>
"""

from __future__ import annotations

import argparse
import heapq
import os
import shutil
import socket
import tempfile
from typing import BinaryIO, List, Optional, Tuple

RECORD_SIZE = 16  # IPv6 занимает ровно 16 байт (128 бит) в packed-форме


def ipv6_to_packed(addr: str) -> bytes:
    """
    Преобразовать IPv6-адрес из строки (валидный, в любом допустимом виде записи)
    в каноническое бинарное представление длиной 16 байт (packed).

    Это представление:
    - однозначно (не зависит от регистра, ведущих нулей, использования '::');
    - компактно и удобно для сортировки/сравнения.

    Параметры:
        addr: строка с IPv6-адресом.

    Возвращает:
        bytes длины 16.

    Исключения:
        OSError: если строка не является валидным IPv6 (в задаче гарантируется валидность).
    """
    return socket.inet_pton(socket.AF_INET6, addr)


def flush_run(buf: List[bytes], tmp_dir: str, run_idx: int) -> str:
    """
    Отсортировать буфер packed-IPv6 (по 16 байт) и записать его в бинарный run-файл.

    Run-файл представляет собой последовательность записей фиксированной длины 16 байт:
        [rec0][rec1][rec2]...

    Параметры:
        buf: список packed-адресов (bytes по 16 байт).
        tmp_dir: директория для временных файлов.
        run_idx: индекс (для уникального имени файла).

    Возвращает:
        Путь к созданному run-файлу.
    """
    buf.sort()
    path = os.path.join(tmp_dir, f"run_{run_idx:06d}.bin")
    with open(path, "wb", buffering=1024 * 1024) as f:
        for rec in buf:
            f.write(rec)
    return path


def generate_initial_runs(input_path: str, tmp_dir: str, chunk_records: int) -> List[str]:
    """
    Прочитать входной текстовый файл и сформировать начальные отсортированные runs на диске.

    Алгоритм:
    - потоково читаем строки;
    - переводим IPv6 в packed-вид (16 байт);
    - накапливаем chunk_records записей в памяти;
    - сортируем и записываем run-файл.

    Параметры:
        input_path: путь к входному текстовому файлу.
        tmp_dir: директория для временных файлов.
        chunk_records: сколько IPv6 хранить в памяти перед сбросом на диск.

    Возвращает:
        Список путей к run-файлам (каждый run уже отсортирован).
    """
    runs: List[str] = []
    buf: List[bytes] = []
    run_idx = 0

    with open(input_path, "r", encoding="ascii", errors="strict", newline="") as fin:
        for line in fin:
            s = line.strip()
            if not s:
                # По условию пустых строк нет, но оставим защиту.
                continue

            buf.append(ipv6_to_packed(s))

            if len(buf) >= chunk_records:
                runs.append(flush_run(buf, tmp_dir, run_idx))
                run_idx += 1
                buf.clear()

    if buf:
        runs.append(flush_run(buf, tmp_dir, run_idx))
        buf.clear()

    return runs


def _open_runs(run_paths: List[str]) -> List[BinaryIO]:
    """Открыть run-файлы для бинарного чтения с увеличенным буфером."""
    files: List[BinaryIO] = []
    for p in run_paths:
        files.append(open(p, "rb", buffering=1024 * 1024))
    return files


def merge_runs_to_file(run_paths: List[str], out_path: str) -> None:
    """
    Слить несколько отсортированных run-файлов в один отсортированный run-файл (k-way merge).

    Параметры:
        run_paths: пути к входным отсортированным runs.
        out_path: путь к выходному run-файлу.
    """
    files = _open_runs(run_paths)
    try:
        heap: List[Tuple[bytes, int]] = []
        for i, f in enumerate(files):
            rec = f.read(RECORD_SIZE)
            if rec:
                heap.append((rec, i))
        heapq.heapify(heap)

        with open(out_path, "wb", buffering=1024 * 1024) as out:
            while heap:
                rec, i = heapq.heappop(heap)
                out.write(rec)
                nxt = files[i].read(RECORD_SIZE)
                if nxt:
                    heapq.heappush(heap, (nxt, i))
    finally:
        for f in files:
            try:
                f.close()
            except Exception:
                pass


def reduce_runs(run_paths: List[str], tmp_dir: str, fan_in: int) -> List[str]:
    """
    Уменьшить количество run-файлов многоступенчатым слиянием партиями.

    Это нужно, потому что нельзя открыть слишком много файлов одновременно
    (лимит файловых дескрипторов ОС).

    Параметры:
        run_paths: список текущих runs.
        tmp_dir: директория для временных файлов.
        fan_in: максимальное число runs, сливаемых за один проход.

    Возвращает:
        Новый список runs, размер которого <= fan_in.
    """
    level = 0
    runs = run_paths[:]

    while len(runs) > fan_in:
        new_runs: List[str] = []

        for batch_start in range(0, len(runs), fan_in):
            batch = runs[batch_start: batch_start + fan_in]
            merged_path = os.path.join(
                tmp_dir, f"merge_{level:03d}_{batch_start // fan_in:06d}.bin")

            merge_runs_to_file(batch, merged_path)
            new_runs.append(merged_path)

            # Удаляем старые файлы партии, чтобы освобождать место на диске.
            for p in batch:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

        runs = new_runs
        level += 1

    return runs


def count_unique_across_runs(run_paths: List[str]) -> int:
    """
    Подсчитать количество уникальных IPv6-адресов по нескольким отсортированным runs,
    выполняя k-way merge и сравнивая текущую запись с предыдущей.

    Параметры:
        run_paths: список путей к отсортированным run-файлам.

    Возвращает:
        Количество уникальных адресов.
    """
    if not run_paths:
        return 0

    files = _open_runs(run_paths)
    try:
        heap: List[Tuple[bytes, int]] = []
        for i, f in enumerate(files):
            rec = f.read(RECORD_SIZE)
            if rec:
                heap.append((rec, i))
        heapq.heapify(heap)

        prev: Optional[bytes] = None
        uniq = 0

        while heap:
            rec, i = heapq.heappop(heap)
            if prev != rec:
                uniq += 1
                prev = rec

            nxt = files[i].read(RECORD_SIZE)
            if nxt:
                heapq.heappush(heap, (nxt, i))

        return uniq
    finally:
        for f in files:
            try:
                f.close()
            except Exception:
                pass


def count_unique_ipv6_external(
    input_path: str,
    output_path: str,
    chunk_records: int = 1_000_000,
    fan_in: int = 128,
    keep_tmp: bool = False,
) -> int:
    """
    Основная функция решения: внешняя сортировка + подсчёт уникальных.

    Параметры:
        input_path: путь к входному текстовому файлу (IPv6 по одному в строке).
        output_path: путь к выходному файлу (одно целое число).
        chunk_records: размер чанка (сколько адресов держим в RAM перед сортировкой и сбросом в run).
        fan_in: максимум файлов, которые сливаем/открываем одновременно.
        keep_tmp: сохранять ли временную директорию (для отладки).

    Возвращает:
        Количество уникальных IPv6-адресов.
    """
    tmp_dir = tempfile.mkdtemp(prefix="ipv6_uniq_")
    try:
        runs = generate_initial_runs(input_path, tmp_dir, chunk_records)
        runs = reduce_runs(runs, tmp_dir, fan_in)
        ans = count_unique_across_runs(runs)

        with open(output_path, "w", encoding="utf-8", newline="\n") as fout:
            fout.write(str(ans))
            fout.write("\n")

        return ans
    finally:
        if keep_tmp:
            print(f"[INFO] Временная директория сохранена: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    """
    Разбирает аргументы командной строки.

    По требованию — два позиционных аргумента:
        1) входной файл
        2) выходной файл
    """
    p = argparse.ArgumentParser(
        description="Подсчёт уникальных IPv6-адресов в большом файле"
    )
    p.add_argument(
        "input_path", help="Путь к входному текстовому файлу (IPv6 по одному в строке).")
    p.add_argument(
        "output_path", help="Путь к выходному файлу (одно целое число).")
    return p.parse_args()


def main() -> None:
    """Точка входа"""
    args = parse_args()
    count_unique_ipv6_external(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
