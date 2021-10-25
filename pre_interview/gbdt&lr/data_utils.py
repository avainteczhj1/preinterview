#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pathlib
import sys

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))


def read_samples(filename):
    """Read the data file and return a sample list.

    Args:
        filename (str): The path of the txt file.

    Returns:
        list: A list conatining all the samples in the file.
    """
    samples,labels = [],[]
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            sample, label = line.strip().split('||||')
            samples.append(sample)
            labels.append(label)
    return samples,labels


def write_samples(samples, labels, file_path, opt='w'):
    """Write the samples into a file.

    Args:
        samples (list): The list of samples to write.
        file_path (str): The path of file to write.
        opt (str, optional): The "mode" parameter in open(). Defaults to 'w'.
    """   
    with open(file_path, opt, encoding='utf8') as file:
        for n, line in enumerate(samples):
            file.write(line + "||||" + str(labels[n]))
            file.write('\n')

def isChinese(word):
    """Distinguish Chinese words from non-Chinese ones.

    Args:
        word (str): The word to be distinguished.

    Returns:
        bool: Whether the word is a Chinese word.
    """
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False
