#!/usr/bin/env python3
"""
Copyright (c) 2022 Samsung Electronics Co., Ltd
SPDX-License-Identifier: GPLv2-or-later or Apache-2.0
"""
import pytest
import numpy as np
import pandas as pd

from nvme_lint import extractor
from nvme_lint import utils


@pytest.mark.parametrize("before, after", [("misaligned_1", "aligned"),
                                           ("misaligned_2", "aligned"),
                                           ("aligned", "aligned"),
                                           ("misaligned_with_missing_values", "aligned_with_missing_values"),
                                           ("aligned_with_missing_values", "aligned_with_missing_values"),
                                           ("misaligned_3_col", "aligned_3_col"),
                                           ("aligned_3_col", "aligned_3_col"),
                                           ("aligned_mixed_col", "aligned_mixed_col"),
                                           ("misaligned_mixed_col_1", "aligned_mixed_col"),
                                           ("misaligned_mixed_col_2", "aligned_mixed_col"),
                                           ("misaligned_mixed_col_3", "aligned_mixed_col"),
                                           ("misaligned_3_col_with_missing_values", "aligned_3_col_with_missing_values"),
                                           ("aligned_3_col_with_missing_values", "aligned_3_col_with_missing_values"),
                                           ("misaligned_3_col_with_missing_columns", "aligned_3_col_with_missing_columns"),
                                           ("aligned_3_col_with_missing_columns", "aligned_3_col_with_missing_columns"),
                                           ("misaligned_double_nested", "aligned_double_nested"),
                                           ("aligned_double_nested", "aligned_double_nested"),
                                           ("aligned_no_nesting", "aligned_no_nesting")])
def test_clean_table(before, after):
    prefix = "tests/data/"
    postfix = ".csv"
    before = utils.expand_path(prefix+before+postfix)
    after = utils.expand_path(prefix+after+postfix)
    try:
        before_df = pd.read_csv(before)
        after_df = pd.read_csv(after)

    except FileNotFoundError:
        pytest.fail("tests must be run from 'nvme-lint/'")

    # Replace NaN with the empty string to mimic the actual tables
    before_df.replace(np.nan, "", inplace=True)
    after_df.replace(np.nan, "", inplace=True)

    clean = extractor.clean_table(before_df)

    assert compare_dataframes(clean, after_df)


def compare_dataframes(left, right):
    """This function is a replacement of '.equals()',
    this is necessary because '.equals()' is too strict with the typing for these tests.
    This function attempts a comparison of the values as floats,
    if the conversion fails then the comparison is made as strings"""
    if left.shape != right.shape:
        return False

    rows, cols = left.shape
    for i in range(rows):
        for j in range(cols):
            try:
                if float(left.iloc[i-1, j-1]) != float(right.iloc[i-1, j-1]):
                    return False
            except ValueError:
                if str(left.iloc[i-1, j-1]) != str(right.iloc[i-1, j-1]):
                    return False
    return True
