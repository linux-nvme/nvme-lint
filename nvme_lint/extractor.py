#!/usr/bin/env python3
"""
Copyright (c) 2022 Samsung Electronics Co., Ltd
SPDX-License-Identifier: GPLv2-or-later or Apache-2.0

Extract raw content from NVMe specification tables
"""

from . import utils
import camelot
import numpy as np
import pandas as pd


def extract_tables(file_path, page_height, page_number, content):
    """Extract the tables on given page"""
    tables_on_page = camelot.read_pdf(file_path, page_number, line_scale=35)
    tables = {}
    for caption, table in match_caption_to_table(tables_on_page, page_height, content):
        if caption != "skip":
            table = clean_table(table.df)

            # discard tables with less than two rows
            # a table with only headings is irrelevant
            if len(table.index) > 1:
                tables.update({caption: table})
    return page_number, tables


def match_caption_to_table(tables, page_height, content):
    for table in tables:
        table_y = table.cells[0][0].lt[1]/page_height
        caption_differences = [calc_difference(caption_y, table_y, content["height"])
                               for caption_y in content["top_coordinates"]]
        minimum_index = caption_differences.index(min(caption_differences))
        yield content["captions"][minimum_index], table


def calc_difference(caption_y, table_y, content_height):
    return abs((1 - caption_y / content_height) - table_y)


def clean_table(table):
    """Fix holes in nested tables, remove empty columns and align subtables."""
    # Replace empty strings with NaN, to make the clean-up easier
    table.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    # Drop empty rows and columns
    table.dropna(how="all", axis=0, inplace=True)
    table.reset_index(drop=True, inplace=True)

    table.dropna(how="all", axis=1, inplace=True)

    subtables = partition_into_subtables(0, table)

    while holes := find_holes(subtables, table):
        # Fix holes in the outermost cols first
        index = max(holes.keys())
        table = fix_hole(index, holes[index], table)

    # Replace the NaN with empty strings, to make the parsing and transformations easier
    table.replace(np.nan, "", inplace=True)

    # Reset column index to remove gaps
    table.columns = range(table.shape[1])
    return table


def find_holes(subtables, table):
    """Find holes between columns of the first row of each subtable.
    Return dictionary with kv pairs of {column: subtable}"""
    holes = {}
    for subtable in subtables:
        row = table.iloc[min(subtable)]
        if hole := find_hole_in_row(row):
            holes[hole] = subtable

    return holes


def find_hole_in_row(row):
    """Find hole in row.
    Return index of column with hole"""
    non_nan_indices = set()
    for i, value in enumerate(row):
        if not pd.isnull(value):
            non_nan_indices.add(i)
    if hole := set(range(min(non_nan_indices), max(non_nan_indices))) - non_nan_indices:
        # If there are multiple empty cols between the values we return the largest index
        return max(hole)

    return None


def fix_hole(col, rows_with_hole, table):
    """Fix hole in column 'col'."""
    # If there are no conflicts between col and col+1 for all rows
    if all(pd.isnull(row[col]) or pd.isnull(row[col+1])
           for row in table.itertuples(index=False, name=None)):

        # Merge entire col into col+1
        return pd.concat([table[table.columns[:col]],
                          table[table.columns[col]].combine_first(table[table.columns[col+1]]),
                          table[table.columns[col+2:]]], axis=1)

    # If there are no conflicts between col-1 and col for all rows
    elif all(pd.isnull(row[col-1]) or pd.isnull(row[col])
             for row in table.itertuples(index=False, name=None)):

        # Merge entire col into col-1
        return pd.concat([table[table.columns[:col-1]],
                          table[table.columns[col-1]].combine_first(table[table.columns[col]]),
                          table[table.columns[col+1:]]], axis=1)

    # If it is not possible to merge entire column
    else:
        # Swap col and col+1 only for rows_with_hole
        left = [row[col] for i, row in enumerate(table.itertuples(index=False, name=None)) if i in rows_with_hole]
        right = [row[col+1] for i, row in enumerate(table.itertuples(index=False, name=None)) if i in rows_with_hole]
        for i, row in enumerate(rows_with_hole):
            table.iat[row, col] = right[i]
            table.iat[row, col+1] = left[i]

        return table


def partition_into_subtables(current_first_row, table):
    """Partition tables recursively into subtables.
    Return a list of sets containing the indices of the rows they represent"""

    _, n_cols = table.shape
    subtables = []

    current_last_col = 0

    for col in reversed(range(n_cols)):
        if not pd.isnull(table.iat[current_first_row, col]):
            current_last_col = col
            break

    current_last_row = find_outer_end(current_first_row, table)

    current_subtable = set(range(current_first_row, current_last_row+1))

    while next_first_row := find_next_subtable(current_first_row, current_last_col, table):
        current_first_row = next_first_row
        next_subtables = partition_into_subtables(next_first_row, table)

        for subtable in next_subtables:
            current_subtable -= subtable
            subtables.append(subtable)
            current_first_row += len(subtable)

    if current_subtable:
        # Don't include empty subtables
        subtables.append(current_subtable)

    return subtables


def find_outer_end(first_row, table):
    """Return index of the final row of the outer shape of the subtable.
    This is done by finding the first row where there is content to the LEFT of the subtable"""
    n_rows, n_cols = table.shape

    subtable_first_col = 0
    for col in range(n_cols):
        if not pd.isnull(table.iat[first_row, col]):
            subtable_first_col = col
            break

    for i in range(first_row, n_rows):
        if any(not pd.isnull(table.iat[i, x]) for x in range(subtable_first_col)):
            return i-1

    # Subtable is in the final row of the table
    return n_rows-1


def find_next_subtable(first_row, last_col, table):
    """Return index of the first row of the next subtable.
    This is done by finding the first row where there is content to the RIGHT of the current subtable.
    If there is no such row, return None"""
    n_rows, n_cols = table.shape
    for i in range(first_row+1, n_rows):
        if any(not pd.isnull(table.iat[i, x]) for x in range(last_col+1, n_cols)):
            return i

    return None


def main(file_path, page_height, page_number, content):
    """Entry point"""
    global logger
    logger = utils.get_logger(f"Extractor.Page {page_number}")
    try:
        return extract_tables(file_path, page_height, page_number, content)
    except FileNotFoundError as e:
        logger.critical(e)
        return page_number, {}
