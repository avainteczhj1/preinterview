#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd


def clean_context(df, column_list):
    end_column_name = None
    for column_index, i in enumerate(column_list):
        if "</Root>" in str(df[i]):
            end_column_name = i
            end_column_index = column_index
            break

    if end_column_name:
        if end_column_name == "RECORD_CONTENT":
            return df["RECORD_CONTENT"]
        else:
            df["NEW_RECORD_CONTENT"] = ""
            for i in column_list[5:end_column_index]:
                df["NEW_RECORD_CONTENT"] += str(df[i])

            return df["NEW_RECORD_CONTENT"]
    else:
        return None


def clean_other_column(df, column_list, bias):
    end_column_name = None
    for column_index, i in enumerate(column_list):
        if "</Root>" in str(df[i]):
            end_column_name = i
            end_column_index = column_index
            break

    if end_column_name:
        return df[column_list[end_column_index + bias]]
    else:
        return None


if __name__ == '__main__':
    case_filepath = ""
    save_filepath = ""
    save_feature = []
    case_file_df = pd.read_csv(case_filepath, encoding='utf-8', engine='python', error_bad_lines=False)
    column_list = case_file_df.columns.tolist()

    case_file_df["NEW_RECORD_CONTENT"] = case_file_df.apply(clean_context, args=(column_list,), axis=1)
    case_file_df["NEW_patientId"] = case_file_df.apply(clean_other_column, args=(column_list, 1), axis=1)
    case_file_df["NEW_Inhosp_Num"] = case_file_df.apply(clean_other_column, args=(column_list, 2), axis=1)
    case_file_df["NEW_WSXH"] = case_file_df.apply(clean_other_column, args=(column_list, 3), axis=1)

    case_file_df[save_feature].to_csv(save_filepath)
