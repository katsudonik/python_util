#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import pytablewriter

def end_of_month_date(date):
    return (date + relativedelta(months=1)).replace(day=1) - dt.timedelta(days=1)

def daterange(start_date, end_date):
    def _daterange(_start, _end):
      for n in range((_end - _start).days):
        yield _start + dt.timedelta(n)

    dates = []
    for i in _daterange(start_date, end_date + dt.timedelta(days=1)):
        dates.append(i)
    return np.array(dates)

def pickle_load(path):
     return pickle.load(open(path, 'rb'))

def mkdir(_dir):
    os.makedirs(_dir, exist_ok=True)
    return _dir

def ds_del_decimal(float_vals):
    return np.modf(float_vals)[1]

def ds_replcate_inf(ds, v):
    return ds.replace([np.inf, -np.inf], v)

# avoid duplicate column by merge
# save left index
# check same data size between before/after merge
def df_merge(left, right, on, how, merge_columns=[]):
    if 'index' in list(right.columns):
        right = right.drop('index', axis=1)

    left['index'] = left.index

    right_columns = right.columns
    if len(merge_columns) > 0:
        right_columns = on + merge_columns

    before_len = len(left)
    left = pd.merge(left=left[list(set(left.columns) - set(merge_columns))], right=right[right_columns], on=on, how='left')
    if len(left) != before_len:
        raise ValueError(f'merged data is different length (on:{on}). before:{before_len}, after:{len(left)}')
    left = left.set_index('index')        
    return left

def df_rename(df, columns):
    for olg_k, upd_k in columns.items():
        if upd_k in df.columns:
            print(f'key({upd_k}) is exist. so, override it')
        df[upd_k] = df[olg_k]
        df = df.drop(olg_k, axis=1)
    return df

# ascending: dict
# columns: list
# int_columns: list
# TODO test
def df_sort(ds, columns, int_columns, ascending={}):
    if len(int_columns) == 0:
        if len(ascending) == 0:
            return ds.sort_values(columns)
        return ds.sort_values(columns, ascending=ascending)
    
    if len(ascending) == 0:
        return ds.astype(dict(zip(int_columns, [int] * len(int_columns)))).sort_values(columns)
    return ds.astype(dict(zip(int_columns, [int] * len(int_columns)))).sort_values(columns, ascending=ascending)

def df_add_group_column(ds, group_columns, sep='|'):
    tmp = ds[group_columns[0]]
    for column in group_columns[1:]:
        tmp = tmp.str.cat([ds[column]], sep=sep)
    group_key = sep.join(group_columns)
    ds[group_key] = tmp
    return ds, group_key

# TODO test
def df_group_count(ds, group_columns, cnt_key='num'):
    ds[cnt_key] = 1
    return ds.groupby(groupby)[cnt_key].sum().reset_index()

def df_check(ds):
    if len(ds) == 0:
        raise ValueError('data is none')
    i = 0
    _column = []
    _type = []
    _val = []
    for val in ds.iloc[0]:
        _column.append(ds.columns[i])
        _type.append(type(val))
        _val.append(val)
        i += 1
    display(pd.DataFrame({'column': _column, 'type': _type, 'val': _val}))

def df_check_duplicate(ds, categories_list, pks, print_duplicate_pks=True, occur_error=True):
    error_paths = []
    if len(ds) != len(ds[categories_list].drop_duplicates()):
        tmp = ds.copy()
        tmp['num'] = 1
        counts = tmp.groupby(categories_list)['num'].sum().reset_index().sort_values(pks)
        error_ds = counts[counts['num'] > 1]
        error_ds_output_path = './duplicates_all.csv'
        error_ds.to_csv(error_ds_output_path, encoding='utf8', index=False)    
        print(f'error ds out path:{error_ds_output_path}')
        error_paths.append(error_ds_output_path)
        raise ValueError('all categories is duplicate!')

    pk_num = len(ds[pks].drop_duplicates()) 
    if pk_num != len(ds):             
        error_columns = []            
        for column in categories_list:
            error_ds_output_path = ''
            if column in pks:
                continue
            grouped_ds = ds[pks + [column]].drop_duplicates()
            grouped_ds['num'] = 1
            size = len(grouped_ds)
            if size > pk_num:
                error_columns.append(column)
                if print_duplicate_pks == True:
                    counts = grouped_ds.groupby(pks)['num'].sum().reset_index().sort_values(pks)
                    error_ds = counts[counts['num'] > 1]
                    error_ds_output_path = './duplicates_' + column + '.csv'
                    error_ds.to_csv(error_ds_output_path, encoding='utf8', index=False)    
                    print(f'error ds out path:{error_ds_output_path}')
                    error_paths.append(error_ds_output_path)
        if occur_error == True:            
            raise ValueError(f'unique key is not unique!. unique num:{pk_num}, data num:{len(ds)}, duplicated_columns:{error_columns}')
    return error_paths
    
def copy_series_to_ds(series, num):
    tmp = []
    for i in range(num):
        tmp.append(series)
    return pd.DataFrame(tmp)        

def df_copy_row(row, num):
    _list = {}
    for column in row.columns:
        _list[column] = list(np.full(num, row[column]))
    return pd.DataFrame(_list, columns=row.columns)

# return group key
def df_group_by_interquartile(ds, column):
    _ds = ds.copy()
    desc = _ds[column].describe()
    return = pd.cut(_ds[column], [-1, desc['25%'], desc['50%'], desc['75%'], desc['max']], labels = False, duplicates='drop')

# return train_ds, test_ds
def df_train_test_split_on_each_interquartile_range(ds, column, frac, range_column='range'):
    _ds = ds.copy()
    if len(list(set(_ds.index))) != len(_ds):
        raise ValueError('reset index')
    
    _ds[range_column] = df_group_by_interquartile(_ds, column)
    
    _frac = frac
    if frac > 0.5:
        _frac = 1 - frac
    
    sample_ds = pd.concat(list(map(
        lambda group:
        _ds[_ds[range_column] == group].sample(frac=_frac),
        list(_ds[range_column].drop_duplicates())
    )))
    other_ds = _ds[~_ds.index.isin(sample_ds.index)]
    
    if frac > 0.5:
        return other_ds, sample_ds # train_ds, test_ds
    return sample_ds, other_ds # train_ds, test_ds

def df_to_plot(ds, x_column, y_column, title, save_path):
    ds = ds.sort_values(x_column)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(title)
    plt.plot(ds[x_column], ds[y_column])
    plt.savefig(save_path)
    plt.show()

def df_to_md(ds, path):
    writer = pytablewriter.MarkdownTableWriter()
    writer.from_dataframe(ds)
    txt = writer.dumps()
    with open(path, mode='w', encoding='utf-8') as f:
        f.write(txt)

def df_count(ds, groupby, count_label='count'):
    _ds = ds.copy()
    _ds[count_label] = 1
    return _ds.groupby([groupby])[count_label].sum().reset_index()

# calc

def calc_rmse(x, y):
    return np.sqrt(mean_squared_error(x, y))

def calc_rmse_on_each_group(ds, groupby, x_key, y_key):
    ds, group_key = df_add_group_column(ds, groupby)
    group = target[[group_key] + groupby].drop_duplicates().sort_values(groupby)

    group['rmse'] = group[group_key].map(
        lambda key:
        calc_rmse(
            ds[ds[group_key] == key][x_key], 
            ds[ds[group_key] == key][y_key] 
        ))
    )
    return group

def calc_corr(x, y):
    return np.corrcoef(x, y)[0, 1]

def fig_corr(x, y, path):
    corr = calc_corr(x, y)
    
    plt.figure(figsize=(7,7))
    plt.scatter(x, y, marker='*', c='green', alpha=0.5, s=1)
    plt.xlabel('actual')
    plt.ylabel('prediction')
    plt.title(f'prediction vs actual (RMSE:{np.sqrt(mean_squared_error(x,y))})')
    plt.figtext(
        0.25, 0.85,
        'Cor: {:5.4f}'.format(corr),
        ha='right',
    )
    plt.figtext(
        0.85, 0.15,
        'Num. of Observation: {:5.0f}'.format(len(y)),
        ha='right',
    )
    plt.savefig(path)
    plt.close()
    return corr

def fig_corr_on_each_group(group, group_key, target, x_column, y_column, png_dir):
    group['adjust_corr'] = group[group_key].map(
        lambda key: 
        fig_corr(
            target[target[group_key] == key][x_column], 
            pd.Series(target[target[group_key] == key][y_column]), 
            os.path.join(png_dir, f'corr_{key}.png'))
    )

    group = df_merge(
        left=group,
        right=target.groupby([group_key])[x_column, y_column].max().reset_index().rename(columns={x_column:f'max_{x_column}', y_column: f'max_{y_column}'}),
        on=[group_key],
        how='left'
    )
    group = df_merge(
        left=group,
        right=target.groupby([group_key])[y_column].std().reset_index().rename(columns={y_column: f'std_{y_column}'}),
        on=[group_key],
        how='left'
    )

    tmp = target.copy()
    tmp['data_num'] = 1
    group = df_merge(
        left=group,
        right=tmp.groupby([group_key])['data_num'].sum().reset_index(),
        on=[group_key],
        how='left'
    )
    
    return group


# TODO test
# return cluster_id
def df_kmeans_clustering(ds, cluster_by_columns, cluster_rate=0.1):
    n_clusters = int(len(ds) * cluster_rate)
    if n_clusters < 1:
        n_clusters = 1
    print('n_ds:', len(ds))
    print('n_clusters:', n_clusters)
    
    _units = []
    for unit in cluster_by_columns:
        _units.append(ds[unit].tolist())
    cust_array = np.array(_units).T
    pred = KMeans(n_clusters=n_clusters).fit_predict(cust_array)
    return pred.astype(str)



# tested
from ipywidgets import interact, IntSlider
from IPython.display import display

def freeze_header(df, num_rows=30, num_columns=10, step_rows=1,
                  step_columns=1):
    """
    Freeze the headers (column and index names) of a Pandas DataFrame. A widget
    enables to slide through the rows and columns.
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame to display
    num_rows : int, optional
        Number of rows to display
    num_columns : int, optional
        Number of columns to display
    step_rows : int, optional
        Step in the rows
    step_columns : int, optional
        Step in the columns
    Returns
    -------
    Displays the DataFrame with the widget
    """
    @interact(last_row=IntSlider(min=min(num_rows, df.shape[0]),
                                 max=df.shape[0],
                                 step=step_rows,
                                 description='rows',
                                 readout=False,
                                 disabled=False,
                                 continuous_update=True,
                                 orientation='horizontal',
                                 slider_color='purple'),
              last_column=IntSlider(min=min(num_columns, df.shape[1]),
                                    max=df.shape[1],
                                    step=step_columns,
                                    description='columns',
                                    readout=False,
                                    disabled=False,
                                    continuous_update=True,
                                    orientation='horizontal',
                                    slider_color='purple'))
    def _freeze_header(last_row, last_column):
        display(df.iloc[max(0, last_row-num_rows):last_row,
                        max(0, last_column-num_columns):last_column])


        
    def solve2(a, b, c):
        _d = b**2 - 4*a*c
        if _d < 0:
            return np.array([])
        return np.array([(-b + np.sqrt(_d) ) / (2*a), (-b - np.sqrt(_d) ) / (2*a)])
    
     def solve2_positive(f, var_x):
        a =  np.array([f.coeff(var_x, 2)], dtype='float')
        b =  np.array([f.coeff(var_x, 1)], dtype='float')
        c =  np.array([f.coeff(var_x, 0)], dtype='float')
        solved = solve2(a,b,c)
        positive_solved = solved[solved >= 0]
        if len(positive_solved) == 0:
            return None
        return positive_solved.min()

    
