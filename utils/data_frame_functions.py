import numpy as np
import params as p 
import pandas as pd

def group_df(df, by):
    
    grouped = df.groupby([by], sort = True)
    current_group = 0
    groups = [None] * grouped.ngroups

    for group_id, rows in grouped:
        groups[current_group] = {}
        for column in df.columns:
            if column == by:
                groups[current_group][column] =  np.ones_like(rows[p.X].values)*group_id
            else:
                groups[current_group][column] = rows[column].values   
        current_group+= 1
    return groups

def group2df(group):
    df_list = []
    for item in group:
        df=pd.DataFrame.from_dict(item)
        df_list.append(df)
    
    df = pd.concat(df_list, sort=True)
    return df 
