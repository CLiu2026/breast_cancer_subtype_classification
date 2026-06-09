import pandas as pd

def load_data(filepath):
    """加载原始数据，设置样本编号为索引"""
    df = pd.read_csv(filepath, index_col=0)
    return df

def clean_data(df):
    """
    清洗数据：
    - 剔除 cell_line
    - 保留 normal + 四种肿瘤亚型
    返回清洗后的 DataFrame
    """
    cancer_types = ['basal', 'HER', 'luminal_A', 'luminal_B']
    keep = cancer_types + ['normal']
    df_clean = df[df['type'].isin(keep)].copy()
    return df_clean

def create_binary_label(df):
    """添加二分类标签:0=normal, 1=tumor"""
    df['binary_label'] = (df['type'] != 'normal').astype(int)
    return df