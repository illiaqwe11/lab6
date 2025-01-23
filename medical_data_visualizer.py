import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_data(df):
    """Додає стовпець overweight, що вказує чи є людина з надмірною вагою."""
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['overweight'] = df['bmi'].apply(lambda x: 1 if x > 25 else 0)
    
    df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
    df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)
    
    return df

def draw_cat_plot(df):
    """Побудова категоріального графіку."""
    df_cat = pd.melt(df, id_vars=["cardio"], value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])
    df_cat = df_cat.groupby(["cardio", "variable", "value"]).size().reset_index(name="total")
    df_cat.rename(columns={"variable": "feature"}, inplace=True)
    
    g = sns.catplot(data=df_cat, x="feature", hue="value", col="cardio", kind="count", height=5, aspect=1)
    g.set_axis_labels("Feature", "Count")
    g.set_titles("Cardio: {col_name}")
    
    return g.fig

def draw_heat_map(df):
    """Побудова теплової карти."""
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
                 (df['height'] >= df['height'].quantile(0.025)) & 
                 (df['height'] <= df['height'].quantile(0.975)) & 
                 (df['weight'] >= df['weight'].quantile(0.025)) & 
                 (df['weight'] <= df['weight'].quantile(0.975))]

    corr = df_heat.corr()

    mask = corr.where(pd.np.triu(pd.np.ones(corr.shape), k=1).astype(bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.5)
    
    return plt.gcf()
