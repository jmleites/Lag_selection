import pandas as pd

from codebase.workflows.config import LAGS

lags_df = pd.DataFrame(LAGS).reset_index()
lags_df = lags_df.rename(columns={'index':'Method'}).sort_values('Method')

tab = lags_df.to_latex(caption='caption', label='tab:selectedlags', index=False)
print(tab)
