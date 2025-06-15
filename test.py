# %%
import polars as pl
import plotly.express as px
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from sklearn.linear_model import LinearRegression


# %%
df = pl.read_csv('https://datasets-nixtla.s3.amazonaws.com/air-passengers.csv', schema_overrides={"ds": pl.Date})
df.head()


# %%
df['unique_id'].value_counts()

# %%
fig = px.line(df, x="ds", y="y")
fig.show()

# %%
fcst = MLForecast(
    models=LinearRegression(),
    freq='1mo',  # our serie has a monthly frequency
    lags=[12],
    target_transforms=[Differences([1])],
)
fcst.fit(df)

# %%
preds = fcst.predict(12)
preds = preds.with_columns(
    (pl.col("unique_id") + "_prediction").alias("unique_id"),
    pl.col("LinearRegression").alias("y").cast(pl.Int64)
).select(
    pl.col("unique_id"),
    pl.col("ds"),
    pl.col("y"),
)

# %%
preds

# %%
fig = px.line(pl.concat([df, preds]), x="ds", y="y", color="unique_id")
fig.show()

# %%
