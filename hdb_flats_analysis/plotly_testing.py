import datetime as dt
import math
import os
from ast import Index

import dash
import dash_bootstrap_components as dbc
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from click import option
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output, State
from dateutil.relativedelta import relativedelta

resale_file_loc = "../datasets/resale_hdb_price_coords_mrt_20jun.csv"
resale_df = pd.read_csv(
    resale_file_loc,
    parse_dates=["month", "lease_commence_date"],
    index_col=0,
    dtype={"x": "float64", "y": "float64", "postal": "object"},
    low_memory=False,
)
resale_df["remaining_lease"] = resale_df["lease_commence_date"].apply(
    lambda x: relativedelta(
        x + relativedelta(years=99), dt.date.today().replace(day=1)
    ).years
)

mrt_lrt_file_loc = "../datasets/mrt_lrt_stations.csv"
mrt_df = pd.read_csv(mrt_lrt_file_loc, index_col=0)

mrt_lines = mrt_df["color"].unique()
mrt_line_color_dict = {
    "Red": "#c03731",
    "Green": "#1f844d",
    "Purple": "#953aa6",
    "Orange": "#fba01d",
    "Blue": "#134f9a",
    "Brown": "#9d6633",
    "Grey": "#8c958c",
}

school_file_loc = "../datasets/schools_for_plotly.csv"
school_df = pd.read_csv(school_file_loc, index_col=0)
school_df = school_df.query("mainlevel_code in ['PRIMARY', 'MIXED LEVELS']")

school_levels = school_df["mainlevel_code"].unique()
school_color_dict = {
    "PRIMARY": "#ADD8E6",  # Light Blue
    "SECONDARY": "#4682B4",  # Medium Blue
    "JUNIOR COLLEGE": "#1E3A5F",  # Dark Blue
    "CENTRALISED INSTITUTE": "#1E3A5F",  # Dark Blue
    "MIXED LEVELS": "#1E3A5F",  # Dark Blue
    # "MIXED ": "#6A5ACD",  # Slate Blue
}

############################## UTILITY FUNCTIONS ##############################


def hex_to_rgba(hexcode, a):
    if len(hexcode) == 7:
        rgb = tuple(int(hexcode[i : i + 2], 16) for i in (1, 3, 5))
    else:
        rgb = tuple(int(hexcode[i : i + 2], 16) for i in (0, 2, 4))

    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{a})"


#################################### TAB 1 ####################################


#################################### DASH #####################################

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.H1(
            children="Resale Market Dashboard",
            className="mb-3 mt-3 bg-white rounded text-center fw-bold",
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.H6("Select to filter between flat types: ", className="mt-2"),
                    width={"size": 2},
                ),
                dbc.Col(
                    dcc.Checklist(
                        id="enable-flat-type-check",
                        options=[{"label": "enabled", "value": "enable"}],
                        className="text-left mt-1",
                    ),
                    width={"size": 1},
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="flat-type-dropdown",
                        options=[
                            {"label": x, "value": x}
                            for x in np.sort(resale_df["flat_type"].unique())
                        ]
                        + [{"label": "All", "value": "All"}],
                        # placeholder,
                        disabled=True,
                        multi=True,
                    ),
                    width={"size": 4},
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id="plot-variable",
                            options=[
                                {
                                    "label": "No. Resale Flat Transactions",
                                    "value": "num_trans",
                                },
                                {
                                    "label": "Median Resale Price",
                                    "value": "median_resale_price",
                                },
                                {
                                    "label": "Median Price per Square Feet",
                                    "value": "median_price_psf",
                                },
                            ],
                            value="num_trans",
                            multi=False,
                        )
                    ]
                )
            ]
        ),
        dcc.Dropdown(
            id="dropdown-ura-planning-areas",
            options=[
                {"label": x, "value": x}
                for x in np.sort(resale_df["planning_area_ura"].unique())
            ],
            multi=True,
            placeholder="Select Period of Frequency",
        ),
        dcc.Dropdown(
            id="dropdown-avg-resale-price-period",
            options=[
                {"label": "Year", "value": "YS"},
                {"label": "Quarter", "value": "QS"},
                {"label": "Month", "value": "MS"},
            ],
            value="YS",
            multi=False,
        ),
        dcc.Graph(id="graph-avg-resale-price", figure={}),
        dcc.Graph(id="facet-graph-towns-compare", figure={}),
    ],
    fluid=True,
)


################################## CALLBACKS ##################################


@app.callback(
    Output(component_id="flat-type-dropdown", component_property="placeholder"),
    Output(component_id="flat-type-dropdown", component_property="value"),
    Output(component_id="flat-type-dropdown", component_property="disabled"),
    Input(component_id="enable-flat-type-check", component_property="value"),
    prevent_initial_call=True,
)
def update(check):
    if len(check) > 0:
        return "", ["All"], False
    return "Enable filtering by flat type", ["All"], True


@app.callback(
    Output(component_id="graph-avg-resale-price", component_property="figure"),
    Input(component_id="dropdown-avg-resale-price-period", component_property="value"),
    Input(component_id="flat-type-dropdown", component_property="value"),
    Input(component_id="plot-variable", component_property="value"),
    # prevent_initial_call=True,
)
def update_avg_price_graph(period, flat_type, var):
    if flat_type is None:
        flat_type = ["All"]

    # Visualising average price of flats over the years
    df_grouped = resale_df.groupby([pd.Grouper(key="month", freq=period), "flat_type"])[
        ["resale_price", "price_per_sqft"]
    ].agg(
        {
            "resale_price": ["count", "median"],
            "price_per_sqft": ["median"],
        }
    )
    df_grouped = df_grouped.droplevel(level=0, axis=1)
    df_grouped.columns = [
        "num_trans",
        "median_resale_price",
        "median_price_psf",
    ]
    df_grouped = df_grouped.reset_index()
    df_grouped["pct_chg_num_trans"] = np.round(
        df_grouped.groupby(["flat_type"])["num_trans"].pct_change() * 100, 2
    )
    df_grouped["pct_chg_median_resale_price"] = np.round(
        df_grouped.groupby(["flat_type"])["median_resale_price"].pct_change() * 100, 2
    )
    df_grouped["pct_chg_median_price_psf"] = np.round(
        df_grouped.groupby(["flat_type"])["median_price_psf"].pct_change() * 100, 2
    )

    if "All" not in flat_type:
        df_grouped = df_grouped.query("flat_type in @flat_type")
    if period == "YS":
        df_grouped["month_strf"] = df_grouped["month"].dt.strftime("%Y")
    elif period == "QS":
        df_grouped["month_strf"] = df_grouped["month"].dt.strftime("%B-%Y")
    elif period == "MS":
        df_grouped["month_strf"] = df_grouped["month"].dt.strftime("%B-%Y")

    fig = px.line(
        data_frame=df_grouped,
        x="month",
        y=var,
        color="flat_type",
        category_orders={"flat_type": np.sort(resale_df["flat_type"].unique())},
        height=800,
        markers=True,
        template="seaborn",
        custom_data=["month_strf", "median_resale_price", "median_price_psf"],
        labels={
            "flat_type": "Flat Type",
            "month": "Year",
            "num_trans": "Number of Transactions",
            "median_price_psf": "Median Price per Square Foot",
            "median_resale_price": "Median Resale Price",
        },
    )

    return fig


@app.callback(
    Output(component_id="facet-graph-towns-compare", component_property="figure"),
    Input(component_id="dropdown-avg-resale-price-period", component_property="value"),
    Input(component_id="dropdown-ura-planning-areas", component_property="value"),
    Input(component_id="plot-variable", component_property="value"),
    # prevent_initial_call=True,
)
def update_facet_graph(period, towns, var):
    if towns is None:
        towns = ["All"]

    # This dataframe is used to draw the facet grids
    df_filtered = resale_df.query("planning_area_ura in @towns")
    df_grouped = df_filtered.groupby(
        [pd.Grouper(key="month", freq=period), "flat_type", "planning_area_ura"]
    )[["resale_price", "price_per_sqft"]].agg(
        {
            "resale_price": ["count", "median"],
            "price_per_sqft": ["median"],
        }
    )
    df_grouped = df_grouped.droplevel(level=0, axis=1)
    df_grouped.columns = [
        "num_trans",
        "median_resale_price",
        "median_price_psf",
    ]

    df_grouped = df_grouped.reset_index()
    df_grouped["pct_chg_num_trans"] = np.round(
        df_grouped.groupby(["flat_type"])["num_trans"].pct_change() * 100, 2
    )
    df_grouped["pct_chg_median_resale_price"] = np.round(
        df_grouped.groupby(["flat_type"])["median_resale_price"].pct_change() * 100, 2
    )
    df_grouped["pct_chg_median_price_psf"] = np.round(
        df_grouped.groupby(["flat_type"])["median_price_psf"].pct_change() * 100, 2
    )

    if period == "YS":
        df_grouped["month_strf"] = df_grouped["month"].dt.strftime("%Y")
    elif period == "QS":
        df_grouped["month_strf"] = df_grouped["month"].dt.strftime("%B-%Y")
    elif period == "MS":
        df_grouped["month_strf"] = df_grouped["month"].dt.strftime("%B-%Y")

    # For drawing the average line in the facet grid
    df_avg = resale_df.groupby([pd.Grouper(key="month", freq=period), "flat_type"])[
        ["resale_price", "price_per_sqft"]
    ].agg(
        {
            "resale_price": ["count", "median"],
            "price_per_sqft": ["median"],
        }
    )
    df_avg = df_avg.droplevel(level=0, axis=1)
    df_avg.columns = [
        "num_trans",
        "median_resale_price",
        "median_price_psf",
    ]

    df_avg = df_avg.reset_index()
    df_avg["num_trans"] = df_avg["num_trans"] / resale_df["planning_area_ura"].nunique()
    df_avg["pct_chg_num_trans"] = np.round(
        df_avg.groupby(["flat_type"])["num_trans"].pct_change() * 100, 2
    )
    df_avg["pct_chg_median_resale_price"] = np.round(
        df_avg.groupby(["flat_type"])["median_resale_price"].pct_change() * 100, 2
    )
    df_avg["pct_chg_median_price_psf"] = np.round(
        df_avg.groupby(["flat_type"])["median_price_psf"].pct_change() * 100, 2
    )

    if period == "YS":
        df_avg["month_strf"] = df_avg["month"].dt.strftime("%Y")
    elif period == "QS":
        df_avg["month_strf"] = df_avg["month"].dt.strftime("%B-%Y")
    elif period == "MS":
        df_avg["month_strf"] = df_avg["month"].dt.strftime("%B-%Y")

    uniq_flat_types = df_grouped["flat_type"].unique()
    df_avg = df_avg.query("flat_type in @uniq_flat_types")

    num_cols = math.ceil(df_avg["flat_type"].nunique() / 2)

    fig = px.line(
        data_frame=df_grouped,
        x="month",
        y=var,
        color="planning_area_ura",
        category_orders={
            "planning_area_ura": np.sort(df_grouped["planning_area_ura"].unique())
        },
        height=1000,
        facet_col="flat_type",
        facet_col_wrap=num_cols,
        template="seaborn",
        markers=True,
        custom_data=["month"],
    )

    for index, flat_type in enumerate(np.sort(df_avg["flat_type"].unique())):
        temp_df = df_avg.query("flat_type == @flat_type")
        col_index = index % num_cols + 1
        row_index = 2 if index // num_cols + 1 == 1 else 1
        fig.add_trace(
            go.Scatter(
                x=temp_df["month"],
                y=temp_df[var],
                mode="lines",
                line=dict(width=4, color="gray", dash="dot"),
                name="AVERAGE",
                showlegend=False,
                # hoverinfo="skip",
            ),
            col=col_index,
            row=row_index,
        )

    return fig


if __name__ == "__main__":
    app.run(debug=True)
