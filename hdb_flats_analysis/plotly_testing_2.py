import datetime as dt
import math
import os
from ast import Index
from turtle import width

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

resale_file_loc = "../datasets/resale_hdb_price_coords_mrt_01jul.csv"
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
resale_df["year"] = resale_df["month"].dt.year

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


def rgb_to_rgba(rgb, a):
    """Converts rgb color notation into rgba notation, with a representing opacity"""
    rgb_trunc = rgb[4:-1]
    return f"rgba({rgb_trunc},{a})"


def hex_to_rgba(hexcode, a):
    """Converts hex color notation into rgba notation, with a representing opacity"""
    if len(hexcode) == 7:
        rgb = tuple(int(hexcode[i : i + 2], 16) for i in (1, 3, 5))
    else:
        rgb = tuple(int(hexcode[i : i + 2], 16) for i in (0, 2, 4))

    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{a})"


def obtain_period_string(row, period):
    """Create string notation of the period frequency"""
    if period == "YS":
        return row["month"].strftime("%Y")
    elif period == "QS":
        mth = row["month"].strftime("%b")
        year = row["month"].strftime("%Y")
        if mth == "Jan":
            return f"Q1 {year}"
        elif mth == "Apr":
            return f"Q2 {year}"
        elif mth == "Jul":
            return f"Q3 {year}"
        elif mth == "Oct":
            return f"Q4 {year}"
    elif period == "MS":
        return row["month"].strftime("%b %Y")


def dataframe_wrangling(period, town: bool = False, flat_type: bool = False):
    grouping_keys = [pd.Grouper(key="month", freq=period)]
    if town is True:
        grouping_keys.append("planning_area_ura")
    if flat_type == True:
        grouping_keys.append("flat_type")
    result = resale_df.groupby(grouping_keys)[["resale_price", "price_per_sqft"]].agg(
        {"resale_price": ["count", "mean"], "price_per_sqft": "mean"}
    )
    result = result.droplevel(level=0, axis=1)
    result.columns = ["num_trans", "avg_resale_price", "avg_price_psf"]
    result = result.reset_index()
    result["period_str"] = result.apply(obtain_period_string, axis=1, period=period)
    result["avg_resale_price_k"] = result["avg_resale_price"] / 1000

    return result


################################# COMPONENTS ##################################

t4_select_town_dropdown = dcc.Dropdown(
    id="t4-town-dropdown",
    options=[
        {"label": x.title(), "value": x}
        for x in np.sort(resale_df["planning_area_ura"].unique())
    ],
    placeholder="Drill down to a particular town: ",
    value="ANG MO KIO",
    clearable=False,
    multi=False,
)

t4_period_frequency = dcc.Dropdown(
    id="t4-period-frequency",
    options=[
        {"label": "Year", "value": "YS"},
        {"label": "Quarter", "value": "QS"},
        {"label": "Month", "value": "MS"},
    ],
    multi=False,
    value="YS",
    clearable=False,
)

t4_flat_type_dropdown = dcc.Dropdown(
    id="t4-flat-type-dropdown",
    options=[
        {"label": x.title(), "value": x}
        for x in np.sort(resale_df["flat_type"].unique())
    ],
    multi=False,
    value="1 ROOM",
    clearable=False,
)


t4_row1 = dbc.Row(
    html.H1(
        children="Town-specific Dashboard:",
        id="t4-town-title",
        className="text-center mt-4 mb-4 fw-bolder",
    ),
    align="center",
)

t4_row2 = dbc.Row(
    [
        dbc.Col(
            [
                html.H6("View more details about flats in this town: "),
                t4_select_town_dropdown,
            ],
            width={"size": 5},
        ),
        dbc.Col(
            [
                html.H6("View dashboard in this frequency: "),
                t4_period_frequency,
            ],
            width={"size": 5},
        ),
    ],
    align="center",
    class_name="mb-4",
)

t4_row3 = dbc.Row(
    [
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(
                        children="",
                        id="t4-num-trans-title",
                        class_name="card-title fw-bold text-center",
                    ),
                    dbc.CardBody(
                        [
                            html.H6(
                                children="",
                                id="t4-num-trans-value",
                                className="card-title text-center display-6 fw-bolder",
                            ),
                            dcc.Graph(id="t4-num-trans-graph", figure={}),
                        ]
                    ),
                ]
            ),
            width={"size": 4},
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(
                        children="",
                        id="t4-resale-price-title",
                        class_name="card-title fw-bold text-center",
                    ),
                    dbc.CardBody(
                        [
                            html.H6(
                                children="",
                                id="t4-resale-price-value",
                                className="card-title text-center display-6 fw-bolder",
                            ),
                            dcc.Graph(id="t4-resale-price-graph", figure={}),
                        ]
                    ),
                ]
            ),
            width={"size": 4},
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(
                        children="",
                        id="t4-price-per-sqft-title",
                        class_name="card-title fw-bold text-center",
                    ),
                    dbc.CardBody(
                        [
                            html.H6(
                                children="",
                                id="t4-price-per-sqft-value",
                                className="card-title text-center display-6 fw-bolder",
                            ),
                            dcc.Graph(
                                id="t4-price-per-sqft-graph",
                                figure={},
                            ),
                        ]
                    ),
                ]
            ),
            width={"size": 4},
        ),
    ],
    class_name="g-1",
)


t4_row4 = dbc.Row(
    [
        dbc.Col(
            [
                html.H6(children="Drill down further into specific flat types"),
                t4_flat_type_dropdown,
            ],
            width={"size": 5},
        ),
        dbc.Col("", width={"size": 5}),
    ],
    align="center",
    class_name="mt-4 mb-4",
)

t4_row5 = dbc.Row(
    [
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.P(
                                children="",
                                id="t4-num-trans-flattype-title-1",
                                className="card-title fw-bold text-center mt-0 mb-0",
                            ),
                            html.P(
                                children="",
                                id="t4-num-trans-flattype-title-2",
                                className="card-title fw-bold text-center mt-0 mb-0",
                            ),
                        ]
                    ),
                    dbc.CardBody(
                        [
                            html.H6(
                                children="",
                                id="t4-num-trans-flattype-value",
                                className="card-title text-center display-6 fw-bolder",
                            ),
                            dcc.Graph(id="t4-num-trans-flattype-graph", figure={}),
                        ]
                    ),
                ]
            ),
            width={"size": 4},
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.P(
                                children="",
                                id="t4-resale-price-flattype-title-1",
                                className="card-title fw-bold text-center mt-0 mb-0",
                            ),
                            html.P(
                                children="",
                                id="t4-resale-price-flattype-title-2",
                                className="card-title fw-bold text-center mt-0 mb-0",
                            ),
                        ]
                    ),
                    dbc.CardBody(
                        [
                            html.H6(
                                children="",
                                id="t4-resale-price-flattype-value",
                                className="card-title text-center display-6 fw-bolder",
                            ),
                            dcc.Graph(id="t4-resale-price-flattype-graph", figure={}),
                        ]
                    ),
                ]
            ),
            width={"size": 4},
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.H6(
                                children="",
                                id="t4-psf-flattype-title-1",
                                className="card-title fw-bold text-center mt-0 mb-0",
                            ),
                            html.H6(
                                children="",
                                id="t4-psf-flattype-title-2",
                                className="card-title fw-bold text-center mt-0 mb-0",
                            ),
                        ]
                    ),
                    dbc.CardBody(
                        [
                            html.H6(
                                children="",
                                id="t4-psf-flattype-value",
                                className="card-title text-center display-6 fw-bolder",
                            ),
                            dcc.Graph(
                                id="t4-psf-flattype-graph",
                                figure={},
                            ),
                        ]
                    ),
                ]
            ),
            width={"size": 4},
        ),
    ]
)

tab4 = dbc.Container([t4_row1, t4_row2, t4_row3, t4_row4, t4_row5])


#################################### DASH #####################################

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = tab4


################################## CALLBACKS ##################################


@app.callback(
    Output(component_id="t4-town-title", component_property="children"),
    Input(component_id="t4-town-dropdown", component_property="value"),
)
def update_title(town):
    town = town.title()
    return f"Town-specific Dashboard: {town}"


@app.callback(
    Output(component_id="t4-num-trans-title", component_property="children"),
    Output(component_id="t4-num-trans-value", component_property="children"),
    Output(component_id="t4-num-trans-graph", component_property="figure"),
    Input(component_id="t4-town-dropdown", component_property="value"),
    Input(component_id="t4-period-frequency", component_property="value"),
)
def update_num_trans_card(town, period):

    # 1: Group records by the chosen time frequency and towns
    town_period_grouped = dataframe_wrangling(period, town=True, flat_type=False)

    max_trans = town_period_grouped["num_trans"].max()
    min_trans = town_period_grouped["num_trans"].min()

    # 2. Filter records by the town chosen by the audience
    town_period_grouped = town_period_grouped.query("planning_area_ura == @town")

    # 3: find the latest entry
    latest_entry = town_period_grouped.loc[town_period_grouped["month"].idxmax(), :]
    latest_period_str = latest_entry["period_str"]
    latest_period_val = latest_entry["num_trans"]

    # 4: Preparing title label and value
    town_title = town.title()
    title = f"No. flats (re)sold for {town_title} in {latest_period_str}"
    value = f"{latest_period_val:,} flat(s) sold"

    # 5: Preparing to draw average line
    town_period_avg = dataframe_wrangling(period, town=True, flat_type=False)
    town_avg = town_period_avg.groupby([pd.Grouper(key="month", freq=period)])[
        ["num_trans"]
    ].agg("mean")
    town_avg = town_avg.reset_index()

    fig = px.line(
        data_frame=town_period_grouped,
        x="month",
        y="num_trans",
        height=350,
        markers=True,
        custom_data=["period_str", "num_trans"],
        range_y=(min_trans, max_trans),
    )
    fig.update_traces(
        hovertemplate="<b>%{customdata[1]:,}</b>", name="No. sold"  # for updating
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)

    scatter_temp = go.Scatter(
        name="Avg. sold",
        x=town_avg["month"],
        y=town_avg["num_trans"],
        mode="lines",
        line=dict(color="grey", width=3, dash="dot"),
        customdata=np.array(town_avg["num_trans"]).reshape((-1, 1)),
        showlegend=False,
    )

    fig.add_trace(scatter_temp)
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]:,.0f}</b>",
        selector={"name": "Avg. sold"},
    )
    fig.update_layout(hovermode="x unified")

    return title, value, fig


@app.callback(
    Output(component_id="t4-resale-price-title", component_property="children"),
    Output(component_id="t4-resale-price-value", component_property="children"),
    Output(component_id="t4-resale-price-graph", component_property="figure"),
    Input(component_id="t4-town-dropdown", component_property="value"),
    Input(component_id="t4-period-frequency", component_property="value"),
)
def update_resale_price_card(town, period):

    # 1: Group records by the chosen time frequency and towns
    town_period_grouped = dataframe_wrangling(period, town=True, flat_type=False)

    max_resale_price = town_period_grouped["avg_resale_price"].max()
    min_resale_price = town_period_grouped["avg_resale_price"].min()

    # 2: Filter records by the town chosen by the audience
    town_period_grouped = town_period_grouped.query("planning_area_ura == @town")

    # 3: Obtain details for the most recent period
    latest_entry = town_period_grouped.loc[town_period_grouped["month"].idxmax(), :]
    latest_period_str = latest_entry["period_str"]
    latest_period_val = latest_entry["avg_resale_price"]

    # 4: Preparing title label and value
    town_title = town.title()
    title = f"Avg. Resale Price for {town_title} in {latest_period_str}"
    value = f"SGD {latest_period_val:,.0f}"

    # 5: Preparing dataframe to draw average line
    avg_period_grouped = dataframe_wrangling(period, town=False, flat_type=False)

    fig = px.line(
        data_frame=town_period_grouped,
        x="month",
        y="avg_resale_price",
        height=350,
        markers=True,
        range_y=(min_resale_price, max_resale_price),
        custom_data=["period_str", "avg_resale_price_k"],
    )
    fig.update_traces(name="Price", hovertemplate="SGD <b>%{customdata[1]:,.0f}K</b>")
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)

    scatter_temp = go.Scatter(
        name="Avg. price",
        x=avg_period_grouped["month"],
        y=avg_period_grouped["avg_resale_price"],
        mode="lines",
        line=dict(color="grey", width=3, dash="dot"),
        customdata=np.array(avg_period_grouped["avg_resale_price_k"]).reshape((-1, 1)),
        showlegend=False,
    )

    fig.add_trace(scatter_temp)
    fig.update_traces(
        hovertemplate="SGD <b>%{customdata[0]:,.0f}K</b>",
        selector={"name": "Avg. price"},
    )
    fig.update_layout(hovermode="x unified")

    return title, value, fig


@app.callback(
    Output(component_id="t4-price-per-sqft-title", component_property="children"),
    Output(component_id="t4-price-per-sqft-value", component_property="children"),
    Output(component_id="t4-price-per-sqft-graph", component_property="figure"),
    Input(component_id="t4-town-dropdown", component_property="value"),
    Input(component_id="t4-period-frequency", component_property="value"),
)
def update_price_per_sqft_card(town, period):

    # 1: Group records by the chosen time frequency and towns
    town_period_grouped = dataframe_wrangling(period, town=True, flat_type=False)

    max_price_per_sqft = town_period_grouped["avg_price_psf"].max()
    min_price_per_sqft = town_period_grouped["avg_price_psf"].min()

    # 2: Filter records by the town chosen by the audience
    town_period_grouped = town_period_grouped.query("planning_area_ura == @town")

    # 3: Obtain details for the most recent period
    latest_entry = town_period_grouped.loc[town_period_grouped["month"].idxmax(), :]
    latest_period_str = latest_entry["period_str"]
    latest_period_val = latest_entry["avg_price_psf"]

    # 4: Preparing title label and value
    town_title = town.title()
    title = f"Avg. price psf for {town_title} in {latest_period_str}"
    value = f"SGD {latest_period_val:,.0f}"

    # 5: Preparing dataframe to draw average line
    avg_period_grouped = dataframe_wrangling(period, town=False, flat_type=False)

    fig = px.line(
        data_frame=town_period_grouped,
        x="month",
        y="avg_price_psf",
        height=350,
        markers=True,
        range_y=(min_price_per_sqft, max_price_per_sqft),
        custom_data=["period_str", "avg_price_psf"],
    )
    fig.update_traces(
        hovertemplate="SGD <b>%{customdata[1]:,.0f}</b> psf", name="Price psf"
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)

    scatter_temp = go.Scatter(
        name="Avg. price psf",
        x=avg_period_grouped["month"],
        y=avg_period_grouped["avg_price_psf"],
        mode="lines",
        line=dict(color="grey", width=3, dash="dot"),
        customdata=np.stack(avg_period_grouped["avg_price_psf"]).reshape((-1, 1)),
        showlegend=False,
    )

    fig.add_trace(scatter_temp)
    fig.update_traces(
        hovertemplate="SGD <b>%{customdata[0]:,.0f}</b> psf",
        selector={"name": "Avg. price psf"},
    )
    fig.update_layout(hovermode="x unified")

    return title, value, fig


@app.callback(
    Output(component_id="t4-num-trans-flattype-title-1", component_property="children"),
    Output(component_id="t4-num-trans-flattype-title-2", component_property="children"),
    Output(component_id="t4-num-trans-flattype-value", component_property="children"),
    Output(component_id="t4-num-trans-flattype-graph", component_property="figure"),
    Input(component_id="t4-town-dropdown", component_property="value"),
    Input(component_id="t4-period-frequency", component_property="value"),
    Input(component_id="t4-flat-type-dropdown", component_property="value"),
)
def update_num_trans_flattype_card(town, period, flat_type):
    # 1: Group records by the chosen time frequency and towns
    town_period_grouped = dataframe_wrangling(period, town=True, flat_type=True)

    max_trans = town_period_grouped["num_trans"].max()
    min_trans = town_period_grouped["num_trans"].min()

    # 2: Filter records by the town chosen by the audience
    town_period_grouped_filter = town_period_grouped.query(
        "planning_area_ura == @town & flat_type == @flat_type"
    )

    # 3: Obtain details for the most recent period
    if town_period_grouped_filter.shape[0] > 0:
        latest_entry = town_period_grouped_filter.loc[
            town_period_grouped_filter["month"].idxmax(), :
        ]
        most_recent_period_val = latest_entry["num_trans"]
    else:
        most_recent_period_val = 0

    # 4: Preparing title label and value
    town_title = town.title()
    flat_type_title = flat_type.title().replace(" ", "-")
    title_1 = f"No. of {flat_type_title} flats"
    title_2 = f"(re)sold in {town_title}"
    value = f"{most_recent_period_val:,} flat(s) sold"

    # 5: Preparing dataframe to draw average line
    town_period_avg = dataframe_wrangling(period, town=True, flat_type=True)
    town_period_avg = town_period_avg.query("flat_type == @flat_type")
    town_avg = town_period_avg.groupby([pd.Grouper(key="month", freq=period)])[
        ["num_trans"]
    ].agg("mean")
    town_avg = town_avg.reset_index()

    fig = px.line(
        data_frame=town_period_grouped_filter,
        x="month",
        y="num_trans",
        height=350,
        markers=True,
        custom_data=["period_str", "num_trans"],
        range_y=(min_trans, max_trans),
    )
    fig.update_traces(
        hovertemplate="<b>%{customdata[1]:,}</b>", name="No. sold"  # for updating
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)

    scatter_temp = go.Scatter(
        name="Avg. sold",
        x=town_avg["month"],
        y=town_avg["num_trans"],
        mode="lines",
        line=dict(color="grey", width=3, dash="dot"),
        customdata=np.array(town_avg["num_trans"]).reshape((-1, 1)),
        showlegend=False,
    )

    fig.add_trace(scatter_temp)
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]:,.0f}</b>",
        selector={"name": "Avg. sold"},
    )
    fig.update_layout(hovermode="x unified")

    return title_1, title_2, value, fig


@app.callback(
    Output(
        component_id="t4-resale-price-flattype-title-1", component_property="children"
    ),
    Output(
        component_id="t4-resale-price-flattype-title-2", component_property="children"
    ),
    Output(
        component_id="t4-resale-price-flattype-value", component_property="children"
    ),
    Output(component_id="t4-resale-price-flattype-graph", component_property="figure"),
    Input(component_id="t4-town-dropdown", component_property="value"),
    Input(component_id="t4-period-frequency", component_property="value"),
    Input(component_id="t4-flat-type-dropdown", component_property="value"),
)
def update_resale_price_flattype_card(town, period, flat_type):
    # 1: Group records by the chosen time frequency and towns
    town_period_grouped = dataframe_wrangling(period, town=True, flat_type=True)

    max_price = town_period_grouped["avg_resale_price"].max()
    min_price = town_period_grouped["avg_resale_price"].min()

    # 2: Filter records by the town chosen by the audience
    town_period_grouped_filter = town_period_grouped.query(
        "planning_area_ura == @town & flat_type == @flat_type"
    )

    # 3: Preparing title label and value
    town_title = town.title()
    flat_type_title = flat_type.title().replace(" ", "-")
    title_1 = f"Avg. Resale Price of {flat_type_title} flats"
    title_2 = f"(re)sold in {town_title}"

    # 4: Obtain details for the most recent period, accounting for 0 records in the filter
    if town_period_grouped_filter.shape[0] > 0:
        latest_entry = town_period_grouped_filter.loc[
            town_period_grouped_filter["month"].idxmax(), :
        ]
        most_recent_price_val = latest_entry["avg_resale_price"]
        value = f"SGD {most_recent_price_val:,.0f}"
    else:
        most_recent_price_val = "None"
        value = f"SGD -"

    # 5: Preparing dataframe to draw average line
    town_period_avg = dataframe_wrangling(period, town=False, flat_type=True)
    town_avg = town_period_avg.query("flat_type == @flat_type")

    fig = px.line(
        data_frame=town_period_grouped_filter,
        x="month",
        y="avg_resale_price",
        height=350,
        markers=True,
        custom_data=["period_str", "avg_resale_price_k"],
        range_y=(min_price, max_price),
    )
    fig.update_traces(
        hovertemplate="<b>SGD %{customdata[1]:.0f} K</b>", name="Price"  # for updating
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)

    scatter_temp = go.Scatter(
        name="Avg. Price",
        x=town_avg["month"],
        y=town_avg["avg_resale_price"],
        mode="lines",
        line=dict(color="grey", width=3, dash="dot"),
        customdata=np.array(town_avg["avg_resale_price_k"]).reshape((-1, 1)),
        showlegend=False,
    )

    fig.add_trace(scatter_temp)
    fig.update_traces(
        hovertemplate="SGD <b>%{customdata[0]:.0f} K</b>",
        selector={"name": "Avg. Price"},
    )
    fig.update_layout(hovermode="x unified")

    return title_1, title_2, value, fig


@app.callback(
    Output(component_id="t4-psf-flattype-title-1", component_property="children"),
    Output(component_id="t4-psf-flattype-title-2", component_property="children"),
    Output(component_id="t4-psf-flattype-value", component_property="children"),
    Output(component_id="t4-psf-flattype-graph", component_property="figure"),
    Input(component_id="t4-town-dropdown", component_property="value"),
    Input(component_id="t4-period-frequency", component_property="value"),
    Input(component_id="t4-flat-type-dropdown", component_property="value"),
)
def update_resale_price_flattype_card(town, period, flat_type):
    # 1: Group records by the chosen time frequency and towns
    town_period_grouped = dataframe_wrangling(period, town=True, flat_type=True)

    max_price = town_period_grouped["avg_price_psf"].max()
    min_price = town_period_grouped["avg_price_psf"].min()

    # 2: Filter records by the town chosen by the audience
    town_period_grouped_filter = town_period_grouped.query(
        "planning_area_ura == @town & flat_type == @flat_type"
    )

    # 4: Preparing title label and value
    town_title = town.title()
    flat_type_title = flat_type.title().replace(" ", "-")
    title_1 = f"Avg. price psf for {flat_type_title}"
    title_2 = f"in {town_title}"

    # 3: Obtain details for the most recent period
    if town_period_grouped_filter.shape[0] > 0:
        latest_entry = town_period_grouped_filter.loc[
            town_period_grouped_filter["month"].idxmax(), :
        ]
        most_recent_price_psf = latest_entry["avg_price_psf"]
        value = f"SGD {most_recent_price_psf:,.0f}"
    else:
        most_recent_price_psf = "None"
        value = f"SGD -"

    # 5: Preparing dataframe to draw average line
    town_period_avg = dataframe_wrangling(period, town=False, flat_type=True)
    town_avg = town_period_avg.query("flat_type == @flat_type")

    fig = px.line(
        data_frame=town_period_grouped_filter,
        x="month",
        y="avg_price_psf",
        height=350,
        markers=True,
        custom_data=["period_str", "avg_price_psf"],
        range_y=(min_price, max_price),
    )
    fig.update_traces(
        hovertemplate="SGD <b>%{customdata[1]:,}</b>", name="Price psf."  # for updating
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)

    scatter_temp = go.Scatter(
        name="Avg. price psf",
        x=town_avg["month"],
        y=town_avg["avg_price_psf"],
        mode="lines",
        line=dict(color="grey", width=3, dash="dot"),
        customdata=np.array(town_avg["avg_price_psf"]).reshape((-1, 1)),
        showlegend=False,
    )

    fig.add_trace(scatter_temp)
    fig.update_traces(
        hovertemplate="SGD <b>%{customdata[0]:,.0f}</b>",
        selector={"name": "Avg. price psf"},
    )
    fig.update_layout(hovermode="x unified")

    return title_1, title_2, value, fig


if __name__ == "__main__":
    app.run(debug=True)
