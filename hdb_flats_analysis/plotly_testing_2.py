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

school_file_loc = "schools_for_plotly.csv"
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
    rgb_trunc = rgb[4:-1]
    return f"rgba({rgb_trunc},{a})"


def hex_to_rgba(hexcode, a):
    if len(hexcode) == 7:
        rgb = tuple(int(hexcode[i : i + 2], 16) for i in (1, 3, 5))
    else:
        rgb = tuple(int(hexcode[i : i + 2], 16) for i in (0, 2, 4))

    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{a})"


def convert_month_to_quarter(month):
    temp = month.strftime("%b").strip()
    if temp == "Jan":
        return "Q1"
    elif temp == "Apr":
        return "Q2"
    elif temp == "Jul":
        return "Q3"
    elif temp == "Oct":
        return "Q4"


def obtain_period_string(row, period):
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


################################# COMPONENTS ##################################

t1_period_frequency = dcc.Dropdown(
    id="t1_period_frequency",
    options=[
        {"label": "Year", "value": "YS"},
        {"label": "Quarter", "value": "QS"},
        {"label": "Month", "value": "MS"},
    ],
    multi=False,
    value="YS",
    clearable=False,
)

t1_town_dropdown = dcc.Dropdown(
    id="t1_town_dropdown",
    options=[
        {"label": x.title(), "value": x}
        for x in np.sort(resale_df["planning_area_ura"].unique())
    ],
    placeholder="Drill down to a particular town: ",
    value="ANG MO KIO",
    clearable=False,
    multi=False,
)
t1_flat_type_selection_dropdown = dcc.Dropdown(
    id="t1_flat_type_selection_dropdown",
    options=[{"label": "All", "value": "all"}]
    + [
        {"label": x.title(), "value": x}
        for x in np.sort(resale_df["flat_type"].unique())
    ],
    multi=True,
    value="all",
    clearable=False,
)

t1_select_plot_type = dcc.Dropdown(
    id="t1_select_plot_type",
    options=[
        {"label": "Number of Transactions", "value": "num_trans"},
        {"label": "Avg. Resale Price (SGD)", "value": "avg_resale_price"},
        {"label": "Avg. Price per Sqft (SGD)", "value": "avg_price_psf"},
    ],
    value="num_trans",
    multi=False,
)

tab_1 = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [html.H6("Select a Period", className="fw-bold"), t1_period_frequency],
                width={"size": 4, "offset": 1},
            )
        ),
        dbc.Row(
            dbc.Col(
                [html.H6("Select town: ", className="fw-bold"), t1_town_dropdown],
                width={"size": 4, "offset": 1},
            )
        ),
        dbc.Row(
            dbc.Col(
                [
                    html.H6("Include only the flat types of: "),
                    t1_flat_type_selection_dropdown,
                ],
                width={"offset": 1, "size": 4},
            )
        ),
        dbc.Row(
            dbc.Col(
                [html.H6("Select Plot type: "), t1_select_plot_type],
                width={"offset": 1, "size": 4},
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.Graph(id="t1_line_graph_plot", figure={}),
                width={"offset": 1, "size": 4},
            )
        ),
    ]
)


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

t4_row1 = dbc.Row(
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
)
t4_row2 = dbc.Row(
    html.H1(
        children="Town-specific Dashboard:",
        id="t4-town-title",
        className="text-center mt-4 mb-4 fw-bolder",
    ),
    align="center",
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

tab4 = dbc.Container([t4_row1, t4_row2, t4_row3])


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

    town_filter = resale_df.query("planning_area_ura == @town")
    town_period_grouped = town_filter.groupby([pd.Grouper(key="month", freq=period)])[
        "resale_price"
    ].size()
    town_period_grouped.name = "num_trans"
    town_period_grouped = town_period_grouped.reset_index()
    town_period_grouped["period_str"] = town_period_grouped.apply(
        obtain_period_string, axis=1, period=period
    )

    latest_entry = town_period_grouped.loc[town_period_grouped["month"].idxmax(), :]
    latest_period_str = latest_entry["period_str"]
    latest_period_val = latest_entry["num_trans"]

    town_title = town.title()
    title = f"No. flats (re)sold for {town_title} in {latest_period_str}"
    value = f"{latest_period_val:,} flat(s) sold"

    town_period_avg = resale_df.groupby(
        [pd.Grouper(key="month", freq=period), "planning_area_ura"]
    )["resale_price"].size()
    town_period_avg.name = "num_trans"
    town_period_avg = town_period_avg.reset_index()

    town_avg = town_period_avg.groupby(pd.Grouper(key="month", freq=period))[
        ["num_trans"]
    ].agg("mean")
    town_avg.columns = ["avg_num_trans"]
    town_avg = town_avg.reset_index()

    fig = px.line(
        data_frame=town_period_grouped,
        x="month",
        y="num_trans",
        height=350,
        markers=True,
        custom_data=["period_str", "num_trans"],
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
        y=town_avg["avg_num_trans"],
        mode="lines",
        line=dict(color="grey", width=3, dash="dot"),
        customdata=np.array(town_avg["avg_num_trans"]).reshape((-1, 1)),
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

    town_filter = resale_df.query("planning_area_ura == @town")
    town_period_grouped = town_filter.groupby([pd.Grouper(key="month", freq=period)])[
        "resale_price"
    ].agg("mean")
    town_period_grouped = town_period_grouped.reset_index()
    town_period_grouped["resale_price_k"] = town_period_grouped["resale_price"] / 1000
    town_period_grouped["period_str"] = town_period_grouped.apply(
        obtain_period_string, axis=1, period=period
    )

    latest_entry = town_period_grouped.loc[town_period_grouped["month"].idxmax(), :]
    latest_period_str = latest_entry["period_str"]
    latest_period_val = latest_entry["resale_price"]

    town_title = town.title()
    title = f"Avg. Resale Price for {town_title} in {latest_period_str}"
    value = f"SGD {latest_period_val:,.0f}"

    avg_period_grouped = resale_df.groupby([pd.Grouper(key="month", freq=period)])[
        "resale_price"
    ].agg("mean")
    avg_period_grouped.name = "avg_resale_price"
    avg_period_grouped = avg_period_grouped.reset_index()
    avg_period_grouped["avg_resale_price_k"] = (
        avg_period_grouped["avg_resale_price"] / 1000
    )

    fig = px.line(
        data_frame=town_period_grouped,
        x="month",
        y="resale_price",
        height=350,
        markers=True,
        custom_data=["period_str", "resale_price_k"],
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
        customdata=np.stack(
            [
                avg_period_grouped["avg_resale_price"],
                avg_period_grouped["avg_resale_price_k"],
            ],
            axis=-1,
        ),
        showlegend=False,
    )

    fig.add_trace(scatter_temp)
    fig.update_traces(
        hovertemplate="SGD <b>%{customdata[1]:,.0f}K</b>",
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

    town_filter = resale_df.query("planning_area_ura == @town")
    town_period_grouped = town_filter.groupby([pd.Grouper(key="month", freq=period)])[
        "price_per_sqft"
    ].agg("mean")
    town_period_grouped.name = "avg_price_per_sqft"
    town_period_grouped = town_period_grouped.reset_index()
    town_period_grouped["avg_price_per_sqft_k"] = (
        town_period_grouped["avg_price_per_sqft"] / 1000
    )
    town_period_grouped["period_str"] = town_period_grouped.apply(
        obtain_period_string, axis=1, period=period
    )

    latest_entry = town_period_grouped.loc[town_period_grouped["month"].idxmax(), :]
    latest_period_str = latest_entry["period_str"]
    latest_period_val = latest_entry["avg_price_per_sqft"]

    town_title = town.title()
    title = f"Avg. price psf for {town_title} in {latest_period_str}"
    value = f"SGD {latest_period_val:,.0f}"

    # Compute the average number of transactions
    avg_period_grouped = resale_df.groupby([pd.Grouper(key="month", freq=period)])[
        "price_per_sqft"
    ].agg("mean")
    avg_period_grouped.name = "avg_price_per_sqft"
    avg_period_grouped = avg_period_grouped.reset_index()
    avg_period_grouped["avg_price_per_sqft_k"] = (
        avg_period_grouped["avg_price_per_sqft"] / 1000
    )

    latest_avg_entry = avg_period_grouped.loc[avg_period_grouped["month"].idxmax(), :]
    latest_avg_val = latest_avg_entry["avg_price_per_sqft"]

    avg_string = f"Avg. price psf in {latest_period_str}: SGD {latest_avg_val:,.0f}"

    fig = px.line(
        data_frame=town_period_grouped,
        x="month",
        y="avg_price_per_sqft",
        height=350,
        markers=True,
        custom_data=["period_str", "avg_price_per_sqft"],
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
        y=avg_period_grouped["avg_price_per_sqft"],
        mode="lines",
        line=dict(color="grey", width=3, dash="dot"),
        customdata=np.stack(
            [
                avg_period_grouped["avg_price_per_sqft"],
                avg_period_grouped["avg_price_per_sqft_k"],
            ],
            axis=-1,
        ),
        showlegend=False,
    )

    fig.add_trace(scatter_temp)
    fig.update_traces(
        hovertemplate="SGD <b>%{customdata[0]:,.0f}</b> psf",
        selector={"name": "Avg. price psf"},
    )
    fig.update_layout(hovermode="x unified")

    return title, value, fig


# @app.callback(
#     Output(component_id="t1_line_graph_plot", component_property="figure"),
#     Input(component_id="t1_period_frequency", component_property="value"),
#     Input(component_id="t1_town_dropdown", component_property="value"),
#     Input(component_id="t1_flat_type_selection_dropdown", component_property="value"),
#     Input(component_id="t1_select_plot_type", component_property="value"),
# )
# def update_t1_plot(period, town, flat_types, var):

#     town_filter = resale_df.query("planning_area_ura == @town")
#     if "all" in flat_types:
#         town_filter = town_filter.copy()
#     else:
#         town_filter = town_filter.query("flat_type in @flat_types")

#     # Group by the period, and get the mean of both the average resale price and average price per square foot
#     town_grouped = town_filter.groupby([pd.Grouper(key="month", freq=period)])[
#         ["resale_price", "price_per_sqft"]
#     ].agg({"resale_price": ["mean", "count"], "price_per_sqft": ["mean"]})
#     town_grouped = town_grouped.droplevel(0, axis=1)
#     town_grouped.columns = ["avg_resale_price", "num_trans", "avg_price_psf"]
#     town_grouped = town_grouped.reset_index()

#     # Get the % change
#     town_grouped["pchg_avg_resale_price"] = town_grouped[
#         "avg_resale_price"
#     ].pct_change()
#     town_grouped["pchg_num_trans"] = town_grouped["num_trans"].pct_change()
#     town_grouped["pchg_avg_price_psf"] = town_grouped["avg_price_psf"].pct_change()

#     # Formatting % change strings
#     town_grouped["pchg_avg_resale_price_str"] = town_grouped[
#         "pchg_avg_price_psf"
#     ].apply(lambda x: f"{x * 100:.2f}" if not np.isnan(x) else "-")
#     town_grouped["pchg_num_trans_str"] = town_grouped["pchg_num_trans"].apply(
#         lambda x: f"{x * 100:.2f}" if not np.isnan(x) else "-"
#     )
#     town_grouped["pchg_avg_price_psf_str"] = town_grouped["pchg_avg_price_psf"].apply(
#         lambda x: f"{x * 100:.2f}" if not np.isnan(x) else "-"
#     )

#     # Formatting period strings
#     if period == "YS":
#         town_grouped["period_str"] = town_grouped["month"].dt.strftime("%Y")
#     elif period == "QS":
#         town_grouped["period_str"] = (
#             town_grouped["month"].apply(convert_month_to_quarter)
#             + " "
#             + town_grouped["month"].dt.strftime("%Y")
#         )
#     elif period == "MS":
#         town_grouped["period_str"] = town_grouped["month"].dt.strftime("%b %Y")

#     scatter_colors = {
#         "pos": pio.templates["seaborn"]["layout"]["colorway"][2],
#         "neg": pio.templates["seaborn"]["layout"]["colorway"][3],
#         "nan": pio.templates["seaborn"]["layout"]["colorway"][0],
#     }

#     fig = px.line(
#         data_frame=town_grouped,
#         x="month",
#         y=var,
#         template="seaborn",
#         custom_data=[
#             "period_str",
#             "avg_resale_price",
#             "pchg_avg_resale_price_str",
#             "num_trans",
#             "pchg_num_trans_str",
#             "avg_price_psf",
#             "pchg_avg_price_psf_str",
#         ],
#         labels={
#             "month": "Year",
#             "avg_resale_price": "Avg. Resale Price (SGD)",
#             "num_trans": "Number of transactions",
#             "avg_price_psf": "Average Price per Square Foot (SGD)",
#         },
#         markers=True,
#         hover_data={
#             "month": False,
#             "avg_resale_price": False,
#             "num_trans": False,
#             "avg_price_psf": False,
#         },
#     )
#     fig.update_traces(line_color="gray")

#     # Plotting Markers
#     for_scatter_plotting = []
#     scatter_df_names = ["pos", "neg", "nan"]

#     if var == "avg_resale_price":
#         for_scatter_plotting.append(town_grouped.query("pchg_avg_resale_price > 0"))
#         for_scatter_plotting.append(town_grouped.query("pchg_avg_resale_price < 0"))
#         for_scatter_plotting.append(
#             town_grouped.query("pchg_avg_resale_price.isnull()")
#         )
#     elif var == "avg_price_psf":
#         for_scatter_plotting.append(town_grouped.query("pchg_avg_price_psf > 0"))
#         for_scatter_plotting.append(town_grouped.query("pchg_avg_price_psf < 0"))
#         for_scatter_plotting.append(town_grouped.query("pchg_avg_price_psf.isnull()"))
#     elif var == "num_trans":
#         for_scatter_plotting.append(town_grouped.query("pchg_num_trans > 0"))
#         for_scatter_plotting.append(town_grouped.query("pchg_num_trans < 0"))
#         for_scatter_plotting.append(town_grouped.query("pchg_num_trans.isnull()"))

#     for df, name in zip(for_scatter_plotting, scatter_df_names):
#         scatter_temp = go.Scatter(
#             name=name,
#             x=df["month"],
#             y=df[var],
#             mode="markers",
#             customdata=np.stack(
#                 [
#                     df["period_str"],
#                     df["avg_resale_price"],
#                     df["pchg_avg_resale_price_str"],
#                     df["num_trans"],
#                     df["pchg_num_trans_str"],
#                     df["avg_price_psf"],
#                     df["pchg_avg_price_psf_str"],
#                 ],
#                 axis=-1,
#             ),
#             marker=dict(
#                 color=scatter_colors[name],
#                 size=10,
#                 line=dict(color="black", width=1),  # border color  # border width
#             ),
#         )
#         fig.add_trace(scatter_temp)

#         hover_template = "Period: <b>%{customdata[0]}</b><br>"
#         if var == "avg_resale_price":
#             hover_template += "Avg. resale price: <b>SGD %{customdata[1]:,.0f}</b><br>Change: <b>%{customdata[2]} %"
#         elif var == "num_trans":
#             hover_template += "Transactions: <b>%{customdata[3]:,}</b><br>Change: <b>%{customdata[4]} %"
#         elif var == "avg_price_psf":
#             hover_template += "Avg. price per sqft: <b>SGD %{customdata[5]:,.0f}</b><br>Change: <b>%{customdata[6]} %"

#         fig.update_traces(
#             hoverlabel=dict(
#                 bgcolor=rgb_to_rgba(scatter_colors[name], 0.5),
#             ),
#             hovertemplate=hover_template,
#             selector=({"name": name}),
#         )

#         fig.update_layout(
#             showlegend=False,
#         )

#     # Plotting Average Line
#     town_avg = resale_df.groupby([pd.Grouper(key="month", freq=period)])[
#         ["resale_price", "price_per_sqft"]
#     ].agg({"resale_price": ["mean", "count"], "price_per_sqft": ["mean"]})
#     town_avg = town_avg.droplevel(0, axis=1)
#     town_avg.columns = ["avg_resale_price", "num_trans", "avg_price_psf"]
#     town_avg["num_trans"] = (
#         town_avg["num_trans"] / resale_df["planning_area_ura"].nunique()
#     )
#     town_avg = town_avg.reset_index()

#     avg_line = go.Scatter(
#         x=town_avg["month"],
#         y=town_avg[var],
#         mode="lines",
#         line=dict(width=4, color="gray", dash="dot"),
#         name="avg",
#         showlegend=False,
#         hovertemplate="",
#         hoverinfo="skip",
#     )
#     fig.add_trace(avg_line)

#     return fig


if __name__ == "__main__":
    app.run(debug=True)
