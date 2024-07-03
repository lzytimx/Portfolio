import datetime as dt
import os

import dash
import dash_bootstrap_components as dbc
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
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
# school_df = school_df.query("mainlevel_code in ['PRIMARY', 'MIXED LEVELS']")

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


def pvif(interest, years):
    return (1 - (1 / (1 + interest) ** years)) / interest


def leasehold_pvif(years):
    return np.round(pvif(0.035, years) / pvif(0.035, 999), 3)


#################################### TAB 1 #####################################

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

#################################### TAB 2 ####################################

t2_filter_options_title = dbc.Row(
    dbc.Col(
        html.H4("", className="text-start fw-bold"),
        width={"offset": 1},
    )
)

# Region and Planning Areas
t2_filter_options_row1 = dbc.Row(
    [
        dbc.Col(
            html.P("Region(s):", className="mt-2"),
            width={"size": 1},
            className="mt-1 mb-1",
        ),
        dbc.Col(
            dcc.Dropdown(
                id="t2-geography-dropdown",
                options=[
                    {"label": x.title(), "value": x}
                    for x in resale_df["region_ura"].unique()
                ]
                + [{"label": "All", "value": "all"}],
                multi=True,
                # value="CENTRAL REGION",
                className="mt-1 mb-1",
            ),
            width={"size": 3},
        ),
        dbc.Col(
            dcc.Dropdown(
                id="t2-town-dropdown",
                options=[
                    {"label": x.title(), "value": x}
                    for x in mrt_df["planning_area_ura"].unique()
                ],
                placeholder="Select town(s):",
                multi=True,
                className="mt-1 mb-1",
                searchable=True,
                style={"overflow-y": "visible"},
            ),
            width={"size": 6},
        ),
    ],
    justify="center",
)

# X months, flat_type, amenities
t2_filter_options_row2 = dbc.Row(
    [
        dbc.Col(
            [
                html.P(
                    children="Period:",
                    id="t2-title-dropdown-x-months",
                    className="mb-1 mt-1",
                ),
                dcc.Dropdown(
                    id="t2-dropdown-x-months",
                    multi=False,
                    clearable=False,
                    placeholder="Plot transactions for the past:",
                    value=1,
                    options=[
                        {"label": f"{x} month(s)", "value": x}
                        for x in [1, 3, 6, 12, 24, 36, 48]
                    ]
                    + [{"label": "All", "value": 120}],
                    className="mb-1",
                ),
            ],
            width={"size": 3},
        ),
        dbc.Col(
            [
                html.P(
                    children="Flat type(s)",
                    id="t2-title-dropdown-flat-type",
                    className="mb-1 mt-1",
                ),
                dcc.Dropdown(
                    id="t2-dropdown-flat-type",
                    # multi=True,
                    placeholder="Select flat type(s):",
                    value="4 ROOM",
                    clearable=True,
                    options=[
                        {"label": x.title().replace(" ", "-"), "value": x}
                        for x in np.sort(resale_df["flat_type"].unique())
                    ],
                    className="mb-1",
                ),
            ],
            width={"offset": 0, "size": 4},
        ),
        dbc.Col(
            [
                html.P(
                    children="Amenities: ",
                    id="t2-title-dropdown-amenities",
                    className="mb-1 mt-1",
                ),
                dcc.Dropdown(
                    id="t2-dropdown-amenities",
                    multi=True,
                    value=[],
                    placeholder="Choose amenities for plotting: ",
                    options=[
                        {"label": "MRTs", "value": "MRT"},
                        {"label": "Schools", "value": "school"},
                    ],
                    className="mb-1",
                ),
            ],
            width={"offset": 0, "size": 3},
        ),
    ],
    justify="center",
)

# vix_dropdown
t2_filter_options_row3 = dbc.Row(
    [
        dbc.Col(
            html.P(
                "Select Plot: ",
                className="mb-1 mt-4 text-end pt-1 fw-bold",
            ),
            width={"size": 1},
        ),
        dbc.Col(
            dcc.Dropdown(
                id="t2-viz-options",
                options=[
                    {"label": "Avg. resale price", "value": "avg_resale_price"},
                    {"label": "Avg. price sqft", "value": "avg_price_psf"},
                    {"label": "Remaining lease", "value": "remaining_lease"},
                    {
                        "label": "Distance to closest MRT/LRT (m)",
                        "value": "distance_to_mrt_meters",
                    },
                    {
                        "label": "Distance to closest Primary School (m)",
                        "value": "distance_to_pri_school_meters",
                    },
                ],
                className="mb-1 mt-4",
                value="avg_resale_price",
            ),
            width={"size": 9},
        ),
    ],
    justify="center",
)

t2_filter_mapbox_graph = dbc.Row(
    [
        dbc.Col(dcc.Graph(id="t2-mapbox-graph", figure={}), className=""),
    ],
    justify="center",
)

t2_filter_data_table = dbc.Row(
    [
        dbc.Col(
            [
                html.P(
                    id="t2-data-table-title",
                    className="fw-bold",
                    children="Dataframe of transactions for the selected period",
                ),
                html.Pre(id="t2-data-table", className="mb-2"),
            ],
            width={"size": 11},
        )
    ],
    justify="center",
)

#################################### TAB 3 ####################################

most_exp_flat = resale_df.loc[resale_df["resale_price"].idxmax(), :]
most_exp_flat_building_name = most_exp_flat["building"].title()
most_exp_flat_blk = most_exp_flat["blk_no"]
most_exp_flat_road = most_exp_flat["road_name"].title()
most_exp_flat_postal = most_exp_flat["postal"]
most_exp_flat_type = most_exp_flat["flat_type"].title()
most_exp_flat_model = most_exp_flat["flat_model"].title()
most_exp_flat_price = np.round(most_exp_flat["resale_price"] / 1_000_000, 2)
most_exp_flat_mth = most_exp_flat["month"].strftime("%B %Y")
most_exp_flat_psf = most_exp_flat["price_per_sqft"]

t3_most_exp_card = dbc.Card(
    [
        dbc.CardHeader(children="Most expensive flat to-date:", class_name="fw-bold"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.CardImg(
                        src="https://live.staticflickr.com/65535/51000598339_47be46cba3_b.jpg",
                        className="img-fluid rounded-start",
                    ),
                    className="col-md-5",
                ),
                dbc.Col(
                    dbc.CardBody(
                        [
                            html.H4(
                                f"${most_exp_flat_price} Million; ${most_exp_flat_psf:,.0f} psf",
                                className="card-title",
                            ),
                            html.P(
                                [
                                    most_exp_flat_building_name,
                                    html.Br(),
                                    f"{most_exp_flat_blk} {most_exp_flat_road}, S{most_exp_flat_postal}",
                                    html.Br(),
                                    f"{most_exp_flat_type} HDB Flat; {most_exp_flat_model} Model",
                                ],
                                className="card-text",
                            ),
                            html.Small(
                                f"Sold: {most_exp_flat_mth}",
                                className="card-text text-muted",
                            ),
                        ]
                    ),
                    className="col-md-7",
                ),
            ],
            className="g-0 d-flex align-items-center",
        ),
    ],
    className="mb-3",
)

mill_dollar_trans = resale_df.query("resale_price > 1_000_000 & year == 2024").shape[0]
all_trans = resale_df.query("year == 2024").shape[0]
proportion_mill = np.round((mill_dollar_trans / all_trans * 100), 2)

t3_gen_info_card = dbc.Card(
    [
        dbc.CardHeader(
            children="General Information",
            class_name="card-title fw-bold",
        ),
        dbc.CardBody(
            [
                html.P("No. of Million-dollar Flats in 2024", className="card-text"),
                html.H5(
                    f"{mill_dollar_trans} of {all_trans:,} transactions",
                    className="card-title fw_bold",
                ),
                html.P(
                    "Proportion of Million-dollar Flats in 2024",
                    className="card-text",
                ),
                html.H5(f"{proportion_mill:.2f}%", className="card-title fw_bold"),
            ],
            class_name="mb-1",
        ),
    ],
    class_name="mb-2 d-flex",
)

t3_graph_dropdown = dcc.Dropdown(
    id="t3-graph-dropdown",
    options=[
        {"label": "Yearly", "value": "YS"},
        {"label": "Quarterly", "value": "QS"},
        {"label": "Monthly", "value": "MS"},
    ],
    className="mt-2",
    value="YS",
    multi=False,
)

t3_period_dropdown = dcc.Dropdown(
    id="t3-period-dropdown",
    options=[{"label": f"{x} month(s)", "value": x} for x in [1, 3, 6, 12, 24, 36, 48]],
    value=3,
    multi=False,
)

t3_prop_million_flats = dcc.Graph(id="t3-prop-million-flats", figure={})

t3_million_flats_mapbox = dcc.Graph(id="t3-million-flats-mapbox", figure={})


t3_general_info = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H6("Select a Period:"), class_name="pt-3 col-md-3"
                        ),
                        dbc.Col(t3_graph_dropdown, class_name="col-md-9"),
                    ],
                    justify="left",
                ),
                t3_prop_million_flats,
            ],
            width={"size": 5},
        ),
        dbc.Col([t3_gen_info_card, t3_most_exp_card], width={"size": 5}),
    ],
    justify="center",
)

t3_figures = dbc.Row(
    [
        dbc.Col(
            [html.H3("Select a Period: "), t3_period_dropdown, t3_million_flats_mapbox],
            width={"size": 10},
        )
    ],
    justify="center",
)

#################################### TABS #####################################

tab1_card = dbc.Card(
    dbc.CardBody([t4_row1, t4_row2, t4_row3, t4_row4, t4_row5]),
    class_name="mt-3",
)

tab2_card = dbc.Card(
    dbc.CardBody(
        [
            t2_filter_options_title,
            t2_filter_options_row1,
            t2_filter_options_row2,
            t2_filter_options_row3,
            t2_filter_mapbox_graph,
            t2_filter_data_table,
        ],
    ),
    className="mt-3",
)

tab3_card = dbc.Card(dbc.CardBody([t3_general_info, t3_figures]), class_name="mt-3")


tabs = dbc.Tabs(
    [
        dbc.Tab(tab1_card, label="Tab 1"),
        dbc.Tab(tab2_card, label="Map"),
        dbc.Tab(tab3_card, label="Million-dollar Flats"),
    ],
    id="card-tabs",
    active_tab="Map",
)

#################################### DASH #####################################

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.H1(
            children="Resale Market Dashboard",
            className="mb-3 mt-3 bg-white rounded text-center fw-bold",
        ),
        dbc.Card(
            dbc.CardHeader(tabs),
        ),
    ],
    fluid=True,
)

############################### CALLBACKS TAB 1 ###############################


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


############################### CALLBACKS TAB 2 ###############################


@app.callback(
    Output(component_id="t2-town-dropdown", component_property="value"),
    Input(component_id="t2-geography-dropdown", component_property="value"),
    prevent_initial_call=True,
)
def update_dropdown(geog_value):
    region_dict = {
        i: mrt_df.query("region_ura == @i")["planning_area_ura"].unique()
        for i in mrt_df["region_ura"].unique()
    }

    if "all" in geog_value:
        return mrt_df["planning_area_ura"].unique()
    else:
        result = []
        for i in geog_value:
            result = np.concatenate([result, region_dict[i]])
        return pd.Series(result).unique()


@app.callback(
    Output(component_id="t2-mapbox-graph", component_property="figure"),
    Input(component_id="t2-town-dropdown", component_property="value"),
    Input(component_id="t2-dropdown-x-months", component_property="value"),
    Input(component_id="t2-dropdown-flat-type", component_property="value"),
    Input(component_id="t2-dropdown-amenities", component_property="value"),
    Input(component_id="t2-viz-options", component_property="value"),
    State(component_id="t2-mapbox-graph", component_property="figure"),
    # prevent_initial_call=True,
)
def update_map(towns, months, flat_type, ancillary, viz, fig_state):
    # Filter by region
    if towns is None:
        towns = []
    df_filtered = resale_df.query("planning_area_ura in @towns")

    # Filter by x_months
    today = dt.date.today()
    past_x_month = today.replace(day=1) - relativedelta(months=months)
    df_filtered = df_filtered.query("month >= @past_x_month")

    # Filter by flat_type
    if flat_type is None:
        flat_type = []
    df_filtered = df_filtered.query("flat_type in @flat_type")

    # perform Grouping
    df_grouped = (
        df_filtered.groupby(
            [
                "blk_no",
                "road_name",
                "postal",
                "latitude",
                "longitude",
                "planning_area_ura",
                "flat_type",
                "closest_mrt_station",
                "distance_to_mrt_meters",
                "transport_type",
                "line_color",
                "closest_pri_school",
                "distance_to_pri_school_meters",
                "remaining_lease",
            ]
        )[["price_per_sqft", "resale_price"]]
        .agg({"price_per_sqft": ["mean", "count"], "resale_price": "mean"})
        .droplevel(level=0, axis=1)
    )
    df_grouped.columns = ["avg_price_psf", "num_trans", "avg_resale_price"]
    df_grouped = df_grouped.reset_index()

    # This is used to for setting range in colorbar
    past_year = today.replace(day=1) - relativedelta(months=12)
    df_pastyr = resale_df.query("month >= @past_year")

    # The range of the dataset uses the entire dataset as reference to ensure
    # that the range of the dataset remains constant when switching between regions
    viz_dict = {
        "avg_resale_price": [
            df_pastyr["resale_price"].quantile(0.3),
            df_pastyr["resale_price"].quantile(0.9),
        ],
        "avg_price_psf": [
            df_pastyr["price_per_sqft"].quantile(0.3),
            df_pastyr["price_per_sqft"].quantile(0.9),
        ],
        "remaining_lease": [
            df_pastyr["remaining_lease"].min(),
            df_pastyr["remaining_lease"].max(),
        ],
        "distance_to_mrt_meters": [
            df_pastyr["distance_to_mrt_meters"].min(),
            df_pastyr["distance_to_mrt_meters"].max(),
        ],
        "distance_to_pri_school_meters": [
            df_pastyr["distance_to_pri_school_meters"].min(),
            df_pastyr["distance_to_pri_school_meters"].max(),
        ],
    }

    fig = px.scatter_mapbox(
        data_frame=df_grouped,
        lat="latitude",
        lon="longitude",
        color=viz,
        range_color=[
            viz_dict.get(viz)[0],
            viz_dict.get(viz)[1],
        ],
        color_continuous_scale="Bluered",
        category_orders={"flat_type": np.sort(resale_df["flat_type"].unique())},
        custom_data=[
            "blk_no",
            "road_name",  # 1
            "postal",
            "avg_resale_price",
            "planning_area_ura",
            "flat_type",  # 5
            "num_trans",
            "avg_price_psf",
            "remaining_lease",
            "closest_mrt_station",
            "distance_to_mrt_meters",  # 10
            "transport_type",
            "closest_pri_school",
            "distance_to_pri_school_meters",  # 13
        ],
        height=800,
        labels={
            "avg_resale_price": "Avg. Resale Price",
            "avg_price_psf": "Avg. Price psf",
        },
    )
    hover_template = """
    <b>%{customdata[5]} FLAT - %{customdata[6]} Sold </b><br>
    Town: %{customdata[4]}<br>
    Address: Blk %{customdata[0]} %{customdata[1]}, %{customdata[2]}<br>
    Avg. Resale Price: SGD %{customdata[3]:,.0f}<br>
    Avg. Price psf: SGD %{customdata[7]:,.0f}<br>
    Remaining Lease: %{customdata[8]} years<br>
    Closest MRT Station: %{customdata[9]} %{customdata[11]}; %{customdata[10]:,.0f} m<br>
    Closest Pri. School: %{customdata[12]}; %{customdata[13]:,.0f} m
    """

    fig.update_traces(hovertemplate=hover_template)
    fig.update_layout(coloraxis_colorbar_title_text="")

    if "MRT" in ancillary:

        mrt_lines = mrt_df["color"].unique()

        # Filter MRT lines for plotting
        mrt_filtered = mrt_df.query("planning_area_ura in @towns")

        mrt_line_list = []
        for line in mrt_lines:
            mrt_line_list.append(mrt_filtered.query("color == @line"))

        for line_color, df in zip(mrt_lines, mrt_line_list):
            # This creates the black border of the MRT markers
            fig.add_scattermapbox(
                lat=df["latitude"],
                lon=df["longitude"],
                hoverinfo="skip",
                marker=dict(
                    color="black",
                    size=13,
                ),
                showlegend=False,
                name="border",
            )
            # This creates the MRT markers
            fig.add_scattermapbox(
                lat=df["latitude"],
                lon=df["longitude"],
                customdata=np.stack(
                    [
                        df["code"],
                        df["station_name"],
                        df["type"],
                        df["planning_area_ura"].str.title(),
                    ],
                    axis=-1,
                ),
                hovertemplate="<b>%{customdata[0]}: %{customdata[1]} %{customdata[2]} Station</b><br>Town: %{customdata[3]}",
                hoverlabel=go.scattermapbox.Hoverlabel(
                    bgcolor=hex_to_rgba(mrt_line_color_dict[line_color], 0.6)
                ),
                # marker=go.scattermapbox.Marker(color=mrt_line_color_dict[line_color], size=10, line=dict(width=2, color='black')),
                marker=dict(color=mrt_line_color_dict[line_color], size=10),
                showlegend=False,
                name=line_color,
            )

    if "school" in ancillary:
        school_filtered = school_df.query("planning_area_ura in @towns")

        school_levels_list = []
        for level in school_levels:
            school_levels_list.append(school_filtered.query("mainlevel_code == @level"))

        for level, df in zip(school_levels, school_levels_list):
            # This creates the black border of the MRT markers
            fig.add_scattermapbox(
                lat=df["latitude"],
                lon=df["longitude"],
                hoverinfo="skip",
                marker=dict(
                    color="black",
                    size=13,
                ),
                showlegend=False,
                name="border",
            )
            # This creates the MRT markers
            fig.add_scattermapbox(
                lat=df["latitude"],
                lon=df["longitude"],
                customdata=np.stack(
                    [
                        df["school_name"],
                        df["blk_no"],
                        df["road_name"],
                        df["postal"],
                    ],
                    axis=-1,
                ),
                hovertemplate="<b>%{customdata[0]}</b>",
                hoverlabel=go.scattermapbox.Hoverlabel(
                    bgcolor=hex_to_rgba(school_color_dict[level], 0.6)
                ),
                marker=dict(color=school_color_dict[level], size=10),
                showlegend=False,
                name=level,
            )

    if fig_state == {}:
        center_lat = 1.35
        center_lon = 103.8
        zoom = 11
    else:
        center_lat = fig_state["layout"]["mapbox"]["center"]["lat"]
        center_lon = fig_state["layout"]["mapbox"]["center"]["lon"]
        zoom = fig_state["layout"]["mapbox"]["zoom"]

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_accesstoken=os.environ.get("MAPBOX_ACCESS_TOKEN"),
        mapbox_center_lat=center_lat,
        mapbox_center_lon=center_lon,
        mapbox_zoom=zoom,
        showlegend=True,
        clickmode="event",
    )

    return fig


@app.callback(
    Output(component_id="t2-data-table", component_property="children"),
    Input(component_id="t2-mapbox-graph", component_property="clickData"),
    Input(component_id="t2-dropdown-x-months", component_property="value"),
    Input(component_id="t2-dropdown-flat-type", component_property="value"),
    prevent_initial_call=True,
)
def update_data_table(click_data, x_months, flat_type):

    if click_data is None:
        return None

    # Filter by postal code
    postal_code = click_data.get("points")[0].get("customdata")[2]
    df_filtered = resale_df.query("postal == @postal_code")

    # Filter by x_months
    today = dt.date.today()
    past_x_month = today.replace(day=1) - relativedelta(months=x_months)
    df_filtered = df_filtered.query("month >= @past_x_month")

    # Filter by flat_type
    df_filtered = df_filtered.query("flat_type in @flat_type")

    df_filtered = df_filtered.sort_values(by=["month"])
    df_filtered = df_filtered.loc[
        :,
        [
            "month",
            "town",
            "flat_type",
            "block",
            "street_name",
            "storey_range",
            "floor_area_sqm",
            "remaining_lease",
            "resale_price",
        ],
    ]

    table = dash_table.DataTable(
        data=df_filtered.to_dict("records"),
        columns=[{"name": i, "id": i} for i in df_filtered.columns],
        sort_action="native",
        sort_mode="multi",
        page_size=25,
    )

    return table


############################### CALLBACKS TAB 3 ###############################


@app.callback(
    Output(component_id="t3-prop-million-flats", component_property="figure"),
    Input(component_id="t3-graph-dropdown", component_property="value"),
    # prevent_initial_call=True,
)
def update_graph(period):
    transacts = (
        resale_df.groupby(by=[pd.Grouper(key="month", freq=period)])
        .size()
        .to_frame()
        .rename(columns={0: "num_trans"})
    )
    transacts_M = (
        resale_df.query("resale_price > 1_000_000")
        .groupby(by=[pd.Grouper(key="month", freq=period)])
        .size()
        .to_frame()
        .rename(columns={0: "transacts_M"})
    )
    mil_df = transacts.merge(
        transacts_M, how="inner", left_index=True, right_index=True
    )
    mil_df["proportion"] = np.round(
        mil_df["transacts_M"] / mil_df["num_trans"] * 100, 2
    )
    mil_df["pct_chg"] = np.round(mil_df["proportion"].pct_change() * 100, 2)
    mil_df = mil_df.reset_index()
    mil_df["year"] = mil_df["month"].dt.year
    if period == "QS":
        mil_df["period_str"] = mil_df["month"].apply(convert_month_to_quarter)
    else:
        mil_df["period_str"] = mil_df["month"].dt.strftime("%b")

    fig = px.line(
        data_frame=mil_df,
        x="month",
        y="proportion",
        template="seaborn",
        custom_data=[
            "year",
            "period_str",
            "num_trans",
            "transacts_M",
            "proportion",
            "pct_chg",
        ],
        labels={"month": "Year", "proportion": "Percentage (%)"},
        title="Percentage (%) of Transacted Million-dollar flats",
    )

    hover_template = """
    Period: <b>%{customdata[0]} %{customdata[1]}</b><br>Percentage : <b>%{customdata[4]:.2f}%</b><br><br>
    No. transacted flats: <b>%{customdata[2]:,}</b><br>
    No. transacted M-dollar flats: <b>%{customdata[3]:,}</b>
    """

    fig.update_traces(hovertemplate=hover_template)
    fig.update_layout(
        margin=dict(t=40, b=0, l=0, r=0), title=dict(y=0.96, yref="container")
    )

    return fig


@app.callback(
    Output(component_id="t3-million-flats-mapbox", component_property="figure"),
    Input(component_id="t3-period-dropdown", component_property="value"),
    State(component_id="t3-million-flats-mapbox", component_property="figure"),
)
def update_mapbox_graph(x_months, fig_state):

    # Filter by x_months
    today = dt.date.today()
    past_x_month = today.replace(day=1) - relativedelta(months=x_months)
    million_df = resale_df.query("month >= @past_x_month & resale_price > 1_000_000")

    df_grouped = million_df.groupby(
        [
            "blk_no",
            "road_name",
            "postal",
            "latitude",
            "longitude",
            "planning_area_ura",
            "flat_type",
            "closest_mrt_station",
            "distance_to_mrt_meters",
            "transport_type",
            "line_color",
            "closest_pri_school",
            "distance_to_pri_school_meters",
            "remaining_lease",
        ]
    )[["resale_price", "price_per_sqft"]].agg(
        {"resale_price": ["count", "mean"], "price_per_sqft": ["mean"]}
    )
    df_grouped = df_grouped.droplevel(level=0, axis=1)
    df_grouped.columns = ["num_trans", "avg_resale_price", "avg_price_psf"]
    df_grouped = df_grouped.reset_index()

    fig = px.scatter_mapbox(
        data_frame=df_grouped,
        lat="latitude",
        lon="longitude",
        color="avg_resale_price",
        color_continuous_scale="Reds",
        custom_data=[
            "blk_no",
            "road_name",  # 1
            "postal",
            "avg_resale_price",
            "planning_area_ura",
            # "flat_type",  # 5
            "num_trans",
            "avg_price_psf",
            "remaining_lease",
            "closest_mrt_station",
            "distance_to_mrt_meters",  # 10
            "transport_type",
            "closest_pri_school",
            "distance_to_pri_school_meters",  # 13
        ],
        height=800,
    )

    hover_template = """
    <b>%{customdata[5]} Sold </b><br>
    Town: %{customdata[4]}<br>
    Address: Blk %{customdata[0]} %{customdata[1]}, %{customdata[2]}<br>
    Avg. Resale Price: SGD %{customdata[3]:,.0f}<br>
    Avg. Price psf: SGD %{customdata[6]:,.0f}<br>
    Remaining Lease: %{customdata[7]} years<br>
    Closest MRT Station: %{customdata[8]} %{customdata[10]}; %{customdata[9]:,.0f} m<br>
    Closest Pri. School: %{customdata[11]}; %{customdata[12]:,.0f} m
    """

    fig.update_traces(hovertemplate=hover_template)

    if fig_state == {}:
        center_lat = 1.35
        center_lon = 103.8
        zoom = 11
    else:
        center_lat = fig_state["layout"]["mapbox"]["center"]["lat"]
        center_lon = fig_state["layout"]["mapbox"]["center"]["lon"]
        zoom = fig_state["layout"]["mapbox"]["zoom"]

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_accesstoken=os.environ.get("MAPBOX_ACCESS_TOKEN"),
        mapbox_center_lat=center_lat,
        mapbox_center_lon=center_lon,
        mapbox_zoom=zoom,
        showlegend=True,
        clickmode="event",
    )

    return fig


if __name__ == "__main__":
    app.run(debug=True)
