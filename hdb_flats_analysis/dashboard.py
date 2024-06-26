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

resale_file_loc = "../datasets/resale_hdb_price_coords_mrt_25jun.csv"
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

school_file_loc = "schools_for_plotly.csv"
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
        return row["month"].dt.strftime("%Y")
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
        return row["month"].dt.strftime("%b %Y")


def pvif(interest, years):
    return (1 - (1 / (1 + interest) ** years)) / interest


def leasehold_pvif(years):
    return np.round(pvif(0.035, years) / pvif(0.035, 999), 3)


#################################### TAB 1 #####################################


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
    dbc.CardBody(
        [
            html.P("This is tab 1!", className="card-text"),
            dbc.Button("Click here", color="success"),
        ]
    ),
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
