import plotly.express as px
import plotly.graph_objects as go

def create_line_chart(df, title):
    """Create a line chart using Plotly."""
    fig = px.line(df.transpose(), title=title)
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Value",
        legend_title="Categories",
        height=500
    )
    return fig

def create_bar_chart(df, title):
    """Create a bar chart using Plotly."""
    fig = px.bar(df.transpose(), title=title, barmode='group')
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Value",
        legend_title="Categories",
        height=500
    )
    return fig

def create_area_chart(df, title):
    """Create a stacked area chart using Plotly."""
    fig = px.area(df.transpose(), title=title)
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Value",
        legend_title="Categories",
        height=500
    )
    return fig

def create_heatmap_chart(df, title):
    """Create a heatmap chart for growth rates."""
    fig = px.imshow(df, 
                    title=title,
                    labels=dict(x="Time Period", y="Category", color="Value"))
    return fig