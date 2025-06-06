# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_visualization_agent
# Time Created: 2025-04-27 19:42:49

def data_visualization(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.graph_objects as go
    import plotly.io as pio





    # Use the provided data_raw DataFrame (Dataset_0)
    df = data_raw.copy()

    # Define distinct colors for each category
    color_map = {
        "Category_Clothing": "#636EFA",
        "Category_Electronics": "#EF553B",
        "Category_Footwear": "#00CC96",
        "Category_Home Appliances": "#AB63FA"
    }

    # Prepare the bar trace for each category (one bar per category)
    bars = []
    for category in df['Category']:
        bars.append(go.Bar(
            name=category,
            x=[category],
            y=df.loc[df['Category'] == category, 'Total Sales'],
            marker_color=color_map.get(category, None),
            text=df.loc[df['Category'] == category, 'Total Sales'].astype(str),
            textposition='outside'
        ))

    # Create the figure with grouped bars (though only one bar per category)
    fig = go.Figure(data=bars)

    # Update layout with titles and legend
    fig.update_layout(
        barmode='group',
        title="Total Sales by Category for Clothing, Electronics, Footwear, and Home Appliances",
        xaxis_title="Product Category",
        yaxis_title="Total Sales",
        legend_title="Category",
        yaxis=dict(showgrid=True),
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        margin=dict(t=100)
    )

    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)

    return fig_dict