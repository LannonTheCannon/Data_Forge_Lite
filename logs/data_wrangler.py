# Disclaimer: This function was generated by AI. Please review before using.
# Agent Name: data_wrangling_agent
# Time Created: 2025-04-27 21:09:57

def data_wrangler(data_list):
    import pandas as pd
    import numpy as np
    '''
    Wrangle the data provided in data_list.

    data_list: A list of one or more pandas data frames containing the raw data to be wrangled.
    
    This function calculates the counts of True and False values for the boolean 
    columns 'Transmission_Manual', 'Fuel type_Diesel', and 'Fuel type_Petrol' 
    from the dataset named 'main' (assumed to be the first or single dataframe in data_list).
    
    Returns a pandas DataFrame summarizing these counts in a tabular format.
    '''


    # Ensure input is a list of DataFrames (if not, convert it)
    if not isinstance(data_list, list):
        data_list = [data_list]

    # From instructions, we have dataset 'main'. Assume the main dataset is the first dataframe in the list.
    df = data_list[0]

    # Define the boolean columns of interest
    bool_cols = ['Transmission_Manual', 'Fuel type_Diesel', 'Fuel type_Petrol']

    # Initialize a dictionary to store counts of True and False values
    summary_counts = {
        'Category': [],
        'True_Count': [],
        'False_Count': []
    }

    # Calculate the count of True and False for each boolean column
    for col in bool_cols:
        # Sum of True values - as True evaluates to 1 in sum()
        true_count = df[col].sum()
        # Count of False values - by summing the negation of the boolean column
        false_count = (~df[col]).sum()

        # Append results to dictionary
        summary_counts['Category'].append(col)
        summary_counts['True_Count'].append(true_count)
        summary_counts['False_Count'].append(false_count)

    # Convert the summary dictionary to a DataFrame for clear tabular presentation
    data_wrangled = pd.DataFrame(summary_counts)

    # Return the resulting summary DataFrame
    return data_wrangled