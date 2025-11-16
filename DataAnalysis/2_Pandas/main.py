import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Example 1: Create a Pandas Series using NumPy random function
# ------------------------------------------------------------------

# A Series is a one-dimensional array-like object in Pandas.
series = pd.Series(np.random.random(10))
print(series)
print(series.dtype)  # Data type of elements
print(series.shape)  # Shape (number of elements)

# ------------------------------------------------------------------
# Example 2: Creating a Series from a Python dictionary
# ------------------------------------------------------------------

data = {
    "Name": "Naveed",
    "Age": 18,
    "Qualification": "FSc Pre-Engineering",
    "college":"GBHSS Samaro",
    "city":"Samaro"
    

}
d = pd.Series(data)
print(d)

# ------------------------------------------------------------------
# Example 3: Creating a DataFrame using NumPy random values
# ------------------------------------------------------------------

# np.random.rand() generates random float values between 0 and 1
df = pd.DataFrame(np.random.rand(5, 3))  # 5 rows × 3 columns
print(df)

# Change the row index
df.index = np.arange(1, 6)
print(df)

# Change the column names
df.columns = ["a", "b", "c"]
print(df)

# Display DataFrame structure
print(df.columns, df.index)

# ------------------------------------------------------------------
# Example 4: Another way to define indexes and columns
# ------------------------------------------------------------------

df = pd.DataFrame(np.random.rand(5, 3), index=np.arange(1, 6), columns=["a", "b", "c"])
print(df)
print(df.columns, df.index)

# ------------------------------------------------------------------
# Example 5: Creating a DataFrame from a dictionary
# ------------------------------------------------------------------

Data = {
    "names": ["Naveed", "Zaviyar", "Arif"],
    "marks": [100, 87, 99]
}
d = pd.DataFrame(Data)
print(d)

# Export DataFrame to a CSV file
d.to_csv("report.csv", index=False)  # 'index=False' removes the index column from CSV

# Display the first 2 and last 2 rows of the DataFrame
print(d.head(2))  # Returns first 2 rows
print(d.tail(2))  # Returns last 2 rows

# ------------------------------------------------------------------
# Example 6: Custom Indexing in DataFrames (New Example)
# ------------------------------------------------------------------

# Create a dictionary with tech company data
companies = {
    "Company Name": ["Google", "Microsoft", "Apple", "Amazon"],
    "Headquarters": ["California", "Washington", "California", "Seattle"],
    "Employees": [190000, 221000, 164000, 154000],
    "Rating": [4.6, 4.5, 4.7, 4.4]  # Average employee satisfaction rating
}

# Create a DataFrame from the dictionary
d = pd.DataFrame(companies)
print("Original DataFrame:\n", d)

# Assign custom index labels (short abbreviations)
d.index = ["GOOG", "MSFT", "AAPL", "AMZN"]

# Display the updated DataFrame with custom indexes
print("\nDataFrame with Custom Index:\n", d)

# Save the DataFrame to a CSV file
d.to_csv("companies.csv", index=True)

# The describe() function summarizes only numeric columns
print("\nStatistical Summary (Numeric Columns Only):")
print(d.describe())