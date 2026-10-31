
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from src.analysis import analyse_trend

# Create output folder
os.makedirs("output", exist_ok=True)

# Load dataset
file_path = "data/UWMadisonArb_JNorth_Project_MonarchsAndMilkweed_1996-2020.csv"
df = pd.read_csv(file_path, low_memory=False)

# Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing text values
df["comments"] = df["comments"].fillna("")
df["school"] = df["school"].fillna("Unknown")

# Convert date and remove invalid dates
df["sighting_date"] = pd.to_datetime(
    df["sighting_date"], errors="coerce"
)
df = df.dropna(subset=["sighting_date"])

# Extract year and month
df["year"] = df["sighting_date"].dt.year
df["month"] = df["sighting_date"].dt.month

# Save cleaned dataset
df.to_csv("output/cleaned_monarch_data.zip", index=False)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

print("\nCleaned Dataset Preview:")
print(df.head())

print("\nTotal Rows:", len(df))
print("Total Columns:", len(df.columns))


# SPRING DATA ANALYSIS (MARCH, APRIL, MAY)

spring_df = df[df["month"].isin([3, 4, 5])].copy()

spring_yearly = spring_df.groupby("year").size()

print("\nYearly Spring Sightings:")
print(spring_yearly)


# GRAPH 1: YEARLY SPRING SIGHTINGS

plt.figure(figsize=(10, 5))
plt.plot(
    spring_yearly.index,
    spring_yearly.values,
    marker="o"
)
plt.title("Yearly Spring Monarch Sightings (1996-2020)")
plt.xlabel("Year")
plt.ylabel("Number of Sightings")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/graph1.png", dpi=300)
plt.show()


# GRAPH 2: MEDIAN SPRING SIGHTING DATE

adult_df = spring_df[
    spring_df["species"] == "Monarch Adult Sighted"
].copy()

adult_df["day_of_year"] = adult_df["sighting_date"].dt.dayofyear

median_dates = adult_df.groupby("year")["day_of_year"].median()

print("\nMedian Spring Adult Sighting Day by Year:")
print(median_dates)

plt.figure(figsize=(10, 5))
plt.plot(
    median_dates.index,
    median_dates.values,
    marker="o"
)
plt.title("Median Spring Adult Monarch Sighting Date")
plt.xlabel("Year")
plt.ylabel("Median Day of Year")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/graph2.png", dpi=300)
plt.show()


# GRAPH 3: SIGHTINGS BY LATITUDE REGION

def classify_region(latitude):
    if latitude < 25:
        return "Southern"
    elif latitude < 35:
        return "Lower-Middle"
    elif latitude < 45:
        return "Upper-Middle"
    else:
        return "Northern"


spring_df["region"] = spring_df["latitude"].apply(classify_region)

region_counts = spring_df["region"].value_counts().reindex(
    ["Southern", "Lower-Middle", "Upper-Middle", "Northern"],
    fill_value=0
)

print("\nSpring Sightings by Latitude Region:")
print(region_counts)

plt.figure(figsize=(8, 5))
region_counts.plot(kind="bar")
plt.title("Spring Monarch Sightings by Latitude Region")
plt.xlabel("Latitude Region")
plt.ylabel("Number of Sightings")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("output/graph3.png", dpi=300)
plt.show()


# STATISTICAL ANALYSIS AND FORECAST

analyse_trend(spring_yearly)


# TEMPERATURE ANALYSIS USING NASA POWER

import requests
from scipy import stats

print("\nDownloading temperature data from NASA POWER...")

url = (
    "https://power.larc.nasa.gov/api/temporal/monthly/point"
    "?parameters=T2M&community=AG"
    "&longitude=-96&latitude=39"
    "&start=1996&end=2020&format=JSON"
)

try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    monthly_temps = data["properties"]["parameter"]["T2M"]

    temperature_records = []

    # Select March, April and May temperatures
    for date, temp in monthly_temps.items():
        if (
            len(date) == 6
            and date.isdigit()
            and date[4:6] in ["03", "04", "05"]
            and temp is not None
        ):
            temperature_records.append({
                "year": int(date[:4]),
                "temperature": float(temp)
            })

    temp_df = pd.DataFrame(temperature_records)

    # Calculate average spring temperature for each year
    spring_temp = (
        temp_df.groupby("year")["temperature"]
        .mean()
        .rename("spring_temperature")
    )

    spring_temp.to_csv("output/spring_temperature.csv")

    print("\nAverage Spring Temperature by Year (°C):")
    print(spring_temp)


    # GRAPH 4: SPRING TEMPERATURE TREND

    plt.figure(figsize=(10, 5))
    plt.plot(
        spring_temp.index,
        spring_temp.values,
        marker="o"
    )
    plt.title("Average Spring Temperature (NASA POWER)")
    plt.xlabel("Year")
    plt.ylabel("Average Temperature (°C)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/graph4_temperature.png", dpi=300)
    plt.show()


    # COMPARE TEMPERATURE AND SPRING SIGHTINGS

    comparison = pd.concat(
        [
            spring_yearly.rename("sightings"),
            spring_temp
        ],
        axis=1
    ).dropna()

    if len(comparison) < 3:
        raise ValueError("Not enough matching yearly data for correlation analysis.")


    # CALCULATE CORRELATION

    correlation, p_value = stats.pearsonr(
        comparison["spring_temperature"],
        comparison["sightings"]
    )

    slope, intercept, r_value, regression_p, std_err = stats.linregress(
        comparison["spring_temperature"],
        comparison["sightings"]
    )

    print("\nTemperature and Sightings Comparison:")
    print("Pearson Correlation:", round(correlation, 3))
    print("P-value:", round(p_value, 6))


    # GRAPH 5: TEMPERATURE VS MONARCH SIGHTINGS

    plt.figure(figsize=(10, 6))

    plt.scatter(
        comparison["spring_temperature"],
        comparison["sightings"],
        s=70,
        alpha=0.75
    )

    # Display actual temperature range with 1°C intervals
    min_temp = comparison["spring_temperature"].min()
    max_temp = comparison["spring_temperature"].max()

    lower_limit = int(min_temp) - 1
    upper_limit = int(max_temp) + 2

    plt.xlim(lower_limit, upper_limit)

    plt.gca().xaxis.set_major_locator(
        ticker.MultipleLocator(1)
    )

    plt.title("Spring Temperature vs Monarch Sightings")
    plt.xlabel("Average Spring Temperature (°C) - March to May")
    plt.ylabel("Number of Spring Monarch Sightings")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(
        "output/graph5_temperature_vs_sightings.png",
        dpi=300
    )
    plt.show()


    # EXCEL REPORT: YEARLY DATA AND STATISTICS

    # Prepare yearly data sheet
    excel_data = comparison.copy()
    excel_data.index.name = "Year"
    excel_data = excel_data.reset_index()

    excel_data = excel_data.rename(columns={
        "spring_temperature": "Average Spring Temperature (°C)",
        "sightings": "Spring Monarch Sightings"
    })


    # Prepare statistics sheet
    statistics_data = pd.DataFrame({
        "Statistical Measure": [
            "Number of Years",
            "Mean Spring Temperature (°C)",
            "Minimum Spring Temperature (°C)",
            "Maximum Spring Temperature (°C)",
            "Mean Monarch Sightings",
            "Pearson Correlation Coefficient",
            "Correlation P-value",
            "Regression Slope",
            "Regression Intercept",
            "R-squared",
            "Regression Standard Error"
        ],

        "Calculated Value": [
            len(comparison),
            comparison["spring_temperature"].mean(),
            comparison["spring_temperature"].min(),
            comparison["spring_temperature"].max(),
            comparison["sightings"].mean(),
            correlation,
            p_value,
            slope,
            intercept,
            r_value ** 2,
            std_err
        ]
    })


    # Save both sheets in one Excel workbook
    with pd.ExcelWriter(
        "output/monarch_temperature_statistics.xlsx",
        engine="openpyxl"
    ) as writer:

        excel_data.to_excel(
            writer,
            sheet_name="Yearly Data",
            index=False
        )

        statistics_data.to_excel(
            writer,
            sheet_name="Statistics",
            index=False
        )


    print("\nExcel report created successfully!")
    print("File: output/monarch_temperature_statistics.xlsx")

    print("\nFinal Correlation:", round(correlation, 3))
    print("Final P-value:", round(p_value, 6))


except Exception as error:
    print("\nTemperature analysis failed:", error)
    print("Existing graphs and analysis are kept separate.")


print("\nAll graphs and reports saved in the output folder.")

