import numpy as np
from scipy import stats

def analyse_trend(yearly_data):
    years = yearly_data.index.to_numpy()
    values = yearly_data.values

    correlation = np.corrcoef(years, values)[0, 1]

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        years, values
    )

    print("\nStatistical Analysis:")
    print("Correlation:", round(correlation, 3))
    print("Average linear change per year:", round(slope, 2))
    print("P-value:", p_value)
    print("Average yearly sightings:", round(np.mean(values), 2))
    print("Standard deviation:", round(np.std(values), 2))

    print("\nSimple Forecast:")

    last_year = int(years[-1])

    for future_year in range(last_year + 1, last_year + 6):
        predicted_value = slope * future_year + intercept
        print(
            f"Estimated records for {future_year}: "
            f"{max(0, round(predicted_value))}"
        )

    return slope, p_value