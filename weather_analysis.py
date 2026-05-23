# ============================================================
#  Weather Data Analysis and Trend Identification
#  Student  : Abarna.L
#  Reg No   : 2117250020003
#  Dept     : Computer Science Engineering
#  Faculty  : Dr. H. Anwar Basha
#  Course   : B.E (CSE) – Python Assignment 4 Mini Project
# ============================================================

# ── 1. Import Libraries ──────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── 2. Create Sample Weather Dataset ────────────────────────
weather_data = {
    'Record_ID'      : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Station_Name'   : ['Chennai', 'Chennai', 'Chennai', 'Chennai',
                        'Delhi',   'Delhi',   'Delhi',   'Delhi',
                        'Mumbai',  'Mumbai',  'Mumbai',  'Mumbai'],
    'Location'       : ['South', 'South', 'South', 'South',
                        'North', 'North', 'North', 'North',
                        'West',  'West',  'West',  'West'],
    'Month'          : ['Jan', 'Apr', 'Jul', 'Oct',
                        'Jan', 'Apr', 'Jul', 'Oct',
                        'Jan', 'Apr', 'Jul', 'Oct'],
    'Temperature_C'  : [24, 32, 30, 28,
                        10, 28, 35, 22,
                        22, 30, 29, 27],
    'Rainfall_mm'    : [5,  20, 120, 80,
                        10, 15, 200, 30,
                        3,  10, 300, 60],
    'Humidity_Pct'   : [70, 65, 85, 80,
                        45, 50, 75, 55,
                        72, 68, 88, 78],
    'Wind_Speed_kmh' : [12, 15, 20, 14,
                        18, 22, 25, 16,
                        10, 13, 18, 11],
}

# ── 3. Create DataFrame ──────────────────────────────────────
df = pd.DataFrame(weather_data)

# ── 4. Display Dataset ───────────────────────────────────────
print("\n" + "="*60)
print("           WEATHER DATASET")
print("="*60)
print(df.to_string(index=False))

# ── 5. Check Missing Values ──────────────────────────────────
print("\n" + "="*60)
print("           MISSING VALUES")
print("="*60)
print(df.isnull().sum())

# ── 6. Basic Statistics ──────────────────────────────────────
print("\n" + "="*60)
print("           BASIC WEATHER STATISTICS")
print("="*60)
print(df[['Temperature_C', 'Rainfall_mm',
          'Humidity_Pct', 'Wind_Speed_kmh']].describe())

# ── 7. Monthly Average Temperature ──────────────────────────
monthly_temp = df.groupby('Month')['Temperature_C'].mean()
print("\n" + "="*60)
print("           MONTHLY AVERAGE TEMPERATURE (°C)")
print("="*60)
print(monthly_temp)

# ── 8. Total Rainfall by Location ───────────────────────────
location_rain = df.groupby('Location')['Rainfall_mm'].sum()
print("\n" + "="*60)
print("           TOTAL RAINFALL BY LOCATION (mm)")
print("="*60)
print(location_rain)

# ── 9. Maximum Temperature by Station ───────────────────────
max_temp = df.groupby('Station_Name')['Temperature_C'].max()
print("\n" + "="*60)
print("           MAXIMUM TEMPERATURE BY STATION (°C)")
print("="*60)
print(max_temp)

# ── 10. Pivot Table ──────────────────────────────────────────
pivot = pd.pivot_table(
    df,
    values='Temperature_C',
    index='Station_Name',
    columns='Month',
    aggfunc=np.mean
)
print("\n" + "="*60)
print("           PIVOT TABLE – AVG TEMPERATURE (°C)")
print("="*60)
print(pivot)

# ── 11. NumPy Statistics ─────────────────────────────────────
temp_array = np.array(df['Temperature_C'])
rain_array = np.array(df['Rainfall_mm'])

print("\n" + "="*60)
print("           NUMPY TEMPERATURE STATS")
print("="*60)
print(f"  Mean : {np.mean(temp_array):.2f} °C")
print(f"  Max  : {np.max(temp_array)} °C")
print(f"  Min  : {np.min(temp_array)} °C")
print(f"  Std  : {np.std(temp_array):.2f}")

print("\n" + "="*60)
print("           NUMPY RAINFALL STATS")
print("="*60)
print(f"  Mean : {np.mean(rain_array):.2f} mm")
print(f"  Max  : {np.max(rain_array)} mm")
print(f"  Min  : {np.min(rain_array)} mm")
print(f"  Std  : {np.std(rain_array):.2f}")

# ── 12. Visualization 1 – Avg Temp by Month (Bar) ────────────
month_order = ['Jan', 'Apr', 'Jul', 'Oct']
monthly_temp = monthly_temp.reindex(month_order)

plt.figure(figsize=(7, 4))
bars = plt.bar(monthly_temp.index, monthly_temp.values,
               color=['steelblue', 'orange', 'tomato', 'teal'],
               edgecolor='white', width=0.5)
for bar, val in zip(bars, monthly_temp.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f'{val:.1f}°C', ha='center', va='bottom', fontsize=10)
plt.title('Average Temperature by Month', fontsize=13, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('chart1_avg_temp_by_month.png', dpi=150)
plt.show()

# ── 13. Visualization 2 – Total Rainfall by Location (Bar) ───
plt.figure(figsize=(6, 4))
bars = plt.bar(location_rain.index, location_rain.values,
               color=['#4CAF50', '#FF9800', '#2196F3'],
               edgecolor='white', width=0.5)
for bar, val in zip(bars, location_rain.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 3,
             f'{val} mm', ha='center', va='bottom', fontsize=10)
plt.title('Total Rainfall by Location', fontsize=13, fontweight='bold')
plt.xlabel('Location')
plt.ylabel('Rainfall (mm)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('chart2_rainfall_by_location.png', dpi=150)
plt.show()

# ── 14. Visualization 3 – Temperature Trend by Station (Line) ─
plt.figure(figsize=(8, 4))
colors = {'Chennai': 'tomato', 'Delhi': 'steelblue', 'Mumbai': 'seagreen'}
for station in df['Station_Name'].unique():
    subset = df[df['Station_Name'] == station].set_index('Month').reindex(month_order)
    plt.plot(month_order, subset['Temperature_C'],
             marker='o', linewidth=2, markersize=7,
             label=station, color=colors[station])
plt.title('Temperature Trend by Station', fontsize=13, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('chart3_temp_trend_by_station.png', dpi=150)
plt.show()

# ── 15. Visualization 4 – Humidity by Station (Bar) ──────────
avg_humidity = df.groupby('Station_Name')['Humidity_Pct'].mean()
plt.figure(figsize=(6, 4))
bars = plt.bar(avg_humidity.index, avg_humidity.values,
               color=['#9C27B0', '#FF5722', '#00BCD4'],
               edgecolor='white', width=0.5)
for bar, val in zip(bars, avg_humidity.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
plt.title('Average Humidity by Station', fontsize=13, fontweight='bold')
plt.xlabel('Station')
plt.ylabel('Humidity (%)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('chart4_humidity_by_station.png', dpi=150)
plt.show()

# ── 16. Visualization 5 – Wind Speed by Station (Bar) ────────
avg_wind = df.groupby('Station_Name')['Wind_Speed_kmh'].mean()
plt.figure(figsize=(6, 4))
bars = plt.bar(avg_wind.index, avg_wind.values,
               color=['#FF9800', '#607D8B', '#8BC34A'],
               edgecolor='white', width=0.5)
for bar, val in zip(bars, avg_wind.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.2,
             f'{val:.1f} km/h', ha='center', va='bottom', fontsize=10)
plt.title('Average Wind Speed by Station', fontsize=13, fontweight='bold')
plt.xlabel('Station')
plt.ylabel('Wind Speed (km/h)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('chart5_wind_speed_by_station.png', dpi=150)
plt.show()

# ── 17. Done ──────────────────────────────────────────────────
print("\n" + "="*60)
print("    WEATHER DATA ANALYSIS COMPLETED SUCCESSFULLY")
print("="*60)
