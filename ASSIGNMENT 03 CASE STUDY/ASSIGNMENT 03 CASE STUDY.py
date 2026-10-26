import numpy as np
import pandas as pd;

audit_log = []

def record_audit(step_name, before_rows, after_rows):
    removed = max(before_rows - after_rows, 0)
    audit_log.append({
        'step': step_name,
        'rows_after': after_rows,
        'rows_removed': removed
    })
    print(f"{step_name}: rows_after={after_rows}, rows_removed={removed}")

# 1) Load
df=pd.read_csv('supermarket_sales_case_study.csv');
print('rows:', df.shape[0]); 
print('columns:', df.shape[1]);
print('first 5 rows:'); 
print(df.head()); 
print('dtypes:'); 
print(df.dtypes); 


# 2) Audit
audit = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percent": (df.isnull().mean() * 100).round(2),
})
print(audit[audit.missing_count > 0])
print("exact duplicate rows:", df.duplicated().sum())

invoice_counts = df["Invoice ID"].value_counts()
repeated_invoices = invoice_counts[invoice_counts > 1]
print("\nRows sharing the same Invoice ID:")
print(repeated_invoices)

print("\nDistinct values in Gender:")
print(df["Gender"].unique())

print("\nDistinct values in Payment:")
print(df["Payment"].unique())


# 3) Drop duplicates
before_rows = df.shape[0]
df = df.drop_duplicates()
record_audit('Step 3 - Drop exact duplicates', before_rows, df.shape[0])
print("After cleaning exact duplicate rows:", df.duplicated().sum())

# 4) Resolve conflicting Invoice IDs
# Rule: if the same Invoice ID appears more than once with different Quantity and Total values,
# the later row is the corrected record, so keep the last occurrence and remove the earlier one.
before_rows = df.shape[0]
df = df.drop_duplicates(subset='Invoice ID', keep='last')
record_audit('Step 4 - Resolve conflicting Invoice IDs', before_rows, df.shape[0])
print('after resolving conflicting Invoice IDs:', df.shape)

# 5) Standardise text columns
gender_map = {
    'male': 'Male',
    'MALE': 'Male',
    'female': 'Female',
    'FEMALE': 'Female'
}

payment_map = {
    'cash': 'Cash',
    'credit card': 'Credit card',
    'creditcard': 'Credit card',
    'ewallet': 'Ewallet',
    'e-wallet': 'Ewallet'
}

df['Gender'] = (
    df['Gender']
    .astype(str)
    .str.strip()
    .str.lower()
    .map(gender_map)
    .fillna(df['Gender'].astype(str).str.strip().str.title())
)

df['Payment'] = (
    df['Payment']
    .astype(str)
    .str.strip()
    .str.lower()
    .map(payment_map)
    .fillna(df['Payment'].astype(str).str.strip().str.title())
)

print('\nAfter standardising text columns:')
print(df[['Gender', 'Payment']].head())

# 6) Fix negative quantities
before_rows = df.shape[0]
df['Quantity'] = df['Quantity'].abs()
record_audit('Step 6 - Fix negative quantities', before_rows, df.shape[0])
print('\nAfter fixing negative quantities:')
print(df['Quantity'])

# 7) Parse mixed dates
before_rows = df.shape[0]
parsed_1 = pd.to_datetime(df['Date'], errors='coerce', format='%m/%d/%Y')
parsed_2 = pd.to_datetime(df['Date'], errors='coerce', format='%d-%m-%Y')
df['Date'] = parsed_1.combine_first(parsed_2)
record_audit('Step 7 - Parse mixed dates', before_rows, df.shape[0])
print('\nAfter parsing dates:')
print(df['Date'].head())

# 8) Check Total
# Formula: Total = Quantity × Unit price × 1.05 (5% tax). If the file Total differs by more than 0.50,
# replace it with the formula result.
before_rows = df.shape[0]
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Unit price'] = pd.to_numeric(df['Unit price'], errors='coerce')
df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
expected_total = df['Quantity'] * df['Unit price'] * 1.05
mask = (df['Total'] - expected_total).abs() > 0.50
df.loc[mask, 'Total'] = expected_total[mask]
record_audit('Step 8 - Recalculate Total', before_rows, df.shape[0])
print('\nAfter checking Total:')
print(df[['Quantity', 'Unit price', 'Total']].head())

# 9) Handle remaining missing values
# Fill missing Customer type with 'Unknown', replace missing Ratings with median, and drop rows without dates.
before_rows = df.shape[0]
df['Customer type'] = df['Customer type'].fillna('Unknown')
df['Rating'] = df['Rating'].fillna(df['Rating'].median())
df = df.dropna(subset=['Date'])
record_audit('Step 9 - Fill missing values and drop invalid dates', before_rows, df.shape[0])
print('\nAfter handling missing values:')
print(df[['Customer type', 'Rating', 'Date']].head())

# 10) Audit trail
print('\nAudit trail:')
for entry in audit_log:
    print(entry)


# Answering the brief

# 1. Add time columns
# Create time-based derived columns from the cleaned Date and Time fields.
df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.to_period('Q').astype(str)
df['Day of week'] = df['Date'].dt.day_name()
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
print('\nAfter adding time columns:')
print(df[['Date', 'Time', 'Year', 'Quarter', 'Day of week', 'Hour']].head())



# 2. Bucket the trading day
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour

# Store opening hours are 10:00-21:00, split into:
# Morning = 10:00-13:00 (3 hours), Afternoon = 13:00-17:00 (4 hours), Evening = 17:00-21:00 (4 hours)
bucket_hours = {'Morning': 3, 'Afternoon': 4, 'Evening': 4}

hours = df['Hour'].to_numpy()
df['Trading day bucket'] = np.select(
    [
        (hours >= 10) & (hours < 13),
        (hours >= 13) & (hours < 17),
        (hours >= 17) & (hours < 21),
    ],
    ['Morning', 'Afternoon', 'Evening'],
    default=None
)

bucket_transactions = {
    bucket: np.count_nonzero(df['Trading day bucket'].to_numpy() == bucket)
    for bucket in bucket_hours
}

bucket_summary = pd.DataFrame({
    'transactions': bucket_transactions,
    'hours_in_bucket': pd.Series(bucket_hours),
})
bucket_summary['transactions_per_hour'] = (
    bucket_summary['transactions'] / bucket_summary['hours_in_bucket']
)

print('\nTrading day bucket summary:')
print(bucket_summary)

# 3. Compare against target
# Revenue per branch per quarter, joined to a targets table ($100,000 per branch).
# Variance = Revenue - Target; % of target = Revenue / Target * 100; Hit = Revenue >= Target
TARGET = 100000

quarterly = (
    df.groupby(['Branch', 'Quarter'], as_index=False)['Total']
    .sum()
    .rename(columns={'Total': 'Revenue'})
)

targets = pd.DataFrame({
    'Branch': sorted(df['Branch'].unique()),
    'Target': TARGET,
})

quarterly = quarterly.merge(targets, on='Branch', how='left')
quarterly['Variance'] = quarterly['Revenue'] - quarterly['Target']
quarterly['% of target'] = (quarterly['Revenue'] / quarterly['Target'] * 100).round(2)
quarterly['Hit'] = quarterly['Revenue'] >= quarterly['Target']

print('\nRevenue vs target (branch x quarter):')
print(quarterly.round(2).to_string(index=False))


# 4. Summarise each branch
# Average quarterly revenue, quarters hit out of 8, best and worst quarter,
# and YoY growth % = (2024 revenue / 2023 revenue - 1) * 100.
def summarise_branch(group):
    best = group.loc[group['Revenue'].idxmax()]
    worst = group.loc[group['Revenue'].idxmin()]
    return pd.Series({
        'Avg quarterly revenue': group['Revenue'].mean(),
        'Quarters hit': f"{int(group['Hit'].sum())} / {len(group)}",
        'Best quarter': f"{best['Quarter']} ({best['Revenue']:,.0f})",
        'Worst quarter': f"{worst['Quarter']} ({worst['Revenue']:,.0f})",
    })

branch_summary = quarterly.groupby('Branch').apply(summarise_branch)

yearly = df.pivot_table(index='Branch', columns='Year', values='Total', aggfunc='sum')
branch_summary['Revenue 2023'] = yearly[2023]
branch_summary['Revenue 2024'] = yearly[2024]
branch_summary['YoY growth %'] = ((yearly[2024] / yearly[2023] - 1) * 100).round(2)

print('\nBranch summary:')
print(branch_summary.round(2).to_string())


# 5. Find the busy times
# Average per day = transactions on that weekday / number of those weekdays in the data.
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Count each calendar weekday once (a date with many sales is still one Monday).
days_in_data = (
    df.drop_duplicates('Date')['Day of week']
    .value_counts()
    .reindex(weekday_order)
)

weekday_transactions = df['Day of week'].value_counts().reindex(weekday_order)
weekday_avg = (weekday_transactions / days_in_data).round(2)

print('\nAverage transactions per day, by weekday:')
print(pd.DataFrame({
    'transactions': weekday_transactions,
    'days_in_data': days_in_data,
    'avg_per_day': weekday_avg,
}))

# Same idea for each hour of each weekday.
hourly_counts = pd.crosstab(df['Day of week'], df['Hour']).reindex(weekday_order)
hourly_avg = hourly_counts.div(days_in_data, axis=0).round(2)

print('\nAverage transactions per hour of each weekday:')
print(hourly_avg)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(hourly_avg.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(hourly_avg.columns)))
ax.set_xticklabels([f'{h}:00' for h in hourly_avg.columns])
ax.set_yticks(range(len(hourly_avg.index)))
ax.set_yticklabels(hourly_avg.index)
for i in range(hourly_avg.shape[0]):
    for j in range(hourly_avg.shape[1]):
        ax.text(j, i, f'{hourly_avg.iat[i, j]:.1f}', ha='center', va='center', fontsize=8)
ax.set_title('Average transactions per hour, by weekday')
ax.set_xlabel('Hour')
fig.colorbar(im, ax=ax, label='Avg transactions')
fig.tight_layout()
fig.savefig('busy_times_heatmap.png', dpi=150)
plt.close(fig)
print('\nSaved heatmap: busy_times_heatmap.png')


# 6. Chart, export, summarise
# Each branch's 8 quarters with a line at the $100,000 target.
revenue_by_quarter = quarterly.pivot(index='Quarter', columns='Branch', values='Revenue').sort_index()

fig, ax = plt.subplots(figsize=(10, 5))
for branch in revenue_by_quarter.columns:
    ax.plot(revenue_by_quarter.index, revenue_by_quarter[branch], marker='o', label=f'Branch {branch}')
ax.axhline(TARGET, color='red', linestyle='--', label='Target ($100,000)')
ax.set_title('Quarterly revenue by branch vs target')
ax.set_xlabel('Quarter')
ax.set_ylabel('Revenue ($)')
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('branch_revenue_vs_target.png', dpi=150)
plt.close(fig)
print('Saved chart: branch_revenue_vs_target.png')

# Export the clean data.
df.to_csv('supermarket_sales_clean.csv', index=False)
print('Saved clean data: supermarket_sales_clean.csv', df.shape)

# Five-line summary for management, built from the numbers above.
total_hits = int(quarterly['Hit'].sum())
total_quarters = len(quarterly)
top_branch = branch_summary['Avg quarterly revenue'].idxmax()
low_branch = branch_summary['Avg quarterly revenue'].idxmin()
fastest_growth = branch_summary['YoY growth %'].idxmax()
busiest_day = weekday_avg.idxmax()
busiest_hour = hourly_avg.stack().idxmax()
busiest_bucket = bucket_summary['transactions_per_hour'].idxmax()

summary_lines = [
    f"1. Targets were hit in {total_hits} of {total_quarters} branch-quarters "
    f"({total_hits / total_quarters * 100:.0f}%).",
    f"2. Branch {top_branch} leads with ${branch_summary.loc[top_branch, 'Avg quarterly revenue']:,.0f} "
    f"average per quarter; Branch {low_branch} trails at "
    f"${branch_summary.loc[low_branch, 'Avg quarterly revenue']:,.0f}.",
    f"3. Fastest year-on-year growth: Branch {fastest_growth} "
    f"({branch_summary.loc[fastest_growth, 'YoY growth %']:+.1f}% from 2023 to 2024).",
    f"4. {busiest_day} is the busiest weekday ({weekday_avg.max():.1f} transactions/day); "
    f"peak slot is {busiest_hour[0]} at {busiest_hour[1]}:00.",
    f"5. {busiest_bucket} is the busiest part of the day "
    f"({bucket_summary.loc[busiest_bucket, 'transactions_per_hour']:.0f} transactions per hour) - staff it accordingly.",
]

print('\nSummary for management:')
for line in summary_lines:
    print(line)
