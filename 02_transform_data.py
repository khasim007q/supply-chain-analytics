import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Create 'data/processed' directory
os.makedirs('data/processed', exist_ok=True)

print("STEP 2: DATA TRANSFORMATION")
print("="*80)

# ==========================================
# 1. LOAD RAW DATA (From Step 1)
# ==========================================
print("\n[2.1] Loading raw data files...")

products = pd.read_csv('data/raw/products.csv')
suppliers = pd.read_csv('data/raw/suppliers.csv')
warehouses = pd.read_csv('data/raw/warehouses.csv')
sales = pd.read_csv('data/raw/sales_transactions.csv')
inventory = pd.read_csv('data/raw/inventory_logs.csv')
supply_orders = pd.read_csv('data/raw/supply_orders.csv')

print(f"✓ Loaded {len(sales)} sales records")
print(f"✓ Loaded {len(inventory)} inventory snapshots")
print(f"✓ Loaded {len(supply_orders)} supply orders")

# ==========================================
# 2. DATA QUALITY CHECKS & CLEANING
# ==========================================
print("\n[2.2] Data Quality Checks...")

# Check for nulls
print(f"  Sales nulls: {sales.isnull().sum().sum()}")
print(f"  Inventory nulls: {inventory.isnull().sum().sum()}")
print(f"  Supply orders nulls: {supply_orders.isnull().sum().sum()}")

# Remove any duplicates
sales = sales.drop_duplicates(subset=['transaction_id'])
supply_orders = supply_orders.drop_duplicates(subset=['order_id'])

# Convert dates to datetime
sales['date'] = pd.to_datetime(sales['date'])
inventory['date'] = pd.to_datetime(inventory['date'])
supply_orders['order_date'] = pd.to_datetime(supply_orders['order_date'])

print("✓ Data cleaned and date columns converted")

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
print("\n[2.3] Feature Engineering...")

# 3.1 Sales Data - Add time features
sales['year'] = sales['date'].dt.year
sales['month'] = sales['date'].dt.month
sales['quarter'] = sales['date'].dt.quarter
sales['day_of_week'] = sales['date'].dt.day_name()
sales['week_of_year'] = sales['date'].dt.isocalendar().week

# 3.2 Sales Data - Merge with product info to calculate revenue/profit
sales = sales.merge(products[['product_id', 'unit_price', 'unit_cost', 'category']], 
                    on='product_id', how='left')

sales['revenue'] = sales['quantity_fulfilled'] * sales['unit_price']
sales['cost'] = sales['quantity_fulfilled'] * sales['unit_cost']
sales['profit'] = sales['revenue'] - sales['cost']

# 3.3 Sales Data - Calculate stockout flag
sales['stockout_flag'] = (sales['quantity_fulfilled'] < sales['quantity_ordered']).astype(int)
sales['fulfillment_rate'] = (sales['quantity_fulfilled'] / sales['quantity_ordered']).round(3)

print(f"✓ Added time features and business metrics to sales data")
print(f"  Total Revenue: ${sales['revenue'].sum():,.2f}")
print(f"  Total Profit: ${sales['profit'].sum():,.2f}")
print(f"  Stockout Rate: {sales['stockout_flag'].mean():.1%}")

# 3.4 Inventory Data - Add alert flags
inventory['temp_alert'] = (inventory['temperature'] > 25).astype(int)  # Temperature threshold
inventory['low_stock_alert'] = (inventory['current_stock'] < 100).astype(int)  # Low stock threshold

print(f"✓ Added alert flags to inventory data")
print(f"  Temperature alerts: {inventory['temp_alert'].sum()} instances")
print(f"  Low stock alerts: {inventory['low_stock_alert'].sum()} instances")

# 3.5 Supply Orders - Calculate delivery performance
supply_orders = supply_orders.merge(
    suppliers[['supplier_id', 'supplier_name', 'country', 'reliability_score']], 
    on='supplier_id', how='left'
)

# Calculate expected delivery based on product lead time
supply_orders = supply_orders.merge(
    products[['product_id', 'lead_time_days']], 
    on='product_id', how='left'
)

supply_orders['expected_delivery_days'] = supply_orders['lead_time_days']
supply_orders['delay_days'] = supply_orders['delivery_days_actual'] - supply_orders['expected_delivery_days']
supply_orders['is_delayed'] = (supply_orders['delay_days'] > 0).astype(int)

# Calculate order value
supply_orders = supply_orders.merge(
    products[['product_id', 'unit_cost']], 
    on='product_id', how='left', suffixes=('', '_prod')
)
supply_orders['order_value'] = supply_orders['qty_ordered'] * supply_orders['unit_cost']

print(f"✓ Enhanced supply orders with delivery metrics")
print(f"  Delayed orders: {supply_orders['is_delayed'].sum()} ({supply_orders['is_delayed'].mean():.1%})")
print(f"  Total order value: ${supply_orders['order_value'].sum():,.2f}")

# ==========================================
# 4. CALCULATE KEY AGGREGATIONS
# ==========================================
print("\n[2.4] Calculating aggregated metrics...")

# 4.1 Product Performance Metrics
product_metrics = sales.groupby('product_id').agg({
    'quantity_ordered': ['sum', 'mean', 'std'],
    'quantity_fulfilled': 'sum',
    'revenue': 'sum',
    'profit': 'sum',
    'stockout_flag': 'sum',
    'transaction_id': 'count'
}).round(2)

product_metrics.columns = ['total_demand', 'avg_demand', 'demand_std', 
                           'total_fulfilled', 'total_revenue', 'total_profit', 
                           'stockout_count', 'transaction_count']
product_metrics = product_metrics.reset_index()

# Calculate coefficient of variation (demand variability)
product_metrics['demand_cv'] = (product_metrics['demand_std'] / product_metrics['avg_demand']).round(2)
product_metrics['stockout_rate'] = (product_metrics['stockout_count'] / product_metrics['transaction_count']).round(3)

# Merge with product details
product_metrics = product_metrics.merge(products[['product_id', 'product_name', 'category']], on='product_id')

print(f"✓ Product metrics calculated for {len(product_metrics)} products")

# 4.2 Warehouse Performance
warehouse_metrics = sales.groupby('warehouse_id').agg({
    'quantity_fulfilled': 'sum',
    'revenue': 'sum',
    'stockout_flag': 'sum',
    'transaction_id': 'count'
}).round(2)

warehouse_metrics.columns = ['total_fulfilled', 'total_revenue', 'stockout_count', 'transaction_count']
warehouse_metrics['stockout_rate'] = (warehouse_metrics['stockout_count'] / warehouse_metrics['transaction_count']).round(3)
warehouse_metrics = warehouse_metrics.reset_index()

warehouse_metrics = warehouse_metrics.merge(warehouses, on='warehouse_id')

print(f"✓ Warehouse metrics calculated for {len(warehouse_metrics)} warehouses")

# 4.3 Supplier Performance
supplier_metrics = supply_orders.groupby('supplier_id').agg({
    'order_id': 'count',
    'qty_ordered': 'sum',
    'order_value': 'sum',
    'is_delayed': 'sum',
    'delay_days': 'mean',
    'delivery_days_actual': 'mean'
}).round(2)

supplier_metrics.columns = ['total_orders', 'total_qty', 'total_value', 
                            'delayed_orders', 'avg_delay', 'avg_delivery_time']
supplier_metrics['on_time_rate'] = (1 - (supplier_metrics['delayed_orders'] / supplier_metrics['total_orders'])).round(3)
supplier_metrics = supplier_metrics.reset_index()

supplier_metrics = supplier_metrics.merge(
    suppliers[['supplier_id', 'supplier_name', 'country', 'reliability_score']], 
    on='supplier_id'
)

print(f"✓ Supplier metrics calculated for {len(supplier_metrics)} suppliers")
print(f"  Avg on-time delivery rate: {supplier_metrics['on_time_rate'].mean():.1%}")

# 4.4 Monthly Trend Data
monthly_sales = sales.groupby(['year', 'month', 'category']).agg({
    'quantity_ordered': 'sum',
    'revenue': 'sum',
    'profit': 'sum',
    'stockout_flag': 'sum'
}).reset_index()

monthly_sales.columns = ['year', 'month', 'category', 'total_demand', 'total_revenue', 'total_profit', 'stockouts']

print(f"✓ Monthly trends calculated: {len(monthly_sales)} month-category combinations")

# ==========================================
# 5. SAVE TRANSFORMED DATA
# ==========================================
print("\n[2.5] Saving transformed data to 'data/processed/'...")

# Save enhanced transactional data
sales.to_csv('data/processed/sales_transformed.csv', index=False)
inventory.to_csv('data/processed/inventory_transformed.csv', index=False)
supply_orders.to_csv('data/processed/supply_orders_transformed.csv', index=False)

# Save aggregated metrics
product_metrics.to_csv('data/processed/product_metrics.csv', index=False)
warehouse_metrics.to_csv('data/processed/warehouse_metrics.csv', index=False)
supplier_metrics.to_csv('data/processed/supplier_metrics.csv', index=False)
monthly_sales.to_csv('data/processed/monthly_trends.csv', index=False)

# Save dimension tables (for data warehouse)
products.to_csv('data/processed/dim_products.csv', index=False)
suppliers.to_csv('data/processed/dim_suppliers.csv', index=False)
warehouses.to_csv('data/processed/dim_warehouses.csv', index=False)

print("\n✓ SUCCESS: Step 2 Complete.")
print(f"\nTransformed Files Created:")
print(f"  - sales_transformed.csv ({len(sales)} records)")
print(f"  - inventory_transformed.csv ({len(inventory)} records)")
print(f"  - supply_orders_transformed.csv ({len(supply_orders)} records)")
print(f"  - product_metrics.csv ({len(product_metrics)} records)")
print(f"  - warehouse_metrics.csv ({len(warehouse_metrics)} records)")
print(f"  - supplier_metrics.csv ({len(supplier_metrics)} records)")
print(f"  - monthly_trends.csv ({len(monthly_sales)} records)")
print("\n" + "="*80)
