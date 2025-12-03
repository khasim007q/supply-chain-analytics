import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Create 'data/raw' directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

print("STEP 1: GENERATING RAW SOURCE DATA")
print("="*80)

# ==========================================
# 1. MASTER DATA (The "Dimensions")
# ==========================================

# 1.1 Products
print("Generating Products Master...")
products = pd.DataFrame({
    'product_id': [f'P{str(i).zfill(4)}' for i in range(1, 51)],
    'product_name': [f'Item_{i}' for i in range(1, 51)],
    'category': np.random.choice(['Electronics', 'Furniture', 'Apparel', 'Food', 'Industrial'], 50),
    'unit_cost': np.round(np.random.uniform(10, 500, 50), 2),
    'lead_time_days': np.random.randint(3, 21, 50)
})
products['unit_price'] = np.round(products['unit_cost'] * np.random.uniform(1.3, 2.5, 50), 2)

# 1.2 Suppliers
print("Generating Suppliers Master...")
suppliers = pd.DataFrame({
    'supplier_id': [f'S{str(i).zfill(3)}' for i in range(1, 21)],
    'supplier_name': [f'Supplier_{chr(65+i)}' for i in range(20)],
    'country': np.random.choice(['USA', 'China', 'Germany', 'India', 'Japan'], 20),
    'reliability_score': np.round(np.random.uniform(0.70, 0.99, 20), 2)
})

# 1.3 Warehouses
print("Generating Warehouses Master...")
warehouses = pd.DataFrame({
    'warehouse_id': [f'W{str(i).zfill(2)}' for i in range(1, 11)],
    'location': np.random.choice(['North', 'South', 'East', 'West', 'Central'], 10),
    'capacity': np.random.randint(5000, 20000, 10)
})

# ==========================================
# 2. TRANSACTIONAL DATA (The "Facts")
# ==========================================

# 2.1 Sales/Demand Transactions
print("Generating Sales Transactions (This may take a moment)...")
dates = pd.date_range(start='2023-01-01', end='2025-11-30', freq='D')
n_transactions = 15000

sales_data = pd.DataFrame({
    'transaction_id': [f'TRX{str(i).zfill(6)}' for i in range(1, n_transactions + 1)],
    'date': np.random.choice(dates, n_transactions),
    'product_id': np.random.choice(products['product_id'], n_transactions),
    'warehouse_id': np.random.choice(warehouses['warehouse_id'], n_transactions),
    'quantity_ordered': np.random.randint(1, 50, n_transactions)
})

# Simulate stockouts (fulfillment < ordered)
sales_data['quantity_fulfilled'] = np.where(
    np.random.random(n_transactions) > 0.05, # 95% chance of full fulfillment
    sales_data['quantity_ordered'],
    (sales_data['quantity_ordered'] * 0.5).astype(int) # 5% chance of partial/stockout
)

# 2.2 Inventory Snapshots (IoT Simulation)
print("Generating Inventory Snapshots...")
inventory_list = []
snapshot_dates = pd.date_range(start='2024-01-01', end='2025-11-30', freq='W') # Weekly snapshots

for wid in warehouses['warehouse_id']:
    # Assume each warehouse stocks a subset of products
    stocked_products = np.random.choice(products['product_id'], 20, replace=False)
    for pid in stocked_products:
        for d in snapshot_dates:
            inventory_list.append({
                'date': d,
                'warehouse_id': wid,
                'product_id': pid,
                'current_stock': np.random.randint(0, 1000),
                'temperature': np.random.normal(22, 2) # IoT Sensor data
            })
inventory_data = pd.DataFrame(inventory_list)

# 2.3 Supply Orders
print("Generating Supply Orders...")
n_orders = 3000
supply_orders = pd.DataFrame({
    'order_id': [f'PO{str(i).zfill(5)}' for i in range(1, n_orders + 1)],
    'order_date': np.random.choice(dates, n_orders),
    'supplier_id': np.random.choice(suppliers['supplier_id'], n_orders),
    'product_id': np.random.choice(products['product_id'], n_orders),
    'qty_ordered': np.random.randint(100, 500, n_orders),
    'delivery_days_actual': np.random.randint(5, 30, n_orders) # We will calculate delays later
})

# ==========================================
# 3. SAVE TO RAW CSV (Simulating Source Systems)
# ==========================================
print("Saving raw files to 'data/raw/'...")

products.to_csv('data/raw/products.csv', index=False)
suppliers.to_csv('data/raw/suppliers.csv', index=False)
warehouses.to_csv('data/raw/warehouses.csv', index=False)
sales_data.to_csv('data/raw/sales_transactions.csv', index=False)
inventory_data.to_csv('data/raw/inventory_logs.csv', index=False)
supply_orders.to_csv('data/raw/supply_orders.csv', index=False)

print("\n✓ SUCCESS: Step 1 Complete.")
print(f"Generated {len(sales_data)} sales records, {len(inventory_data)} inventory logs, and {len(supply_orders)} supply orders.")
