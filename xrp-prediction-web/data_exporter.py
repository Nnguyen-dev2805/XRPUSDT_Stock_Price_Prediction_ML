"""
Data Export Script for XRP Price Prediction
Exports predictions from Jupyter notebooks to CSV for Spring Boot integration
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class PredictionDataExporter:
    """
    Exports prediction data from the model to CSV format compatible with Spring Boot
    """

    def __init__(self, output_dir='./data/exports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_predictions(self, cleaned_data, df_clean=None, predictions_dict=None):
        """
        Export predictions to CSV

        Args:
            cleaned_data: DataFrame with historical data and predictions
            df_clean: Additional clean data DataFrame
            predictions_dict: Dictionary with predictions for different horizons
        """

        # Merge all predictions into a single CSV
        export_data = cleaned_data[[
            'Date',
            'Open',
            'High',
            'Low',
            'Price',  # Close price
            'Vol'     # Volume
        ]].copy()

        export_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

        # Add predictions if available
        if 'RF_Pred_Tomorrow' in cleaned_data.columns:
            export_data['predicted_price_1d'] = cleaned_data['RF_Pred_Tomorrow']

        # Add multi-horizon predictions if available
        if 'Target_Price_3D' in cleaned_data.columns:
            export_data['predicted_price_3d'] = cleaned_data['Target_Price_3D']
            export_data['predicted_price_5d'] = cleaned_data['Target_Price_5D']
            export_data['predicted_price_7d'] = cleaned_data['Target_Price_7D']

        # Export to CSV
        filename = os.path.join(self.output_dir, 'xrp_predictions.csv')
        export_data.to_csv(filename, index=False)
        print(f"✓ Exported {len(export_data)} predictions to {filename}")

        return export_data

    def export_detailed_analysis(self, predictions_dict):
        """
        Export detailed analysis for each horizon
        """

        results = []

        for horizon, df_pred in predictions_dict.items():
            df_analysis = df_pred[[
                'Date',
                'Actual_Price',
                'Predicted_Price',
                'Error'
            ]].copy()

            df_analysis.columns = ['date', 'actual_price', 'predicted_price', f'error_{horizon}']

            filename = os.path.join(self.output_dir, f'xrp_analysis_{horizon}.csv')
            df_analysis.to_csv(filename, index=False)
            print(f"✓ Exported {len(df_analysis)} records to {filename}")

            results.append(df_analysis)

        return results

    def generate_sql_insert(self, export_data):
        """
        Generate SQL INSERT statements for bulk data import
        """

        sql_file = os.path.join(self.output_dir, 'insert_predictions.sql')

        with open(sql_file, 'w') as f:
            f.write('-- XRP Prediction Data Import\n')
            f.write('-- Auto-generated SQL INSERT statements\n\n')

            for idx, row in export_data.iterrows():
                date = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'NULL'
                open_price = round(row['open'], 4) if pd.notna(row['open']) else 'NULL'
                high = round(row['high'], 4) if pd.notna(row['high']) else 'NULL'
                low = round(row['low'], 4) if pd.notna(row['low']) else 'NULL'
                close = round(row['close'], 4) if pd.notna(row['close']) else 'NULL'
                volume = int(row['volume']) if pd.notna(row['volume']) else 0
                pred_1d = round(row['predicted_price_1d'], 4) if 'predicted_price_1d' in row and pd.notna(row['predicted_price_1d']) else 'NULL'
                pred_3d = round(row['predicted_price_3d'], 4) if 'predicted_price_3d' in row and pd.notna(row['predicted_price_3d']) else 'NULL'
                pred_5d = round(row['predicted_price_5d'], 4) if 'predicted_price_5d' in row and pd.notna(row['predicted_price_5d']) else 'NULL'
                pred_7d = round(row['predicted_price_7d'], 4) if 'predicted_price_7d' in row and pd.notna(row['predicted_price_7d']) else 'NULL'

                sql = f"""
INSERT INTO price_data (date, open, high, low, close, volume, 
    predicted_price_1d, predicted_price_3d, predicted_price_5d, predicted_price_7d, created_at)
VALUES ('{date}', {open_price}, {high}, {low}, {close}, {volume}, 
    {pred_1d}, {pred_3d}, {pred_5d}, {pred_7d}, CURRENT_DATE);
"""
                f.write(sql)

        print(f"✓ Generated SQL INSERT statements in {sql_file}")

    @staticmethod
    def import_csv_to_spring(csv_path):
        """
        Helper method to import CSV data to Spring Boot application
        Returns cURL command to upload data
        """

        curl_command = f"""
curl -X POST http://localhost:8080/api/import \\
  -H "Content-Type: application/json" \\
  -d '{{"file": "{csv_path}"}}'
"""
        return curl_command


# Example usage in your Jupyter notebook:
"""
# After training and getting predictions:

exporter = PredictionDataExporter(output_dir='./data/exports')

# Export main predictions
predictions_csv = exporter.export_predictions(
    cleaned_data=cleaned_data,
    df_clean=df_clean,
    predictions_dict=predictions  # Your predictions dict
)

# Generate SQL
exporter.generate_sql_insert(predictions_csv)

# Export detailed analysis
exporter.export_detailed_analysis(predictions)

print("✓ All data exported successfully!")
print(f"Files are ready in: ./data/exports/")
"""
