#!/usr/bin/env python3
"""
Transform JAPFA poultry ecosystem data into the proximity demo format.
Creates entities, transactions, suppliers, directors, financials, and labels_links CSV files.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Input data
JAPFA_DATA = """entity_id,legal_name,ecosystem_role,anchor_group,anchor_level,segment_code,region_province,region_district,country,source_note
ENT_JAPFA_HO,PT Japfa Comfeed Indonesia Tbk,Anchor_Corporate,Japfa_Group,0,Corporate,DKI Jakarta,Jakarta Selatan,Indonesia,Anchor – listed poultry & feed company
ENT_MB_ADIRAMA,PT Multibreeder Adirama Indonesia,Breeding,Japfa_Group,1,Corporate,Jawa Barat,Purwakarta,Indonesia,Poultry breeding subsidiary
ENT_CIOMAS,PT Ciomas Adisatwa,Processing/Slaughterhouse,Japfa_Group,1,Corporate,Jawa Barat,Bogor,Indonesia,Chicken processing & slaughterhouse
ENT_SURI_TANI,PT Suri Tani Pemuka,Aquaculture,Japfa_Group,1,Corporate,Jawa Timur,Gresik,Indonesia,Aquaculture & fish/shrimp feed
ENT_SANTOSA_AGRINDO,PT Santosa Agrindo,Beef/Livestock,Japfa_Group,1,Corporate,Jawa Timur,,Indonesia,Beef and livestock business
ENT_VAKSINDO,PT Vaksindo Satwa Nusantara,Animal_Health/Vaccines,Japfa_Group,1,Corporate,Jawa Barat,,Indonesia,Animal vaccine & health
ENT_FEED_MEDAN,[Japfa] Feed Unit Medan,Feed_Mill,Japfa_Group,1,Corporate,Sumatera Utara,Medan,Indonesia,Feed unit Medan
ENT_FEED_PADANG,[Japfa] Feed Unit Padang,Feed_Mill,Japfa_Group,1,Corporate,Sumatera Barat,Padang,Indonesia,Feed unit Padang
ENT_FEED_LAMPUNG,[Japfa] Feed Unit Lampung,Feed_Mill,Japfa_Group,1,Corporate,Lampung,,Indonesia,Feed unit Lampung
ENT_FEED_CIKANDE,[Japfa] Feed Unit Cikande,Feed_Mill,Japfa_Group,1,Corporate,Banten,Serang/Cikande,Indonesia,Feed unit Cikande
ENT_FEED_TANGERANG,[Japfa] Feed Unit Tangerang,Feed_Mill,Japfa_Group,1,Corporate,Banten,Tangerang,Indonesia,Feed unit Tangerang
ENT_FEED_CIREBON,[Japfa] Feed Unit Cirebon,Feed_Mill,Japfa_Group,1,Corporate,Jawa Barat,Cirebon,Indonesia,Feed unit Cirebon
ENT_FEED_SRAGEN,[Japfa] Feed Unit Sragen,Feed_Mill,Japfa_Group,1,Corporate,Jawa Tengah,Sragen,Indonesia,Feed unit Sragen
ENT_FEED_GEDANGAN,[Japfa] Feed Unit Gedangan,Feed_Mill,Japfa_Group,1,Corporate,Jawa Timur,Gedangan,Indonesia,Feed unit Gedangan
ENT_FEED_SURABAYA,[Japfa] Feed Unit Surabaya,Feed_Mill,Japfa_Group,1,Corporate,Jawa Timur,Surabaya,Indonesia,Feed unit Surabaya
ENT_FEED_SIDOARJO,[Japfa] Feed Unit Sidoarjo,Feed_Mill,Japfa_Group,1,Corporate,Jawa Timur,Sidoarjo,Indonesia,Feed unit Sidoarjo
ENT_FEED_MAKASSAR,[Japfa] Feed Unit Makassar,Feed_Mill,Japfa_Group,1,Corporate,Sulawesi Selatan,Makassar,Indonesia,Feed unit Makassar
ENT_FEED_GROBOGAN,[Japfa] Feed Unit Grobogan,Feed_Mill,Japfa_Group,1,Corporate,Jawa Tengah,Grobogan,Indonesia,Feed unit Grobogan
ENT_FEED_PAREPARE,[Japfa] Feed Unit Parepare,Feed_Mill,Japfa_Group,1,Corporate,Sulawesi Selatan,Parepare,Indonesia,Feed unit Parepare
ENT_FEED_BANJARMASIN,[Japfa] Feed Unit Banjarmasin,Feed_Mill,Japfa_Group,1,Corporate,Kalimantan Selatan,Banjarmasin,Indonesia,Feed unit Banjarmasin
ENT_SH_PARUNG,[Japfa] Chicken Slaughterhouse Parung,Slaughterhouse,Japfa_Group,1,Corporate,Jawa Barat,Parung/Bogor,Indonesia,Chicken slaughterhouse unit
ENT_SH_SADANG,[Japfa] Chicken Slaughterhouse Sadang,Slaughterhouse,Japfa_Group,1,Corporate,Jawa Barat,Sadang/Purwakarta,Indonesia,Chicken slaughterhouse unit
ENT_SH_PABELAN,[Japfa] Chicken Slaughterhouse Pabelan,Slaughterhouse,Japfa_Group,1,Corporate,Jawa Tengah,Pabelan/Semarang,Indonesia,Chicken slaughterhouse unit
ENT_SH_BALONGBENDO,[Japfa] Chicken Slaughterhouse Balongbendo,Slaughterhouse,Japfa_Group,1,Corporate,Jawa Timur,Balongbendo/Sidoarjo,Indonesia,Chicken slaughterhouse unit
ENT_SH_TABANAN,[Japfa] Chicken Slaughterhouse Tabanan,Slaughterhouse,Japfa_Group,1,Corporate,Bali,Tabanan,Indonesia,Chicken slaughterhouse unit
ENT_SH_MAROS,[Japfa] Chicken Slaughterhouse Maros,Slaughterhouse,Japfa_Group,1,Corporate,Sulawesi Selatan,Maros,Indonesia,Chicken slaughterhouse unit
ENT_SH_BATI_BATI,[Japfa] Chicken Slaughterhouse Bati-bati,Slaughterhouse,Japfa_Group,1,Corporate,Kalimantan Selatan,Bati-bati,Indonesia,Chicken slaughterhouse unit
ENT_SH_LAMPUNG,[Japfa] Chicken Slaughterhouse Lampung,Slaughterhouse,Japfa_Group,1,Corporate,Lampung,,Indonesia,Chicken slaughterhouse unit
ENT_KTT_BINA_TANI,Kelompok Tani Ternak Bina Tani,Contract_Farmer_Group,External_Partner,2,SME,Sulawesi Selatan,Pinrang,Indonesia,Farmer group linked to feed supply
ENT_KTT_MITRA_TANI,Kelompok Tani Ternak Mitra Tani Bulu Ballea,Contract_Farmer_Group,External_Partner,2,SME,Sulawesi Selatan,,Indonesia,Farmer group mentioned in feed mill network list"""

# Coordinate mapping for Indonesian cities
COORDINATES = {
    'Jakarta Selatan': (-6.2615, 106.8106),
    'Purwakarta': (-6.5547, 107.4430),
    'Bogor': (-6.5944, 106.7892),
    'Gresik': (-7.1538, 112.6561),
    'Medan': (3.5952, 98.6722),
    'Padang': (-0.9471, 100.3542),
    'Lampung': (-5.4291, 105.2630),
    'Serang/Cikande': (-6.1214, 106.1500),
    'Tangerang': (-6.1783, 106.6319),
    'Cirebon': (-6.7320, 108.5523),
    'Sragen': (-7.4265, 111.0174),
    'Gedangan': (-7.3908, 112.7269),
    'Surabaya': (-7.2575, 112.7521),
    'Sidoarjo': (-7.4492, 112.7183),
    'Makassar': (-5.1477, 119.4327),
    'Grobogan': (-7.0313, 110.9252),
    'Parepare': (-4.0142, 119.6248),
    'Banjarmasin': (-3.3192, 114.5918),
    'Parung/Bogor': (-6.4181, 106.7333),
    'Sadang/Purwakarta': (-6.5697, 107.5033),
    'Pabelan/Semarang': (-7.3104, 110.5077),
    'Balongbendo/Sidoarjo': (-7.4167, 112.7167),
    'Tabanan': (-8.5402, 115.1237),
    'Maros': (-5.0573, 119.5689),
    'Bati-bati': (-3.5675, 114.7556),
    'Pinrang': (-3.7874, 119.6488),
}

# Default coordinates for provinces if district not found
PROVINCE_COORDS = {
    'Jawa Timur': (-7.2459, 112.7378),
    'Jawa Barat': (-6.9175, 107.6191),
    'Sulawesi Selatan': (-5.1477, 119.4327),
}

# Industry code for poultry/agriculture
POULTRY_INDUSTRY_CODE = 10141  # Raising of poultry and birds

# Starting entity ID (existing data goes up to 220)
START_ENTITY_ID = 221

# Common directors for Japfa_Group
JAPFA_DIRECTORS = [
    'Tan Yong Hian',
    'Tan Yong Kiat',
    'Rusman Heriawan',
    'Indra Gunawan',
    'Tatang Sutarna',
    'Iman Santoso',
    'Budi Santoso',
]

def get_coordinates(province, district):
    """Get coordinates based on district or province."""
    if district and district in COORDINATES:
        return COORDINATES[district]
    elif province in PROVINCE_COORDS:
        return PROVINCE_COORDS[province]
    else:
        # Default to Jakarta if not found
        return (-6.2088, 106.8456)

def create_entities_df(df_raw):
    """Create entities.csv"""
    entities = []
    entity_id_mapping = {}  # Map original entity_id to new numeric ID
    
    current_id = START_ENTITY_ID
    for _, row in df_raw.iterrows():
        lat, lon = get_coordinates(row['region_province'], row['region_district'])
        
        # Determine is_customer: 1 for anchor/corporate, 0 for SME/External partners
        is_customer = 1 if row['anchor_group'] == 'Japfa_Group' else 0
        
        # Create about_text
        about_text = f"{row['legal_name']} operates in {row['ecosystem_role']} sector; based in {row['region_province']}"
        if row['region_district']:
            about_text += f", {row['region_district']}"
        about_text += f". {row['source_note']}"
        if row['anchor_group'] == 'Japfa_Group':
            about_text += f" Part of {row['anchor_group']} ecosystem."
        
        entities.append({
            'entity_id': current_id,
            'name': row['legal_name'],
            'industry_code': POULTRY_INDUSTRY_CODE,
            'about_text': about_text,
            'lat': lat,
            'lon': lon,
            'is_customer': is_customer,
        })
        
        entity_id_mapping[row['entity_id']] = current_id
        current_id += 1
    
    return pd.DataFrame(entities), entity_id_mapping

def create_transactions_df(df_raw, entity_id_mapping):
    """Create transactions.csv with realistic patterns."""
    transactions = []
    
    # Get entity mappings
    japfa_ho_id = entity_id_mapping.get('ENT_JAPFA_HO')
    feed_mills = [entity_id_mapping[k] for k in entity_id_mapping.keys() if k.startswith('ENT_FEED_')]
    slaughterhouses = [entity_id_mapping[k] for k in entity_id_mapping.keys() if k.startswith('ENT_SH_')]
    breeding_id = entity_id_mapping.get('ENT_MB_ADIRAMA')
    processing_id = entity_id_mapping.get('ENT_CIOMAS')
    farmer_groups = [entity_id_mapping[k] for k in entity_id_mapping.keys() if k.startswith('ENT_KTT_')]
    
    # Transaction patterns:
    # 1. Feed mills supply feed to farmer groups (raw_materials)
    # 2. Farmer groups supply chickens to slaughterhouses (raw_materials)
    # 3. Slaughterhouses supply processed chicken to processing (raw_materials)
    # 4. HO coordinates with subsidiaries (services)
    # 5. Feed mills buy from breeding (raw_materials)
    
    months = []
    start_date = datetime(2023, 1, 1)
    for year in [2023, 2024]:
        for month in range(1, 13):
            months.append(f"{year}-{month:02d}")
    
    random.seed(42)  # For reproducibility
    np.random.seed(42)
    
    # Feed mills to farmer groups (monthly feed supply)
    for feed_id in feed_mills:
        for farmer_id in farmer_groups:
            for month in months:
                if random.random() < 0.8:  # 80% of months have transactions
                    transactions.append({
                        'src_entity_id': feed_id,
                        'dst_entity_id': farmer_id,
                        'yyyymm': month,
                        'amount': np.random.uniform(50000, 500000),
                        'category': 'raw_materials',
                    })
    
    # Farmer groups to slaughterhouses (chicken supply)
    for farmer_id in farmer_groups:
        for sh_id in slaughterhouses:
            for month in months:
                if random.random() < 0.6:  # 60% of months
                    transactions.append({
                        'src_entity_id': farmer_id,
                        'dst_entity_id': sh_id,
                        'yyyymm': month,
                        'amount': np.random.uniform(30000, 300000),
                        'category': 'raw_materials',
                    })
    
    # Breeding to feed mills (chick supply)
    if breeding_id:
        for feed_id in feed_mills[:5]:  # Top 5 feed mills
            for month in months:
                if random.random() < 0.7:
                    transactions.append({
                        'src_entity_id': breeding_id,
                        'dst_entity_id': feed_id,
                        'yyyymm': month,
                        'amount': np.random.uniform(20000, 200000),
                        'category': 'raw_materials',
                    })
    
    # Slaughterhouses to processing (processed chicken)
    if processing_id:
        for sh_id in slaughterhouses[:3]:  # Top 3 slaughterhouses
            for month in months:
                if random.random() < 0.8:
                    transactions.append({
                        'src_entity_id': sh_id,
                        'dst_entity_id': processing_id,
                        'yyyymm': month,
                        'amount': np.random.uniform(40000, 400000),
                        'category': 'raw_materials',
                    })
    
    # HO to subsidiaries (services/management)
    if japfa_ho_id:
        all_subsidiaries = [entity_id_mapping[k] for k in entity_id_mapping.keys() 
                           if k != 'ENT_JAPFA_HO' and entity_id_mapping[k] != japfa_ho_id]
        for sub_id in all_subsidiaries[:15]:  # Top 15 subsidiaries
            for month in months:
                if random.random() < 0.5:
                    transactions.append({
                        'src_entity_id': japfa_ho_id,
                        'dst_entity_id': sub_id,
                        'yyyymm': month,
                        'amount': np.random.uniform(10000, 100000),
                        'category': 'services',
                    })
    
    return pd.DataFrame(transactions)

def create_suppliers_df(df_raw, entity_id_mapping):
    """Create suppliers.csv based on ecosystem relationships."""
    suppliers = []
    
    # Feed mills are suppliers to farmer groups
    feed_mills = [entity_id_mapping[k] for k in entity_id_mapping.keys() if k.startswith('ENT_FEED_')]
    farmer_groups = [entity_id_mapping[k] for k in entity_id_mapping.keys() if k.startswith('ENT_KTT_')]
    
    for farmer_id in farmer_groups:
        # Each farmer group has 2-3 feed mill suppliers
        selected_feeds = random.sample(feed_mills, min(3, len(feed_mills)))
        for feed_id in selected_feeds:
            suppliers.append({
                'buyer_entity_id': farmer_id,
                'supplier_entity_id': feed_id,
            })
    
    # Breeding supplies to feed mills
    breeding_id = entity_id_mapping.get('ENT_MB_ADIRAMA')
    if breeding_id:
        for feed_id in feed_mills[:8]:
            suppliers.append({
                'buyer_entity_id': feed_id,
                'supplier_entity_id': breeding_id,
            })
    
    # Farmer groups supply to slaughterhouses
    slaughterhouses = [entity_id_mapping[k] for k in entity_id_mapping.keys() if k.startswith('ENT_SH_')]
    for sh_id in slaughterhouses:
        selected_farmers = random.sample(farmer_groups, min(2, len(farmer_groups)))
        for farmer_id in selected_farmers:
            suppliers.append({
                'buyer_entity_id': sh_id,
                'supplier_entity_id': farmer_id,
            })
    
    # Slaughterhouses supply to processing
    processing_id = entity_id_mapping.get('ENT_CIOMAS')
    if processing_id:
        for sh_id in slaughterhouses[:4]:
            suppliers.append({
                'buyer_entity_id': processing_id,
                'supplier_entity_id': sh_id,
            })
    
    return pd.DataFrame(suppliers)

def create_directors_df(df_raw, entity_id_mapping):
    """Create directors.csv with common directors for Japfa_Group."""
    directors = []
    
    # Japfa_Group entities share directors
    japfa_group_ids = [entity_id_mapping[k] for k in entity_id_mapping.keys() 
                      if 'Japfa_Group' in df_raw[df_raw['entity_id'] == k]['anchor_group'].values]
    
    # Assign 2-3 directors to each Japfa_Group entity
    for entity_id in japfa_group_ids:
        num_directors = random.randint(2, 3)
        selected_directors = random.sample(JAPFA_DIRECTORS, num_directors)
        for director in selected_directors:
            directors.append({
                'entity_id': entity_id,
                'person_name': director,
            })
    
    return pd.DataFrame(directors)

def create_financials_df(df_raw, entity_id_mapping):
    """Create financials.csv with realistic financial metrics."""
    financials = []
    
    for _, row in df_raw.iterrows():
        entity_id = entity_id_mapping[row['entity_id']]
        
        # Base revenue depends on role and segment
        if row['anchor_level'] == 0:  # HO
            base_revenue = 50000000
        elif row['ecosystem_role'] == 'Feed_Mill':
            base_revenue = np.random.uniform(5000000, 15000000)
        elif row['ecosystem_role'] == 'Slaughterhouse':
            base_revenue = np.random.uniform(3000000, 10000000)
        elif row['ecosystem_role'] == 'Processing/Slaughterhouse':
            base_revenue = np.random.uniform(8000000, 20000000)
        elif row['segment_code'] == 'SME':
            base_revenue = np.random.uniform(500000, 2000000)
        else:
            base_revenue = np.random.uniform(2000000, 8000000)
        
        # Financials for 2023 and 2024
        for year in [2023, 2024]:
            # Revenue grows slightly year over year
            growth = 1.05 if year == 2024 else 1.0
            revenue = base_revenue * growth * np.random.uniform(0.9, 1.1)
            
            # Margin: higher for processing, lower for farmers
            if 'Processing' in row['ecosystem_role']:
                margin = np.random.uniform(0.15, 0.25)
            elif row['segment_code'] == 'SME':
                margin = np.random.uniform(0.05, 0.12)
            else:
                margin = np.random.uniform(0.10, 0.18)
            
            # Utilization: high for established entities
            if row['anchor_level'] <= 1:
                utilization = np.random.uniform(0.70, 0.90)
            else:
                utilization = np.random.uniform(0.50, 0.75)
            
            financials.append({
                'entity_id': entity_id,
                'year': year,
                'revenue': round(revenue, 2),
                'margin': round(margin, 3),
                'utilization': round(utilization, 3),
                'delinq_flag': 0,
            })
    
    return pd.DataFrame(financials)

def create_labels_links_df(df_raw, entity_id_mapping):
    """Create labels_links.csv for training (positive examples)."""
    labels = []
    
    # All Japfa_Group entities at same level are related
    japfa_level1_ids = [entity_id_mapping[k] for k in entity_id_mapping.keys() 
                       if df_raw[df_raw['entity_id'] == k]['anchor_level'].values[0] == 1 
                       and df_raw[df_raw['entity_id'] == k]['anchor_group'].values[0] == 'Japfa_Group']
    
    japfa_ho_id = entity_id_mapping.get('ENT_JAPFA_HO')
    
    # HO to level 1 subsidiaries (supplier_of relationship)
    if japfa_ho_id:
        for sub_id in japfa_level1_ids[:10]:  # Top 10 relationships
            labels.append({
                'left_entity_id': japfa_ho_id,
                'right_entity_id': sub_id,
                'link_type': 'supplier_of',
            })
    
    # Feed mills to farmer groups (supplier_of)
    feed_mills = [entity_id_mapping[k] for k in entity_id_mapping.keys() if k.startswith('ENT_FEED_')]
    farmer_groups = [entity_id_mapping[k] for k in entity_id_mapping.keys() if k.startswith('ENT_KTT_')]
    
    for farmer_id in farmer_groups:
        # Link to 1-2 feed mills
        selected_feeds = random.sample(feed_mills, min(2, len(feed_mills)))
        for feed_id in selected_feeds:
            labels.append({
                'left_entity_id': feed_id,
                'right_entity_id': farmer_id,
                'link_type': 'supplier_of',
            })
    
    # Farmer groups to slaughterhouses (supplier_of)
    slaughterhouses = [entity_id_mapping[k] for k in entity_id_mapping.keys() if k.startswith('ENT_SH_')]
    for sh_id in slaughterhouses[:4]:
        selected_farmers = random.sample(farmer_groups, min(2, len(farmer_groups)))
        for farmer_id in selected_farmers:
            labels.append({
                'left_entity_id': farmer_id,
                'right_entity_id': sh_id,
                'link_type': 'supplier_of',
            })
    
    # Breeding to feed mills (supplier_of)
    breeding_id = entity_id_mapping.get('ENT_MB_ADIRAMA')
    if breeding_id:
        for feed_id in feed_mills[:5]:
            labels.append({
                'left_entity_id': breeding_id,
                'right_entity_id': feed_id,
                'link_type': 'supplier_of',
            })
    
    return pd.DataFrame(labels)

def main():
    # Parse input data
    from io import StringIO
    df_raw = pd.read_csv(StringIO(JAPFA_DATA))
    
    print(f"Processing {len(df_raw)} entities...")
    
    # Create all dataframes
    entities_df, entity_id_mapping = create_entities_df(df_raw)
    transactions_df = create_transactions_df(df_raw, entity_id_mapping)
    suppliers_df = create_suppliers_df(df_raw, entity_id_mapping)
    directors_df = create_directors_df(df_raw, entity_id_mapping)
    financials_df = create_financials_df(df_raw, entity_id_mapping)
    labels_links_df = create_labels_links_df(df_raw, entity_id_mapping)
    
    # Save to CSV files
    output_dir = 'data/raw'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Append to existing files (or create new)
    print(f"\nAppending {len(entities_df)} entities to entities.csv...")
    existing_entities = pd.read_csv(f'{output_dir}/entities.csv') if os.path.exists(f'{output_dir}/entities.csv') else pd.DataFrame()
    entities_combined = pd.concat([existing_entities, entities_df], ignore_index=True)
    entities_combined.to_csv(f'{output_dir}/entities.csv', index=False)
    
    print(f"Appending {len(transactions_df)} transactions to transactions.csv...")
    existing_transactions = pd.read_csv(f'{output_dir}/transactions.csv') if os.path.exists(f'{output_dir}/transactions.csv') else pd.DataFrame()
    transactions_combined = pd.concat([existing_transactions, transactions_df], ignore_index=True)
    transactions_combined.to_csv(f'{output_dir}/transactions.csv', index=False)
    
    print(f"Appending {len(suppliers_df)} supplier relationships to suppliers.csv...")
    existing_suppliers = pd.read_csv(f'{output_dir}/suppliers.csv') if os.path.exists(f'{output_dir}/suppliers.csv') else pd.DataFrame()
    suppliers_combined = pd.concat([existing_suppliers, suppliers_df], ignore_index=True)
    suppliers_combined.to_csv(f'{output_dir}/suppliers.csv', index=False)
    
    print(f"Appending {len(directors_df)} director entries to directors.csv...")
    existing_directors = pd.read_csv(f'{output_dir}/directors.csv') if os.path.exists(f'{output_dir}/directors.csv') else pd.DataFrame()
    directors_combined = pd.concat([existing_directors, directors_df], ignore_index=True)
    directors_combined.to_csv(f'{output_dir}/directors.csv', index=False)
    
    print(f"Appending {len(financials_df)} financial records to financials.csv...")
    existing_financials = pd.read_csv(f'{output_dir}/financials.csv') if os.path.exists(f'{output_dir}/financials.csv') else pd.DataFrame()
    financials_combined = pd.concat([existing_financials, financials_df], ignore_index=True)
    financials_combined.to_csv(f'{output_dir}/financials.csv', index=False)
    
    print(f"Appending {len(labels_links_df)} label links to labels_links.csv...")
    existing_labels = pd.read_csv(f'{output_dir}/labels_links.csv') if os.path.exists(f'{output_dir}/labels_links.csv') else pd.DataFrame()
    labels_combined = pd.concat([existing_labels, labels_links_df], ignore_index=True)
    labels_combined.to_csv(f'{output_dir}/labels_links.csv', index=False)
    
    print("\n✅ Transformation complete!")
    print(f"\nSummary:")
    print(f"  - Entities: {len(entities_df)} new entities (starting from ID {START_ENTITY_ID})")
    print(f"  - Transactions: {len(transactions_df)} transactions")
    print(f"  - Suppliers: {len(suppliers_df)} relationships")
    print(f"  - Directors: {len(directors_df)} entries")
    print(f"  - Financials: {len(financials_df)} records (2 years × {len(df_raw)} entities)")
    print(f"  - Labels: {len(labels_links_df)} training labels")
    print(f"\nNext steps:")
    print(f"  1. Review the data in data/raw/")
    print(f"  2. Run: make featurize")
    print(f"  3. Run: make index")
    print(f"  4. Run: make train")
    print(f"  5. Run: make app")

if __name__ == '__main__':
    main()

