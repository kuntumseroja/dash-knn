# JAPFA Poultry Ecosystem Data Integration

## Overview

Successfully integrated 30 JAPFA poultry industry ecosystem entities into the proximity demo system. The data includes JAPFA Group subsidiaries (feed mills, slaughterhouses, breeding operations) and external partner farmer groups.

## Data Summary

### Entities Added (Entity IDs: 221-250)

- **Anchor Entity**: PT Japfa Comfeed Indonesia Tbk (JAPFA HQ)
- **Subsidiaries (Level 1)**:
  - 1 Breeding operation (Multibreeder Adirama)
  - 1 Processing/Slaughterhouse (Ciomas Adisatwa)
  - 1 Aquaculture operation (Suri Tani Pemuka)
  - 1 Beef/Livestock operation (Santosa Agrindo)
  - 1 Animal Health/Vaccines (Vaksindo)
  - 13 Feed Mill units (across Java, Sumatra, Sulawesi, Kalimantan)
  - 7 Slaughterhouse units (across Java, Bali, Sulawesi, Kalimantan)
- **External Partners (Level 2)**:
  - 2 Contract Farmer Groups (SMEs in Sulawesi)

### Relationships Created

1. **Transactions**: 1,073 transactions spanning 2023-2024
   - Feed mills → Farmer groups (feed supply)
   - Farmer groups → Slaughterhouses (chicken supply)
   - Breeding → Feed mills (chick supply)
   - Slaughterhouses → Processing (processed chicken)
   - HO → Subsidiaries (management services)

2. **Supplier Relationships**: 34 supplier-buyer links
   - Feed mills supply farmer groups
   - Breeding supplies feed mills
   - Farmer groups supply slaughterhouses
   - Slaughterhouses supply processing

3. **Directors**: 75 director entries
   - Common directors shared across Japfa_Group entities
   - 2-3 directors per corporate entity

4. **Financials**: 60 financial records (2023 & 2024)
   - Revenue, margin, utilization, delinquency flags
   - Sized appropriately by entity role and segment

5. **Training Labels**: 27 positive examples
   - HO → Subsidiaries relationships
   - Feed mills → Farmer groups
   - Farmer groups → Slaughterhouses
   - Breeding → Feed mills

## Industry Classification

All entities use industry code **10141** (Raising of poultry and birds) to ensure they cluster together in the proximity search.

## Geographic Distribution

Entities are distributed across Indonesia:
- **Java**: Jakarta, Bogor, Purwakarta, Tangerang, Cirebon, Sragen, Gedangan, Surabaya, Sidoarjo
- **Sumatra**: Medan, Padang, Lampung
- **Sulawesi**: Makassar, Parepare, Maros, Pinrang
- **Kalimantan**: Banjarmasin, Bati-bati
- **Bali**: Tabanan

## Training vs Inference Data Recommendation

### ✅ **RECOMMENDED: Training Data**

This data should be used as **TRAINING DATA** because:

1. **Known Relationships**: The ecosystem structure is well-documented with clear supplier-buyer relationships that can serve as positive labels
2. **Label Availability**: 27 positive examples created from known supplier relationships
3. **Ecosystem Learning**: Training the model on this data will help it learn patterns for:
   - Feed mill → Farmer group relationships
   - Integrated poultry supply chains
   - Corporate subsidiary relationships
   - Geographic clustering of industry operations

4. **Model Improvement**: Adding this to training will improve the model's ability to identify similar poultry industry relationships in inference data

### Why NOT Inference Only?

- Inference data is typically unlabeled entities where you want to discover unknown relationships
- This JAPFA data has clear, known relationships that are valuable for training
- The model learns better patterns when trained on diverse industry examples

### Usage Instructions

1. **Data is already integrated** into `data/raw/` (appended to existing files)

2. **Retrain the model** with the new data:
   ```bash
   make featurize  # Rebuild embeddings with new entities
   make index      # Rebuild ANN index
   make train      # Retrain reranker with new labels
   make eval       # Evaluate performance (optional)
   ```

3. **Use in the app**:
   ```bash
   make app
   ```
   - Search for "Japfa" or entity ID 221 (JAPFA HQ)
   - Explore relationships within the poultry ecosystem
   - Find related entities in the same industry

## Data Quality Notes

- All entity IDs start from 221 (existing data goes up to 220)
- Coordinates are estimated based on city/district names
- Financial metrics are realistic but synthetic (based on entity role and size)
- Transaction amounts reflect typical poultry industry patterns
- Supplier relationships follow actual business flows (feed → farm → slaughter → process)

## Next Steps

1. ✅ Data transformation complete
2. ⏭️ Review generated data in `data/raw/` files
3. ⏭️ Run `make featurize` to rebuild embeddings
4. ⏭️ Run `make index` to rebuild ANN index  
5. ⏭️ Run `make train` to retrain reranker with new labels
6. ⏭️ Test in Streamlit app (`make app`)

## File Locations

- **Transformation script**: `transform_japfa_data.py` (can be deleted after use)
- **Integrated data**: `data/raw/` (entities.csv, transactions.csv, suppliers.csv, directors.csv, financials.csv, labels_links.csv)

## Anchor Account

**JAPFA (Entity ID: 221)** - PT Japfa Comfeed Indonesia Tbk
- Role: Anchor Corporate
- Industry: Poultry & Feed (10141)
- Location: Jakarta Selatan, DKI Jakarta
- Relationships: Connected to all 29 subsidiary/partner entities

