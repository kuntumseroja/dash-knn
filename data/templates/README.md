# Template Data Files

This directory contains template CSV files that show the expected data format for training and inference.

## Training Data (`training/`)

Use these templates when preparing data to **train the model**. All files are required:

### Required Files:

1. **`entities.csv`** - Core entity information
   - `entity_id` (int): Unique identifier for each entity
   - `name` (string): Company/entity name
   - `industry_code` (int): Industry classification code
   - `about_text` (string): Description text about the entity (used for text embeddings)
   - `lat` (float): Latitude coordinate
   - `lon` (float): Longitude coordinate
   - `is_customer` (int): Binary flag (0 or 1) indicating if entity is a customer

2. **`transactions.csv`** - Transaction history between entities
   - `src_entity_id` (int): Source entity ID
   - `dst_entity_id` (int): Destination entity ID
   - `yyyymm` (string): Date in YYYY-MM format
   - `amount` (float): Transaction amount
   - `category` (string): Transaction category (e.g., "raw_materials", "utilities", "services", "retail_pos", "logistics")

3. **`financials.csv`** - Financial metrics per entity per year
   - `entity_id` (int): Entity identifier
   - `year` (int): Year (e.g., 2023, 2024)
   - `revenue` (float): Annual revenue
   - `margin` (float): Profit margin (0.0 to 1.0)
   - `utilization` (float): Utilization rate (0.0 to 1.0)
   - `delinq_flag` (int): Delinquency flag (0 or 1)

4. **`directors.csv`** - Directors/executives associated with entities
   - `entity_id` (int): Entity identifier
   - `person_name` (string): Director/executive name
   - Note: Multiple rows per entity_id are allowed (one per director)

5. **`suppliers.csv`** - Supplier relationships
   - `buyer_entity_id` (int): Buyer entity ID
   - `supplier_entity_id` (int): Supplier entity ID
   - Note: Represents supplier-buyer relationships for graph embeddings

6. **`labels_links.csv`** - **Training labels** (REQUIRED for training only)
   - `left_entity_id` (int): First entity ID
   - `right_entity_id` (int): Second entity ID
   - `link_type` (string): Relationship type (e.g., "supplier_of", "common_owner", "co_customer")
   - Note: This file defines positive examples for the supervised reranker. The model automatically samples negatives.

## Inference Data (`inference/`)

Use these templates when preparing data for **inference/prediction**. Same structure as training, but **exclude `labels_links.csv`** (not needed for inference).

### Required Files:

1. **`entities.csv`** - Same format as training
2. **`transactions.csv`** - Same format as training
3. **`financials.csv`** - Same format as training
4. **`directors.csv`** - Same format as training
5. **`suppliers.csv`** - Same format as training

### Notes for Inference:

- **No `labels_links.csv`** needed (this is only for training the reranker)
- Include both **query entities** (entities you want to find neighbors for) and **candidate entities** (potential matches) in the same files
- The model will compute proximity scores between all entity pairs

## Data Requirements

### Minimum Data Size:
- At least **2 entities** required (for basic functionality)
- Recommended: **100+ entities** for meaningful results
- More data = better embeddings and more accurate predictions

### Data Quality:
- **entity_id** must be unique and consistent across all files
- **entity_id** values should be integers (or convertable to integers)
- **lat/lon** should be valid coordinates (latitude: -90 to 90, longitude: -180 to 180)
- **about_text** should be descriptive (used for text embeddings)
- **yyyymm** format: "YYYY-MM" (e.g., "2024-03")
- **amount** should be positive numbers
- **margin** and **utilization** should be between 0.0 and 1.0

### Entity ID Consistency:
- All `entity_id` values referenced in `transactions.csv`, `financials.csv`, `directors.csv`, `suppliers.csv`, and `labels_links.csv` must exist in `entities.csv`
- Entity IDs should be sequential integers starting from 1 (or consistent across files)

## Usage

### For Training:
1. Replace template files in `data/templates/training/` with your actual data
2. Copy files to `data/raw/` (or update `config.yaml` paths)
3. Run the training pipeline:
   ```bash
   make featurize
   make index
   make train
   ```

### For Inference:
1. Replace template files in `data/templates/inference/` with your actual data
2. Copy files to `data/raw/` (or update `config.yaml` paths)
3. Ensure trained models exist in `models/`
4. Run the app:
   ```bash
   streamlit run app/app.py
   ```

## Example Workflow

1. **Prepare training data:**
   - Collect entity information → `entities.csv`
   - Collect transaction history → `transactions.csv`
   - Collect financial data → `financials.csv`
   - Collect director information → `directors.csv`
   - Collect supplier relationships → `suppliers.csv`
   - Label known relationships → `labels_links.csv`

2. **Train the model:**
   - Place training files in `data/raw/`
   - Run training pipeline
   - Models saved to `models/`

3. **Prepare inference data:**
   - Collect new entity data (same format, no labels)
   - Place in `data/raw/` (or separate directory)
   - Run inference via Streamlit app

## Tips

- **Text embeddings**: The `about_text` field is important - make it descriptive and informative
- **Transaction categories**: Use consistent category names (e.g., "raw_materials", "utilities", "services")
- **Geographic data**: Accurate lat/lon improves geographic proximity features
- **Director overlap**: Entities sharing directors will have higher proximity scores
- **Supplier relationships**: Include all known supplier-buyer relationships for better graph embeddings

