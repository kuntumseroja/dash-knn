# Named Entity Recognition (NER) Implementation

## Overview

This document describes the Named Entity Recognition (NER) implementation for the Proximity Finder project. NER extracts structured information (organizations, locations, people, etc.) from unstructured text descriptions to enhance text embeddings and improve entity matching.

## Files Modified/Created

### 1. `requirements.txt`
- Added `spacy==3.7.2` for NER processing

### 2. `src/features/build_text.py` (Enhanced)
- Added NER extraction using spaCy
- Extracts entities from `about_text` field
- Appends extracted entities to text before embedding
- Saves extracted entities to `data/processed/ner_entities.json`
- Gracefully handles missing spaCy model (continues without NER)

### 3. `src/features/example_ner.py` (New)
- Standalone example script demonstrating NER processing
- Shows entity type statistics
- Displays detailed examples of extracted entities
- Demonstrates text enhancement with NER
- Saves results to `data/processed/ner_example_results.json`

### 4. `README.md` (Updated)
- Added NER installation instructions
- Documented NER features in feature building section
- Added example script usage

## Installation

1. Install spaCy:
```bash
pip install -r requirements.txt
```

2. Download spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Running NER-Enhanced Text Processing

The standard text processing pipeline now includes NER:

```bash
python src/features/build_text.py
```

This will:
1. Extract named entities from each entity's `about_text`
2. Enhance text with extracted entities (organizations, locations, people)
3. Generate embeddings with enhanced text
4. Save extracted entities to `data/processed/ner_entities.json`

### Running NER Example Script

To see NER in action with detailed examples:

```bash
python src/features/example_ner.py
```

This script:
- Processes sample entities
- Shows statistics about extracted entity types
- Displays detailed examples with extracted entities
- Demonstrates how text is enhanced with NER information

## Entity Types Extracted

The NER model extracts the following entity types:

- **ORG**: Organizations (companies, institutions)
- **GPE**: Geopolitical entities (cities, countries, states)
- **PERSON**: Person names
- **PRODUCT**: Products
- **MONEY**: Monetary values
- **DATE**: Dates
- **LOC**: Locations (non-GPE)
- **NORP**: Nationalities, religious, or political groups
- **FAC**: Facilities (buildings, airports, etc.)
- **EVENT**: Events

## How NER Enhances Text Embeddings

### Before NER:
```
"CV Textiles Abadi Jakarta operates in Other transport support sector; based in Jakarta. Focus on trading and regional supply chains. Industry:52290"
```

### After NER Enhancement:
```
"CV Textiles Abadi Jakarta operates in Other transport support sector; based in Jakarta. Focus on trading and regional supply chains. Industry:52290 Organizations: CV Textiles Abadi Jakarta | Locations: Jakarta"
```

The extracted entities are appended to the text, providing additional context that helps the sentence transformer create more semantically rich embeddings.

## Output Files

### `data/processed/ner_entities.json`
JSON file mapping entity IDs to extracted entities:
```json
{
  "1": {
    "ORG": ["CV Textiles Abadi Jakarta"],
    "GPE": ["Jakarta"]
  },
  "2": {
    "ORG": ["CV Maritim Nusantara Jakarta"],
    "GPE": ["Jakarta"]
  }
}
```

### `data/processed/ner_example_results.json`
Detailed NER results from the example script, including full text and entity mappings.

## Benefits

1. **Improved Semantic Matching**: Entities like company names and locations are explicitly highlighted
2. **Better Context**: Text embeddings capture structured information alongside unstructured text
3. **Transparency**: Extracted entities are saved for analysis and debugging
4. **Flexibility**: System works with or without NER (graceful degradation)

## Performance Considerations

- NER processing adds minimal overhead (~0.1-0.5 seconds per 100 entities)
- spaCy model is loaded once and reused for all entities
- Processing is done in batches for efficiency
- If spaCy model is unavailable, processing continues without NER

## Future Enhancements

Potential improvements:
- Custom entity types for domain-specific entities (e.g., industry codes, business types)
- Entity linking to knowledge bases
- Multi-language support (Indonesian, English)
- Custom spaCy model fine-tuned on business entity data

