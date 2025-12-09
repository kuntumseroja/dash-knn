"""
Example script demonstrating Named Entity Recognition (NER) processing
on sample entity data.

This script shows how to:
1. Extract named entities from company descriptions
2. Analyze entity types (organizations, locations, people, etc.)
3. Use extracted entities to enhance text embeddings
"""

import os
import yaml
import pandas as pd
import spacy
from collections import defaultdict, Counter
import json

# Load configuration
CFG = yaml.safe_load(open('config.yaml'))
ENTITIES = CFG['paths']['entities_csv']

# Load spaCy NER model
print("Loading spaCy NER model...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ Model loaded successfully")
except OSError:
    print("✗ Error: spaCy model 'en_core_web_sm' not found.")
    print("  Install with: python -m spacy download en_core_web_sm")
    exit(1)


def extract_entities(text, nlp_model):
    """
    Extract named entities from text using spaCy NER.
    
    Args:
        text: Input text string
        nlp_model: Loaded spaCy model
        
    Returns:
        Dictionary mapping entity types to lists of entity values
    """
    if not text or pd.isna(text):
        return {}
    
    doc = nlp_model(str(text))
    entities = defaultdict(list)
    
    for ent in doc.ents:
        entities[ent.label_].append(ent.text.strip())
    
    # Remove duplicates while preserving order
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))
    
    return dict(entities)


def analyze_entities(df, nlp_model, sample_size=10):
    """
    Analyze named entities in entity descriptions.
    
    Args:
        df: DataFrame with entity data
        nlp_model: Loaded spaCy model
        sample_size: Number of examples to show in detail
    """
    print(f"\n{'='*60}")
    print("NER Processing Example")
    print(f"{'='*60}\n")
    
    # Process all entities
    all_entities = []
    entity_stats = Counter()
    
    print(f"Processing {len(df)} entities...")
    for idx, row in df.head(100).iterrows():  # Process first 100 for demo
        about_text = str(row['about_text']) if pd.notna(row['about_text']) else ''
        entities = extract_entities(about_text, nlp_model)
        all_entities.append({
            'entity_id': row['entity_id'],
            'name': row['name'],
            'text': about_text,
            'entities': entities
        })
        
        # Count entity types
        for ent_type in entities.keys():
            entity_stats[ent_type] += len(entities[ent_type])
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Entity Type Statistics")
    print(f"{'='*60}")
    print(f"{'Entity Type':<20} {'Count':<10} {'Description'}")
    print("-" * 60)
    
    entity_descriptions = {
        'ORG': 'Organizations (companies, institutions)',
        'GPE': 'Geopolitical entities (cities, countries)',
        'PERSON': 'People names',
        'PRODUCT': 'Products',
        'MONEY': 'Monetary values',
        'DATE': 'Dates',
        'LOC': 'Locations (non-GPE)',
        'NORP': 'Nationalities/Religious/Political groups',
        'FAC': 'Facilities (buildings, airports)',
        'EVENT': 'Events',
    }
    
    for ent_type, count in entity_stats.most_common():
        desc = entity_descriptions.get(ent_type, 'Other')
        print(f"{ent_type:<20} {count:<10} {desc}")
    
    # Show detailed examples
    print(f"\n{'='*60}")
    print(f"Detailed Examples (showing {min(sample_size, len(all_entities))} entities)")
    print(f"{'='*60}\n")
    
    for i, item in enumerate(all_entities[:sample_size]):
        print(f"Example {i+1}:")
        print(f"  Entity ID: {item['entity_id']}")
        print(f"  Name: {item['name']}")
        print(f"  Text: {item['text'][:100]}..." if len(item['text']) > 100 else f"  Text: {item['text']}")
        print(f"  Extracted Entities:")
        
        if item['entities']:
            for ent_type, values in item['entities'].items():
                print(f"    {ent_type}: {', '.join(values[:5])}")  # Show first 5
        else:
            print("    (No entities found)")
        print()
    
    # Show how entities enhance text
    print(f"{'='*60}")
    print("Text Enhancement Example")
    print(f"{'='*60}\n")
    
    example = all_entities[0]
    print("Original text:")
    print(f"  {example['text']}")
    print()
    
    if example['entities']:
        enhanced_parts = []
        if 'ORG' in example['entities']:
            enhanced_parts.append(f"Organizations: {', '.join(example['entities']['ORG'][:3])}")
        if 'GPE' in example['entities']:
            enhanced_parts.append(f"Locations: {', '.join(example['entities']['GPE'][:3])}")
        
        enhanced_text = example['text'] + " | " + " | ".join(enhanced_parts)
        print("Enhanced text (with NER):")
        print(f"  {enhanced_text}")
    else:
        print("  (No entities to enhance)")
    
    return all_entities


def save_ner_results(all_entities, output_path='data/processed/ner_example_results.json'):
    """Save NER results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to serializable format
    results = []
    for item in all_entities:
        results.append({
            'entity_id': int(item['entity_id']),
            'name': item['name'],
            'text': item['text'],
            'entities': item['entities']
        })
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")


def main():
    """Main function to run NER example."""
    # Load data
    print(f"Loading data from {ENTITIES}...")
    df = pd.read_csv(ENTITIES)
    print(f"✓ Loaded {len(df)} entities\n")
    
    # Analyze entities
    all_entities = analyze_entities(df, nlp, sample_size=10)
    
    # Save results
    save_ner_results(all_entities)
    
    print("NER processing example completed!")


if __name__ == '__main__':
    main()

