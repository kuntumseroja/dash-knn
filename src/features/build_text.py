import os, yaml, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from collections import defaultdict
import json

CFG = yaml.safe_load(open('config.yaml'))
ENTITIES = CFG['paths']['entities_csv']
PROC_DIR = CFG['paths']['processed_dir']

os.makedirs(PROC_DIR, exist_ok=True)

# Load spaCy NER model (download with: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
    print("Continuing without NER extraction...")
    nlp = None

def extract_entities(text, nlp_model):
    """
    Extract named entities from text using spaCy NER.
    Returns a dictionary with entity types and their values.
    """
    if nlp_model is None or not text:
        return {}
    
    doc = nlp_model(text)
    entities = defaultdict(list)
    
    for ent in doc.ents:
        # Group entities by type (ORG, GPE, PERSON, etc.)
        entities[ent.label_].append(ent.text.strip())
    
    # Remove duplicates while preserving order
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))
    
    return dict(entities)

def format_entities_for_text(entities_dict):
    """
    Format extracted entities as a string to append to text for embedding.
    """
    if not entities_dict:
        return ""
    
    parts = []
    # Prioritize organization and location entities
    if 'ORG' in entities_dict:
        parts.append(f" Organizations: {', '.join(entities_dict['ORG'][:3])}")
    if 'GPE' in entities_dict:  # Geopolitical entities (cities, countries)
        parts.append(f" Locations: {', '.join(entities_dict['GPE'][:3])}")
    if 'PERSON' in entities_dict:
        parts.append(f" People: {', '.join(entities_dict['PERSON'][:2])}")
    
    return ' '.join(parts)

def main():
    df = pd.read_csv(ENTITIES)
    
    # Extract NER entities from about_text
    print("Extracting named entities from text...")
    entity_data = []
    enhanced_texts = []
    
    for idx, row in df.iterrows():
        about_text = str(row['about_text']) if pd.notna(row['about_text']) else ''
        
        # Extract entities
        entities = extract_entities(about_text, nlp)
        entity_data.append(entities)
        
        # Build enhanced text with NER information
        base_text = about_text + ' Industry:' + str(row['industry_code'])
        ner_suffix = format_entities_for_text(entities)
        enhanced_text = base_text + ner_suffix
        
        enhanced_texts.append(enhanced_text)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)} entities...")
    
    # Save extracted entities as JSON for reference
    entity_dict = {str(eid): ents for eid, ents in zip(df['entity_id'], entity_data)}
    with open(os.path.join(PROC_DIR, 'ner_entities.json'), 'w') as f:
        json.dump(entity_dict, f, indent=2)
    print(f"Saved extracted entities to {os.path.join(PROC_DIR, 'ner_entities.json')}")
    
    # Generate embeddings with NER-enhanced text
    print("Generating text embeddings...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    emb = model.encode(enhanced_texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    
    np.save(os.path.join(PROC_DIR, 'text_embed.npy'), emb)
    df[['entity_id']].to_parquet(os.path.join(PROC_DIR, 'entity_ids.parquet'), index=False)
    
    print('Saved:', emb.shape, '→', os.path.join(PROC_DIR, 'text_embed.npy'))
    
    # Print statistics about extracted entities
    if nlp:
        all_orgs = sum([len(e.get('ORG', [])) for e in entity_data])
        all_locs = sum([len(e.get('GPE', [])) for e in entity_data])
        all_people = sum([len(e.get('PERSON', [])) for e in entity_data])
        print(f"\nNER Statistics:")
        print(f"  Organizations extracted: {all_orgs}")
        print(f"  Locations extracted: {all_locs}")
        print(f"  People extracted: {all_people}")

if __name__ == '__main__':
    main()
