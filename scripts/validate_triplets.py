import pandas as pd
import re
from pathlib import Path

CSV_PATH = Path("data/processed/triplets/manual_validation_50_triplets.csv")

# ============================================================================
# OPTION 1 : AUTO-NOTATION avec heuristiques simples (très rapide)
# ============================================================================

def auto_score_triplet(row):
    """Notation automatique basée sur des règles simples"""
    
    head = str(row['head']).lower()
    relation = str(row['relation']).lower()
    tail = str(row['tail']).lower()
    source = row['source_system']
    
    # REJETER: Entités génériques mauvaises
    bad_entities = {'model', 'method', 'dataset', 'task', 'metric', 'paper', 'study'}
    if head in bad_entities or tail in bad_entities:
        return 0, 'entity_error', 'Entité générique sans valeur'
    
    # REJETER: Relations verbeux LightRAG (trop de mots)
    if source == 'LightRAG' and len(relation.split(',')) > 2:
        return 0, 'relation_error', 'Relation LightRAG trop verbeux'
    
    #  ACCEPTER: Relations REBEL propres
    if source == 'spaCy_REBEL':
        valid_relations = {
            'instance of', 'subclass of', 'use', 'used in', 
            'part of', 'has part', 'related to', 'facet of'
        }
        if any(rel in relation for rel in valid_relations):
            return 1, '', 'REBEL relation standard'
    
    # ACCEPTER: Relations LightRAG raisonnables
    if source == 'LightRAG':
        if len(relation.split()) <= 3 and relation not in ['model', 'dataset']:
            return 1, '', 'Relation LightRAG acceptable'
    
    # PAR DÉFAUT
    return None, '', 'À vérifier manuellement'

# ============================================================================
# OPTION 2 : NOTATION INTERACTIVE (mode dialogue)
# ============================================================================

def interactive_validation():
    """Mode interactif: pose des questions pour chaque triplet"""
    
    df = pd.read_csv(CSV_PATH)
    # Convertir les colonnes de validation au dtype object pour éviter le FutureWarning
    df['manual_is_correct'] = df['manual_is_correct'].astype('object')
    df['manual_error_type'] = df['manual_error_type'].astype('object')
    df['manual_notes'] = df['manual_notes'].astype('object')
    
    for idx, row in df.iterrows():
        print(f"Triplet #{idx + 1}/50 [{row['source_system']}]")
        
        print(f" Document: {row['doc_id']}")
        print(f" Triplet: {row['head']} → {row['relation']} → {row['tail']}")
        print(f"  Existe exactement dans l'autre système? {row['exists_exactly_in_other_system']}")
        print(f" Même paire d'entités dans l'autre système? {row['same_entity_pair_in_other_system']}")
        
        # Input
        while True:
            answer = input("\n✓ Correct? (1/0/s pour skip/q pour quitter): ").strip().lower()
            
            if answer == 'q':
                df.to_csv(CSV_PATH, index=False)
                print(" Fichier sauvegardé!")
                return
            
            if answer == 's':
                print(" Skipped")
                break
            
            if answer in ['1', '0']:
                df.loc[idx, 'manual_is_correct'] = int(answer)
                
                if answer == '0':
                    error_type = input("  Type d'erreur (entity_error/relation_error/hallucination/too_vague): ").strip()
                    if error_type:
                        df.loc[idx, 'manual_error_type'] = error_type
                
                notes = input("  Notes (optionnel, Entrée pour skip): ").strip()
                if notes:
                    df.loc[idx, 'manual_notes'] = notes
                
                print(" Enregistré!")
                break
        
        # Sauvegarde toutes les 10 lignes
        if (idx + 1) % 10 == 0:
            df.to_csv(CSV_PATH, index=False)
            print(f"\n💾 Auto-save: {idx + 1}/50 validations")
    
    df.to_csv(CSV_PATH, index=False)
    print("\n TERMINÉ! Fichier sauvegardé")

# ============================================================================
# OPTION 3 : AUTO + MANUEL (HYBRIDE)
# ============================================================================

def hybrid_validation():
    """D'abord auto-notation, puis complète manuellement ce qui reste"""
    
    df = pd.read_csv(CSV_PATH)
    # Convertir les colonnes de validation au dtype object pour éviter le FutureWarning
    df['manual_is_correct'] = df['manual_is_correct'].astype('object')
    df['manual_error_type'] = df['manual_error_type'].astype('object')
    df['manual_notes'] = df['manual_notes'].astype('object')
    
    print(" PHASE 1 : AUTO-NOTATION")
    print("="*80)
    
    auto_count = 0
    for idx, row in df.iterrows():
        is_correct, error_type, notes = auto_score_triplet(row)
        
        if is_correct is not None:  # Si on peut décider
            df.loc[idx, 'manual_is_correct'] = is_correct
            df.loc[idx, 'manual_error_type'] = error_type
            df.loc[idx, 'manual_notes'] = notes
            auto_count += 1
    
    print(f"{auto_count}/50 triplets auto-notés")
    
    # Sauvegarder les auto-notations
    df.to_csv(CSV_PATH, index=False)
    
    # Afficher ce qui reste
    remaining = df[df['manual_is_correct'].isna()]
    print(f" {len(remaining)} triplets restants à valider manuellement:")
    print(remaining[['validation_id', 'source_system', 'head', 'relation', 'tail']].to_string())
    
    # Mode manuel pour les restants
    print("\n PHASE 2 : VALIDATION MANUELLE")
    print("="*80)
    for idx in remaining.index:
        row = df.loc[idx]
        print(f"\n#{int(row['validation_id'])}: {row['head']} → {row['relation']} → {row['tail']}")
        
        answer = input("✓ Correct? (1/0): ").strip()
        if answer == '1':
            df.loc[idx, 'manual_is_correct'] = 1
        else:
            df.loc[idx, 'manual_is_correct'] = 0
            error = input("  Erreur (entity/relation/hallucination/vague): ").strip()
            df.loc[idx, 'manual_error_type'] = error
    
    df.to_csv(CSV_PATH, index=False)
    print("\n VALIDATION TERMINÉE!")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("VALIDATION DES TRIPLETS")
    print("1. Mode AUTO (heuristiques simples) - 30 sec")
    print("2. Mode INTERACTIF (questions pour chaque) - 1-2 heures")
    print("3. Mode HYBRIDE (auto + manuel) - 30 min (RECOMMANDÉ) ")
    print()
    
    choice = input("Choix (1/2/3): ").strip()
    
    if choice == '1':
        df = pd.read_csv(CSV_PATH)
        # Convertir les colonnes de validation au dtype object pour éviter le FutureWarning
        df['manual_is_correct'] = df['manual_is_correct'].astype('object')
        df['manual_error_type'] = df['manual_error_type'].astype('object')
        df['manual_notes'] = df['manual_notes'].astype('object')
        for idx, row in df.iterrows():
            is_correct, error_type, notes = auto_score_triplet(row)
            if is_correct is not None:
                df.loc[idx, 'manual_is_correct'] = is_correct
                df.loc[idx, 'manual_error_type'] = error_type
                df.loc[idx, 'manual_notes'] = notes
        df.to_csv(CSV_PATH, index=False)
        print(" Auto-notation complétée!")
    
    elif choice == '2':
        interactive_validation()
    
    elif choice == '3':
        hybrid_validation()