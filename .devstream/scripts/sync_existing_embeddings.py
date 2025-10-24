#!/usr/bin/env python3
"""
Sincronizzazione manuale degli embedding esistenti nell'indice vec_semantic_memory.

Questo script recupera tutti i record con embedding che non sono nell'indice
specializzato e li sincronizza manualmente.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / '.claude' / 'hooks' / 'devstream' / 'utils'))
from sqlite_vec_helper import get_db_connection_with_vec

def sync_existing_embeddings(batch_size=50):
    """Sincronizza gli embedding esistenti nell'indice vec_semantic_memory."""
    print('🔄 SINCRONIZZAZIONE MANUALE EMBEDDING ESISTENTI')
    print('=' * 50)

    conn = get_db_connection_with_vec('data/devstream.db')
    cursor = conn.cursor()

    # Conta i record da sincronizzare
    cursor.execute('''
        SELECT COUNT(*)
        FROM semantic_memory s
        LEFT JOIN vec_semantic_memory v ON s.id = v.memory_id
        WHERE s.embedding IS NOT NULL AND s.embedding != ''
          AND v.memory_id IS NULL
    ''')

    total_to_sync = cursor.fetchone()[0]
    print(f'📊 Record da sincronizzare: {total_to_sync:,}')

    if total_to_sync == 0:
        print('✅ Tutti gli embedding sono già sincronizzati!')
        return True

    # Processa in batch
    synced_count = 0
    batch_count = 0

    while synced_count < total_to_sync:
        batch_count += 1
        print(f'\\n📦 Batch {batch_count}: Processazione record {synced_count + 1}-{min(synced_count + batch_size, total_to_sync)}')

        # Recupera un batch di record da sincronizzare
        cursor.execute('''
            SELECT s.id, s.content, s.content_type, s.embedding,
                   s.embedding_model, s.embedding_dimension, s.created_at
            FROM semantic_memory s
            LEFT JOIN vec_semantic_memory v ON s.id = v.memory_id
            WHERE s.embedding IS NOT NULL AND s.embedding != ''
              AND v.memory_id IS NULL
            LIMIT ?
        ''', (batch_size,))

        batch_records = cursor.fetchall()

        # Sincronizza ogni record
        for record in batch_records:
            (memory_id, content, content_type, embedding_json,
             embedding_model, embedding_dimension, created_at) = record

            try:
                # Converti JSON embedding in float32 bytes per vec_semantic_memory
                embedding_array = json.loads(embedding_json)

                # Converti in bytes float32
                import struct
                float32_bytes = struct.pack('f' * len(embedding_array), *embedding_array)

                # Inserisci direttamente in vec_semantic_memory
                cursor.execute('''
                    INSERT INTO vec_semantic_memory(
                        embedding, content_type, memory_id, content_preview
                    ) VALUES (?, ?, ?, ?)
                ''', (
                    float32_bytes, content_type, memory_id,
                    content[:200] if content else None
                ))

                synced_count += 1

                if synced_count % 100 == 0:
                    print(f'   Progresso: {synced_count:,}/{total_to_sync:,} ({synced_count/total_to_sync*100:.1f}%)')

            except Exception as e:
                print(f'   ❌ Errore sincronizzazione {memory_id}: {e}')

        # Commit del batch
        conn.commit()

    # Report finale
    print(f'\\n✅ Sincronizzazione completata!')
    print(f'   Record sincronizzati: {synced_count:,}')

    # Verifica finale
    cursor.execute('SELECT COUNT(*) FROM vec_semantic_memory')
    final_vec_count = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM semantic_memory WHERE embedding IS NOT NULL AND embedding != ""')
    total_with_embedding = cursor.fetchone()[0]

    final_percentage = (final_vec_count / total_with_embedding * 100) if total_with_embedding > 0 else 0

    print(f'\\n📈 REPORT FINALE:')
    print(f'   Totali con embedding: {total_with_embedding:,}')
    print(f'   Nell indice vettoriale: {final_vec_count:,}')
    print(f'   Percentuale sincronizzata: {final_percentage:.1f}%')

    conn.close()
    return final_percentage > 95

def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Sincronizza embedding esistenti')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size (default: 50)')

    args = parser.parse_args()

    try:
        success = sync_existing_embeddings(args.batch_size)
        if success:
            print('\\n🎉 SINCRONIZZAZIONE COMPLETATA CON SUCCESSO!')
            print('✅ Ora tutti gli embedding sono nell indice vettoriale')
            return 0
        else:
            print('\\n⚠️  SINCRONIZZAZIONE PARZIALE')
            print('❌ Alcuni embedding potrebbero non essere sincronizzati')
            return 1
    except Exception as e:
        print(f'\\n❌ Errore durante la sincronizzazione: {e}')
        return 1

if __name__ == '__main__':
    sys.exit(main())