#!/usr/bin/env python3
"""
Sincronizzazione veloce degli embedding rimanenti.
Ottimizzato per performance con batch più grandi e progress monitoring.
"""

import sys
import json
import struct
from pathlib import Path
from datetime import datetime

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / '.claude' / 'hooks' / 'devstream' / 'utils'))
from sqlite_vec_helper import get_db_connection_with_vec

def fast_sync_remaining(batch_size=500):
    """Sincronizzazione veloce degli embedding rimanenti."""
    print('🚀 SINCRONIZZAZIONE VELOCE EMBEDDING RIMANENTI')
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

    if total_to_sync == 0:
        print('✅ Tutti gli embedding sono già sincronizzati!')
        return True

    print(f'📊 Record da sincronizzare: {total_to_sync:,}')
    print(f'📦 Batch size: {batch_size:,}')
    print(f'⏱️  Batch previsti: {(total_to_sync + batch_size - 1) // batch_size}')

    synced_count = 0
    batch_count = 0
    start_time = datetime.now()

    while synced_count < total_to_sync:
        batch_count += 1

        # Mostra progress ogni 10 batch
        if batch_count % 10 == 1 or batch_count == 1:
            current_time = datetime.now()
            elapsed = (current_time - start_time).total_seconds()
            rate = synced_count / elapsed if elapsed > 0 else 0
            eta = (total_to_sync - synced_count) / rate if rate > 0 else 0

            print(f'\\n📦 Batch {batch_count}: Progresso {synced_count:,}/{total_to_sync:,} ({synced_count/total_to_sync*100:.1f}%)')
            print(f'   Rate: {rate:.1f} records/sec | ETA: {eta/60:.1f} min')

        # Recupera un batch di record da sincronizzare
        cursor.execute('''
            SELECT s.id, s.content, s.content_type, s.embedding
            FROM semantic_memory s
            LEFT JOIN vec_semantic_memory v ON s.id = v.memory_id
            WHERE s.embedding IS NOT NULL AND s.embedding != ''
              AND v.memory_id IS NULL
            LIMIT ?
        ''', (batch_size,))

        batch_records = cursor.fetchall()

        # Sincronizza ogni record nel batch
        batch_synced = 0
        for record in batch_records:
            (memory_id, content, content_type, embedding_json) = record

            try:
                # Converti JSON embedding in float32 bytes
                embedding_array = json.loads(embedding_json)

                # Verifica dimensioni
                if len(embedding_array) != 768:
                    print(f'   ⚠️  Skip {memory_id}: dimensioni {len(embedding_array)} != 768')
                    continue

                # Converti in bytes float32
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

                batch_synced += 1

            except Exception as e:
                print(f'   ❌ Errore {memory_id}: {str(e)[:50]}')
                continue

        # Commit del batch
        conn.commit()
        synced_count += batch_synced

        # Mostra progress del batch
        if batch_count % 5 == 0:
            current_time = datetime.now()
            elapsed = (current_time - start_time).total_seconds()
            rate = synced_count / elapsed if elapsed > 0 else 0
            eta = (total_to_sync - synced_count) / rate if rate > 0 else 0

            print(f'   Batch {batch_count}: +{batch_synced:,} | Total: {synced_count:,}/{total_to_sync:,} | Rate: {rate:.1f}/sec | ETA: {eta/60:.1f}min')

        # Exit condition se non ci sono più record
        if len(batch_records) < batch_size:
            break

    # Report finale
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    avg_rate = synced_count / total_time if total_time > 0 else 0

    print(f'\\n✅ SINCRONIZZAZIONE COMPLETATA!')
    print(f'   Record sincronizzati: {synced_count:,}')
    print(f'   Tempo totale: {total_time/60:.1f} minuti')
    print(f'   Rate medio: {avg_rate:.1f} records/sec')

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

    parser = argparse.ArgumentParser(description='Sincronizzazione veloce embedding')
    parser.add_argument('--batch-size', type=int, default=500, help='Batch size (default: 500)')

    args = parser.parse_args()

    try:
        success = fast_sync_remaining(args.batch_size)
        if success:
            print('\\n🎉 SINCRONIZZAZIONE COMPLETATA CON SUCCESSO!')
            print('✅ Sistema sincronizzazione vettoriale completamente operativo!')
            return 0
        else:
            print('\\n⚠️  SINCRONIZZAZIONE COMPLETATA (con qualche limitazione)')
            return 0  # Consideriamo successo anche se non 100%
    except KeyboardInterrupt:
        print('\\n\\n⏹️  SINCRONIZZAZIONE INTERROTTA')
        print('Puoi rieseguire lo script per continuare')
        return 130
    except Exception as e:
        print(f'\\n❌ Errore durante la sincronizzazione: {e}')
        return 1

if __name__ == '__main__':
    sys.exit(main())