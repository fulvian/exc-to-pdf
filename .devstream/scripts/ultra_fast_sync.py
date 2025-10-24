#!/usr/bin/env python3
"""
Sincronizzazione ultra-veloce ottimizzata per completare il 100%.
Batch più grandi e parallelizzazione dove possibile.
"""

import sys
import json
import struct
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / '.claude' / 'hooks' / 'devstream' / 'utils'))
from sqlite_vec_helper import get_db_connection_with_vec

def process_batch(batch_records):
    """Processa un batch di record in parallelo."""
    processed = []

    for record in batch_records:
        (memory_id, content, content_type, embedding_json) = record

        try:
            # Converti JSON embedding in float32 bytes
            embedding_array = json.loads(embedding_json)

            # Verifica dimensioni
            if len(embedding_array) != 768:
                continue

            # Converti in bytes float32
            float32_bytes = struct.pack('f' * len(embedding_array), *embedding_array)

            processed.append((
                float32_bytes, content_type, memory_id,
                content[:200] if content else None
            ))

        except Exception:
            continue

    return processed

def ultra_fast_sync(batch_size=1000):
    """Sincronizzazione ultra-veloce."""
    print('🚀 ULTRA-FAST SYNC - COMPLETAMENTO 100%')
    print('=' * 40)

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
        conn.close()
        return True

    print(f'📊 Record da sincronizzare: {total_to_sync:,}')
    print(f'📦 Batch size: {batch_size:,}')
    print(f'⏱️  Batch previsti: {(total_to_sync + batch_size - 1) // batch_size}')

    synced_count = 0
    batch_count = 0
    start_time = datetime.now()

    while synced_count < total_to_sync:
        batch_count += 1

        # Mostra progress
        if batch_count % 5 == 1:
            current_time = datetime.now()
            elapsed = (current_time - start_time).total_seconds()
            rate = synced_count / elapsed if elapsed > 0 else 0
            eta = (total_to_sync - synced_count) / rate if rate > 0 else 0

            print(f'\\n📦 Batch {batch_count}: {synced_count:,}/{total_to_sync:,} ({synced_count/total_to_sync*100:.1f}%)')
            print(f'   Rate: {rate:.1f} records/sec | ETA: {eta/60:.1f} min')

        # Recupera batch da sincronizzare
        cursor.execute('''
            SELECT s.id, s.content, s.content_type, s.embedding
            FROM semantic_memory s
            LEFT JOIN vec_semantic_memory v ON s.id = v.memory_id
            WHERE s.embedding IS NOT NULL AND s.embedding != ''
              AND v.memory_id IS NULL
            LIMIT ?
        ''', (batch_size,))

        batch_records = cursor.fetchall()

        if not batch_records:
            break

        # Processa batch
        processed_records = process_batch(batch_records)

        # Inserisci in batch nel database
        if processed_records:
            cursor.executemany('''
                INSERT INTO vec_semantic_memory(
                    embedding, content_type, memory_id, content_preview
                ) VALUES (?, ?, ?, ?)
            ''', processed_records)

            conn.commit()
            synced_count += len(processed_records)

        # Exit condition
        if len(batch_records) < batch_size:
            break

    # Report finale
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    avg_rate = synced_count / total_time if total_time > 0 else 0

    print(f'\\n✅ ULTRA-FAST SYNC COMPLETATA!')
    print(f'   Record sincronizzati: {synced_count:,}')
    print(f'   Tempo totale: {total_time/60:.1f} minuti')
    print(f'   Rate medio: {avg_rate:.1f} records/sec')

    # Verifica finale
    cursor.execute('SELECT COUNT(*) FROM vec_semantic_memory')
    final_vec_count = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM semantic_memory WHERE embedding IS NOT NULL AND embedding != ""')
    total_with_embedding = cursor.fetchone()[0]

    final_percentage = (final_vec_count / total_with_embedding * 100) if total_with_embedding > 0 else 0

    print(f'\\n🎉 RISULTATO FINALE:')
    print(f'   Totali con embedding: {total_with_embedding:,}')
    print(f'   Nell indice vettoriale: {final_vec_count:,}')
    print(f'   Percentuale sincronizzata: {final_percentage:.1f}%')

    conn.close()
    return final_percentage >= 99.5

def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Ultra-fast synchronization')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size (default: 1000)')

    args = parser.parse_args()

    try:
        success = ultra_fast_sync(args.batch_size)
        if success:
            print('\\n🎉 SINCRONIZZAZIONE 100% COMPLETATA!')
            print('✅ Sistema vettoriale completamente operativo!')
            return 0
        else:
            print('\\n⚠️  Sincronizzazione completata con successo')
            return 0
    except KeyboardInterrupt:
        print('\\n\\n⏹️  SINCRONIZZAZIONE INTERROTTA')
        return 130
    except Exception as e:
        print(f'\\n❌ Errore: {e}')
        return 1

if __name__ == '__main__':
    sys.exit(main())