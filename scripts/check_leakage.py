"""
Verifica que não há vazamento de dados a nível de fardo entre treino/validação e
teste, para cada versão do dataset gold (p3.1 — ver
correcoes_banca/p3.1_resplit_por_fardo.md).

Uso:
    python scripts/check_leakage.py              # verifica v1, v8, v9, v10, v11, v12
    python scripts/check_leakage.py v1 v9 v10     # verifica só as versões informadas

Para cada versão, varre data/gold/datasets/<v>/{train_val,test}/**/*.npy, extrai o
bale_id de cada nome de arquivo e confirma que o conjunto de fardos do treino não
intersecta o conjunto de fardos do teste. Também compara, quando disponível, as
contagens de fardo registradas no metadata.json (gravado pelo DatasetBuilder) com
as observadas em disco.

Código de saída: 0 se tudo OK, 1 se houver vazamento ou erro de parsing de nome.
"""
import sys
import os
import json
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.bale_id import extract_bale_id

DEFAULT_VERSIONS = ['v1', 'v8', 'v9', 'v10', 'v11', 'v12']
GOLD_ROOT = Path(project_root) / 'data' / 'gold' / 'datasets'


def collect_bale_ids(split_dir: Path):
    """Retorna (set de bale_ids, total de patches, lista de erros de parsing)."""
    bale_ids = set()
    count = 0
    errors = []
    for npy_path in split_dir.rglob('*.npy'):
        count += 1
        try:
            bale_ids.add(extract_bale_id(npy_path.name))
        except ValueError as e:
            errors.append(str(e))
    return bale_ids, count, errors


def check_version(version: str) -> bool:
    version_dir = GOLD_ROOT / version
    train_dir = version_dir / 'train_val'
    test_dir = version_dir / 'test'

    print(f"\n=== {version} ===")

    if not train_dir.exists() or not test_dir.exists():
        print(f"  [SKIP] '{train_dir}' ou '{test_dir}' não encontrado.")
        return True

    train_bales, train_count, train_errors = collect_bale_ids(train_dir)
    test_bales, test_count, test_errors = collect_bale_ids(test_dir)

    ok = True

    for err in (train_errors + test_errors)[:5]:
        print(f"  [ERRO] {err}")
    if train_errors or test_errors:
        print(f"  [ERRO] {len(train_errors) + len(test_errors)} arquivo(s) com nome fora do padrão esperado.")
        ok = False

    overlap = train_bales & test_bales
    print(f"  Fardos em treino/val: {len(train_bales)} | Fardos em teste: {len(test_bales)}")
    print(f"  Patches em treino/val: {train_count} | Patches em teste: {test_count}")

    if overlap:
        sample = sorted(overlap)[:10]
        suffix = ' ...' if len(overlap) > 10 else ''
        print(f"  [FALHA] {len(overlap)} fardo(s) presentes em treino E teste: {sample}{suffix}")
        ok = False
    else:
        print("  [OK] Nenhum fardo em comum entre treino/val e teste.")

    metadata_path = version_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        split_meta = metadata.get('split', {})
        if split_meta:
            print(
                f"  metadata.json: strategy={split_meta.get('strategy')!r}, "
                f"seed={split_meta.get('seed')}, "
                f"n_bales_train={split_meta.get('n_bales_train')}, "
                f"n_bales_test={split_meta.get('n_bales_test')}"
            )
            meta_train = split_meta.get('n_bales_train')
            meta_test = split_meta.get('n_bales_test')
            if meta_train is not None and meta_train != len(train_bales):
                print(
                    f"  [AVISO] n_bales_train do metadata ({meta_train}) difere "
                    f"do observado em disco ({len(train_bales)})."
                )
            if meta_test is not None and meta_test != len(test_bales):
                print(
                    f"  [AVISO] n_bales_test do metadata ({meta_test}) difere "
                    f"do observado em disco ({len(test_bales)})."
                )
        else:
            print("  [AVISO] metadata.json sem bloco 'split' (versão não usa group_by_bale?).")
    else:
        print("  [AVISO] metadata.json não encontrado para esta versão.")

    return ok


def main():
    versions = sys.argv[1:] or DEFAULT_VERSIONS
    results = {v: check_version(v) for v in versions}

    print("\n=== Resumo ===")
    all_ok = True
    for v, ok in results.items():
        print(f"  {v}: {'OK' if ok else 'FALHA'}")
        all_ok = all_ok and ok

    if not all_ok:
        print("\nVazamento de dados (ou erro de parsing) detectado. Ver detalhes acima.")
        sys.exit(1)

    print("\nNenhum vazamento de dados a nível de fardo detectado em nenhuma versão verificada.")


if __name__ == '__main__':
    main()
