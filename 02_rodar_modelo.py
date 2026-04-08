import os

# Importante: definir antes de importar tensorflow
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import argparse
import csv
import gc
import random
import time
from typing import Dict, List, Tuple

import cv2
import tensorflow as tf
from deepface import DeepFace
from tensorflow.keras import backend as K

# =========================
# CONFIG
# =========================
DATASET_PATH = "imagens/utkcropped"
OUTPUT_DIR = "saida_modelo"
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "resultado_final.csv")

SAMPLE_SIZE = 50000
LOG_EVERY = 200
FLUSH_EVERY = 200
RANDOM_SEED = 42

DETECTOR_BACKEND = "opencv"
ENFORCE_DETECTION = False

TASKS = {
    "age": "idade_pred",
    "gender": "genero_pred",
    "race": "etnia_pred",
}


# =========================
# GPU SETUP
# =========================
def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs disponíveis:", gpus)

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("Erro ao configurar memory growth:", e)
    else:
        print("Nenhuma GPU detectada pelo TensorFlow. Rodando em CPU.")


# =========================
# UTILS
# =========================
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def list_files(dataset_path: str, sample_size: int) -> List[str]:
    files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".jpg")]
    files.sort()
    random.seed(RANDOM_SEED)
    return random.sample(files, min(sample_size, len(files)))


def sample_csv_path() -> str:
    return os.path.join(OUTPUT_DIR, f"sample_{SAMPLE_SIZE}_{RANDOM_SEED}.csv")


def save_or_load_sample_files() -> List[str]:
    """
    Garante que age/gender/race usem exatamente a mesma amostra.
    """
    path = sample_csv_path()

    if os.path.exists(path):
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # header
            return [row[0] for row in reader]

    files = list_files(DATASET_PATH, SAMPLE_SIZE)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arquivo"])
        for filename in files:
            writer.writerow([filename])

    return files


def parse_filename(filename: str) -> Tuple[int, int, int]:
    parts = filename.split("_", 3)
    if len(parts) < 4:
        raise ValueError(f"Nome fora do padrão UTKFace: {filename}")
    idade_real, genero_real, etnia_real, _ = parts
    return int(idade_real), int(genero_real), int(etnia_real)


def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Falha ao ler imagem: {path}")
    return img


def analyze_single_action(img, action: str):
    result = DeepFace.analyze(
        img_path=img,
        actions=[action],
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=ENFORCE_DETECTION,
        silent=True,
    )
    if isinstance(result, list):
        result = result[0]
    return result


def extract_prediction(result: dict, action: str):
    if action == "age":
        return int(result["age"])
    if action == "gender":
        return max(result["gender"], key=result["gender"].get)
    if action == "race":
        return max(result["race"], key=result["race"].get)
    raise ValueError(f"Ação desconhecida: {action}")


def partial_csv_path(action: str) -> str:
    return os.path.join(OUTPUT_DIR, f"parcial_{action}.csv")


def cleanup_memory():
    """
    Limpeza explícita ao final de cada task.
    Útil mesmo rodando uma task por processo.
    """
    try:
        from deepface.modules import modeling

        modeling.cached_models = {}
    except Exception:
        pass

    try:
        K.clear_session()
    except Exception:
        pass

    gc.collect()
    time.sleep(2)


# =========================
# PASS EXECUTION
# =========================
def run_task(files: List[str], action: str, output_column: str):
    path_csv = partial_csv_path(action)

    print(f"\n=== Rodando tarefa: {action} ===")
    print(f"Saída parcial: {path_csv}")

    # warm-up
    if files:
        warm_img = load_image(os.path.join(DATASET_PATH, files[0]))
        _ = analyze_single_action(warm_img, action)
        print(f"Warm-up de {action} concluído.")

    start_time = time.time()
    buffer_rows = []

    with open(path_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arquivo", output_column])

        total = len(files)

        for idx, filename in enumerate(files, 1):
            try:
                img = load_image(os.path.join(DATASET_PATH, filename))
                result = analyze_single_action(img, action)
                pred = extract_prediction(result, action)

                buffer_rows.append([filename, pred])

                if len(buffer_rows) >= FLUSH_EVERY:
                    writer.writerows(buffer_rows)
                    buffer_rows.clear()
                    f.flush()

                if idx % LOG_EVERY == 0 or idx == total:
                    elapsed = time.time() - start_time
                    avg_per_img = elapsed / idx
                    remaining = avg_per_img * (total - idx)
                    print(
                        f"[{action}] {idx}/{total} ({100 * idx / total:.1f}%) | "
                        f"tempo: {elapsed:.1f}s | "
                        f"média/img: {avg_per_img:.3f}s | "
                        f"restante: {remaining / 60:.1f} min"
                    )

            except Exception as e:
                print(f"[{action}] Erro em {filename}: {e}")

        if buffer_rows:
            writer.writerows(buffer_rows)
            f.flush()

    print(f"Tarefa {action} finalizada.")
    cleanup_memory()


# =========================
# MERGE
# =========================
def read_partial_predictions(action: str, output_column: str) -> Dict[str, str]:
    path_csv = partial_csv_path(action)
    data = {}

    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"Arquivo parcial não encontrado: {path_csv}")

    with open(path_csv, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["arquivo"]] = row[output_column]

    return data


def merge_results(files: List[str]):
    print("\n=== Unindo resultados finais ===")

    age_map = read_partial_predictions("age", "idade_pred")
    gender_map = read_partial_predictions("gender", "genero_pred")
    race_map = read_partial_predictions("race", "etnia_pred")

    with open(FINAL_OUTPUT, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "arquivo",
                "idade_real",
                "genero_real",
                "etnia_real",
                "idade_pred",
                "genero_pred",
                "etnia_pred",
            ]
        )

        missing = 0

        for filename in files:
            try:
                idade_real, genero_real, etnia_real = parse_filename(filename)

                idade_pred = age_map.get(filename)
                genero_pred = gender_map.get(filename)
                etnia_pred = race_map.get(filename)

                if idade_pred is None or genero_pred is None or etnia_pred is None:
                    missing += 1
                    continue

                writer.writerow(
                    [
                        filename,
                        idade_real,
                        genero_real,
                        etnia_real,
                        idade_pred,
                        genero_pred,
                        etnia_pred,
                    ]
                )
            except Exception as e:
                print(f"Erro ao unir {filename}: {e}")

    print(f"CSV final salvo em: {FINAL_OUTPUT}")
    print(f"Registros incompletos ignorados: {missing}")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["age", "gender", "race", "merge"],
        required=True,
        help="Tarefa a executar",
    )
    args = parser.parse_args()

    ensure_output_dir()
    setup_gpu()

    files = save_or_load_sample_files()
    print(f"Total de imagens na amostra fixa: {len(files)}")

    global_start = time.time()

    if args.task in TASKS:
        run_task(files, args.task, TASKS[args.task])
    elif args.task == "merge":
        merge_results(files)

    total_time = time.time() - global_start
    print(f"\nFinalizado em {total_time / 60:.1f} minutos.")


if __name__ == "__main__":
    main()
