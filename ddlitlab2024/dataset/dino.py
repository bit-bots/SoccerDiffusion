import argparse
import sqlite3
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
from ddlitlab2024 import DB_PATH


def load_dino_model(device='cuda'):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval().to(device)
    return model


def get_transform():
    return T.Compose([
        T.Resize(224, interpolation=Image.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_cls_embeddings(model, batch_tensor, device='cuda'):
    batch_tensor = batch_tensor.to(device)
    print("🧠 Encoding on GPU...")
    outputs = model.forward_features(batch_tensor)
    return outputs['x_norm_clstoken'].cpu().numpy()  # [B, D]

def fetch_image_batch(cursor, batch_size):
    print("📥 Fetching image batch from database...")
    return cursor.fetchmany(batch_size)

def decode_image(image_blob):
    img = np.frombuffer(image_blob, dtype=np.uint8).reshape(480, 480, 3)
    pil_img = Image.fromarray(img)
    return pil_img


def process_batch(model, transform, batch_data, device):
    ids = []
    tensors = []

    for image_id, image_blob in batch_data:
        pil_img = decode_image(image_blob)
        ids.append(image_id)
        tensor = transform(pil_img)
        tensors.append(tensor)

    batch_tensor = torch.stack(tensors)
    embeddings = extract_cls_embeddings(model, batch_tensor, device)
    return ids, embeddings


def main(db_path, batch_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")

    model = load_dino_model(device)
    transform = get_transform()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get total number of images without embeddings
    cursor.execute("SELECT COUNT(*) FROM Image WHERE embedding IS NULL")
    total_unprocessed = cursor.fetchone()[0]

    cursor.execute("SELECT _id, data FROM Image WHERE embedding IS NULL")

    all_image_ids = []
    all_embeddings = []

    pbar = tqdm(total=total_unprocessed, desc="Embedding images")

    while True:
        batch_data = fetch_image_batch(cursor, batch_size)
        if not batch_data:
            break

        image_ids, embeddings = process_batch(model, transform, batch_data, device)
        all_image_ids.extend(image_ids)
        all_embeddings.extend(embeddings)
        pbar.update(len(image_ids))

    pbar.close()

    print("💾 Storing embeddings in database...")
    update_embeddings_bulk(conn, all_image_ids, all_embeddings)

    conn.close()
    print("✅ Done: all images processed.")


def update_embeddings_bulk(conn, image_ids, embeddings):
    cursor = conn.cursor()
    print("💾 Preparing data for bulk update...")
    data = [
        (np.array(emb, dtype=np.float32).tobytes(), image_id)
        for image_id, emb in zip(image_ids, embeddings)
    ]
    print("📤 Executing bulk update...")
    cursor.executemany("UPDATE Image SET embedding = ? WHERE _id = ?", data)
    conn.commit()
    print("✅ Embeddings updated in database.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed DINOv2 CLS tokens into a SQLite database.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DINO embedding")
    args = parser.parse_args()
    main(DB_PATH, args.batch_size)
