import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def main():
    # initialize faiss index
    d = 1024
    index = faiss.IndexFlatIP(d)
    print("initialize faiss index successfully!")
    print("start loading model...")
    model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5',cache_folder="/home/2/uh02312/lyu/checkpoints",trust_remote_code=True)
    print(" successfully load model!")
    pool=model.start_multi_process_pool(target_devices=['cuda:0'])
    
    en_texts = []
    print("loading parallel corpus...")
    with open("/home/2/uh02312/lyu/MT/data/wmt_20-23.en", "r", encoding="utf-8") as fen:
        for line in fen:
            en_texts.append(line.strip())
    num_texts = len(en_texts)
    print("successfully load all parallel sentences! Total {} pairs of sentences".format(num_texts))


    # Step 2 & 3: compute embeddings for all English texts and add them to the index
    batch_size = 512
    with tqdm(total=num_texts, desc="compute embeddings for en", unit="sentence") as pbar:
        for i in range(0, num_texts, batch_size):
            batch_texts = en_texts[i:i+batch_size]
            size_batch = len(batch_texts)
            batch_vectors = model.encode_multi_process(batch_texts,pool=pool,batch_size=size_batch)
            index.add(batch_vectors)

                # update progress bar
            pbar.update(size_batch)
            pbar.set_postfix_str(f"Processed: {i+size_batch}/{num_texts}")

    # Step 4: save the index
    faiss.write_index(index, "/home/2/uh02312/lyu/MT/data/RAG/wmt20-23_en.idx")

if __name__ == "__main__":
    main()