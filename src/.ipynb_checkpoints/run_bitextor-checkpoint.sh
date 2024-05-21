bicleaner-ai-classify lyu/MT/data/NLLB.en-ja.tsv lyu/MT/data/NLLB.en-ja_bitextro_score.tsv bitextor/bicleaner-ai-full-large-en-xx \
    --scol 1 --tcol 2 \
    --score_only \
    --tmp_dir lyu/cache \
    --disable_porn_removal \
    --disable_minimal_length \
    --logfile lyu/MT/data/bitextor.log \
    --batch_size 64
