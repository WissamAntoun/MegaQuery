# Attribution and licensing

MegaQuery was developed by [Wissam Antoun](https://github.com/WissamAntoun).
This release adapts the author's generation, document indexing, hard-negative
mining, and dataset export code and its stage READMEs.

The generation recipe is based on
[Qwen3 Embedding](https://arxiv.org/abs/2506.05176) and its published prompt
approach. This is a reimplementation, rather than the original Qwen training-data
pipeline. [FinePhrase](https://huggingface.co/spaces/HuggingFaceFW/finephrase)
informed the choice of a small generation model.

## Code

The available project source had no top-level code license. Its vendored USearch
and Infinity trees carry their own licenses; those trees are not included here.
The included `src/convert_model_to_seq_cls.py` explicitly carries the vLLM
contributors' **Apache-2.0** copyright and license header, which is preserved.

This standalone code release uses **Apache-2.0**; see [LICENSE](../LICENSE) and
[NOTICE](../NOTICE). This is a code-release licensing choice, independent of the
dataset license, and does not retroactively license the surrounding repository
or any third-party model/data. The conversion utility's upstream license text is
also retained in [licenses/Apache-2.0.txt](../licenses/Apache-2.0.txt).

## Data and models

The published [MegaQuery-680M dataset](https://huggingface.co/datasets/wissamantoun/MegaQuery-680M)
is released under **ODC-By**. Its text references
[FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs); follow that
source's attribution and usage information. The code license does not change
source-document rights.

The workflow uses [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it),
[Qwen3 Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), and
[the BGE/Gemma sequence-classification reranker](https://huggingface.co/wissamantoun/bge-reranker-v2-gemma-seq-cls).
Their weights, tokenizers and upstream model terms are separate from this code.
Persona data is supplied by the operator and needs its own attribution.

Runtime libraries include Ray, vLLM, PyTorch, Transformers, NumPy, PyArrow,
DuckDB, Hugging Face Datasets/Hub, tqdm and USearch. They are installed as
dependencies and retain their own licenses.
