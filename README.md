# MegaQuery

Code for **MegaQuery-680M**, a FinePDFs retrieval dataset with approximately
680 million synthetic queries in English and French. The pipeline retrieves
personas, generates queries, builds a document index, mines and reranks hard
negatives, and exports training examples with language-local document references.

[Dataset](https://huggingface.co/datasets/wissamantoun/MegaQuery-680M) ·
[Blog](https://huggingface.co/blog/wissamantoun/megaquery-680m) ·
[Documentation](docs/README.md)

| Language | Dataset configuration | Queries | Training rows (query groups) |
| --- | --- | ---: | ---: |
| English | `eng_Latn` | 569,212,798 | 145,619,690 |
| French | `fra_Latn` | 108,001,708 | 27,323,217 |
| **Total** | | **677,214,506** | **172,942,907** |

The combined release contains approximately **680 million queries** in
**173 million training groups**. Each training row contains a primary query,
its additional queries, one positive document reference, and shared hard-negative
references with retrieval and reranker scores. A group usually contains four
queries, but the number can vary; query totals include both primary and
additional queries. Training rows also differ from distinct-document counts,
since multiple groups can refer to the same positive document.

Document text stays in the pinned FinePDFs source snapshot. The release includes
document references, source manifests, and loading helpers to recover it.
Document indices are local to each language; keep the configuration name with
each index when combining English and French. See the [data guide](docs/data-guide.md)
for loading examples and the full schema.

- [Setup, dependencies, and hardware](docs/environment.md)
- [Run the stages in order](docs/pipeline.md)
- [Generation and prompts](docs/generation.md)
- [Document embeddings and hard negatives](docs/mining.md)
- [Export and combine English/French](docs/export.md)
- [Load documents and training examples; published schemas](docs/data-guide.md)
- [Reproduction limits and implementation changes](docs/reproduction.md)
- [Attribution and licenses](docs/attribution.md)

Start with the environment guide and copy an example from `configs/`. This
repository contains code and configuration examples; model weights, personas,
source documents, and expensive intermediate artifacts must be supplied or built.
Code: [Apache-2.0](LICENSE). Published dataset: ODC-By.
