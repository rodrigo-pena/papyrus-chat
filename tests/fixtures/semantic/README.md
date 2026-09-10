# Semantic retrieval evaluation seeds

Thirty source-linked, agent-curated questions over the five existing pinned
fixture documents. Judgments are grounded in their editions, published
translations, or explicitly labeled metadata; they have not been reviewed by
an independent papyrologist. See ../idp.data/PROVENANCE.md for source licensing.

This is a development smoke benchmark, not 30 independent documents or an
exhaustive relevance set. Repeated questions test languages and textual facets.
Each has a relevant source and a contrast document. Unjudged results are scored
as zero gain, not asserted to be irrelevant. Top-20 recall on the five-document
fixture corpus is especially permissive; report rankings and the full DDbDP
subset separately before drawing conclusions about usefulness.

baseline.json records exact lexical search with each question as one OR group
of words, across all fields, without collection or HGV subject restrictions.
This reproducible baseline does not simulate an LLM generating Greek synonyms
or choosing HGV labels. It must not be described as the complete agent baseline.
