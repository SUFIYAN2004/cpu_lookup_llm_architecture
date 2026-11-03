# cpu_lookup_llm_architecture
I built a language model that trains in seconds on a CPU. No GPU, no matrices, no backprop.  It’s a lookup-based LLM. Instead of billions of weights, the “model” is a SQLite table that maps contexts to next-token scores.
