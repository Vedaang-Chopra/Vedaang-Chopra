Title: SAI   METIS Presentation
Date: 2025-12-12
Category: Research Paper Analysis
Slug: sai---metis-presentation
Summary: Detailed analysis and presentation notes for SAI   METIS Presentation.

<div class="download-box" style="margin-bottom: 2rem; padding: 1rem; background: var(--btn-bg); border-radius: 8px; display: inline-block;">
    <a href="{attach}../raw_content/CS 8803  SAI - METIS Presentation.pptx" style="text-decoration: none; font-weight: bold;">
        📥 Download Original Slides (PPTX)
    </a>
</div>

## METIS: Fast Quality-Aware RAG Systems with
Configuration Adaptation

Hello all, today I will  be presenting the paper METIS, from researchers at Princeton and University of Chicago

**Key Takeaways:**
*   Presented by:- Vedaang Chopra

## Presentation Flow

Before I begin let me go through the presentation flow of how the paper is divided, so to understand the paper majorly there are 4 sections  The first section is on  A brief background on RAG; An introduction to the problem and some of its configurations The second section is about the Metis architecture The third section are some implementation details of the paper, and its results In the end we can wrap up with some conclusions and the strengths and Limitations of the paper

## Section 1 – Introduction & Background

## What is Retrieval Augmented Generation(RAG) ? 

Large language models (LLMs) — like GPT or LLaMA — are very capable,  but they have two weaknesses: Lack of  domain-specific knowledge (e.g., financial data, medical details). Lack of access to recent or dynamic information (e.g., today’s news). [Debatable] So we use RAG that before answering the question, we first search a large database, pick the most relevant text passages, and then uses them to craft its response. It’s already used in: QA systems like ChatGPT’s “Browse with Bing” Chatbots that need company knowledge bases Search systems that summarize documents

**Key Takeaways:**
*   RAG (Retrieval-Augmented Generation) means that instead of relying only on what the model remembers from training, we retrieve extra knowledge (called “chunks” of text) from a large database before generating an answer. Example:
*   If you ask, “When was Apple’s latest iPhone released?” → The LLM retrieves recent tech news snippets (retrieval). → Then, it reads those snippets and generates a concise answer (generation).
*   RAG — retrieval + reasoning → better answers.

## What problem does this paper aim to solve?

“RAG systems make LLMs more accurate by adding external knowledge — but this comes at a cost: they get slower. So there’s a constant trade-off between quality and speed.” “The first challenge is that natural language queries are vague. A question like ‘Compare NVIDIA’s operating cost…’ doesn’t specify how many documents or chunks are needed — the system has to figure that out.” “Second, there are many configuration options — how many chunks to retrieve, how to combine them, whether to summarize. Trying every combination for every query would be computationally infeasible.” “Finally, scheduling interacts with configuration — even if you choose a good setup, GPU memory or batching constraints may delay the query. The best configuration also depends on available system resources.” “So, as shown in the small table, quality improves when we use more context, but that slows things down; using fewer chunks speeds things up but risks missing key information.” “In short, RAG designers need to balance accuracy versus latency, and this paper — METIS — proposes a systematic way to do that.” RAG designers must balance accuracy vs. latency.

**Key Takeaways:**
*   RAG systems boost quality, but they also slow things down. There’s a trade-off in RAG between quality and speed, which is a Hard Problem. Some of the problems include : -
*   Natural language queries are vague. Queries like “Compare NVIDIA’s operating cost…” doesn’t tell the system how many chunks are required.
*   RAG systems have multiple configurations leading to exponential combinations. Testing all combinations for every query would be too slow.
*   Scheduling interacts with configuration:- The amount of resources available and the configuration applied for a particular query are directly related.

## What did past RAG systems focus on? What did they miss?

Prior works as mentioned in the paper, focused on major two things, either improving the quality of answer, either fetching the right chunks or optimizing the total latency of the system. They didn’t look at the overall balance of accuracy and speed of the system “Before METIS, a lot of RAG research focused on only one side of the trade-off — either making RAG faster or improving its answer quality.” “The speed-focused systems worked on the serving layer — optimizing GPU scheduling, batching similar queries, or memory usage to cut latency.” “The quality-focused systems tried to tune RAG configurations — like deciding how many chunks to retrieve or how to summarize them for better answers.” “But here’s the major gap: none of these methods tried to jointly optimize quality and delay. They either got better accuracy at the cost of speed, or faster responses with lower accuracy.” “METIS is the first to tackle this balance directly — optimizing both together, dynamically per query.”

**Key Takeaways:**
*   There are already many RAG-related systems (the citations [2, 17, 32, 34, 37, 40, 44, 54, 76, 87, 90]) that mainly do two things:
*   Earlier research mostly focused on one side of the trade-off:
*   Speed-focused methods → Focus on model selection or serving optimization. Examples include: -
*   Optimized Scheduling or GPU Memory usage to make queries faster.
*   Batching similar queries to save GPU time.
*   Quality-focused methods → tuned RAG’s internal configurations. Example include: -
*   How many chunks to retrieve ?
*   How to summarize them for better answers?
*   MAJOR LIMITATION: - Prior Works didn’t focus on the  trade-off between response delay and generation quality

## What are the main RAG configuration “knobs”?

Knobs are simply put some configurations of RAG systems. The authors are focused on 3 knobs across the paper that are quite general and are used across different RAG systems.  Number of Chunks: - It is like deciding how many pages of a document you have to share with the LLM for your question. Too few chunks → not enough information → wrong or incomplete answer. Too many chunks → extra reading → slower response, and even confusion (irrelevant text can reduce accuracy).   Synthesis_method: - Once you have the chunks, how do you pass it is as input to the model.  Stuff: - Basically you have 3 chunks pass all of them to the model. LLM reads everything at once. If input is small it is fine, for large chunks it is a problem as the model can “forget” details in the middle — the lost-in-the-middle problem. Map-Rerank :- Here pass all the chunks to the model separately, and pick the answer where model is most confident. This a fast and light method and works well if answers are contained in one chunk. Fails if info is spread across multiple chunks as model will not be able to combine facts. Map-Reduce:- Each chunk is summarized separately (Map phase). Those summaries are combined, and the LLM produces the final answer (Reduce phase).Great for complex reasoning across multiple chunks. Slower and uses more computation. Quality depends on how long the summaries are.  Summary length: - This only matters when you use map-reduce. Short summaries → faster, but might lose important details. Long summaries → more accurate but slower.

**Key Takeaways:**
*   Every RAG query has parameters (or knobs) that control how it’s run:
*   Number of Chunks
*   Synthesis Method
*   Stuff:- Concatenate all chunks and feed them to the LLM together
*   Map-Rerank: - LLM reads each chunk separately, answers for each, and then we pick the most confident answer
*   Map-Reduce: - Each chunk is summarized separately (Map phase). Those summaries are combined, and the LLM produces the final answer (Reduce phase).
*   Summary length:- This only matters when you use map-reduce

## How do these configurations affect quality and delay?

“This part of the paper shows why RAG configuration tuning is so important — the authors run controlled experiments to test how different settings change both response time and answer quality.” “They used three queries from the MuSiQue dataset, which has reasoning-based QA tasks. Q1 is a simple single-fact question, Q2 needs comparing multiple entities, and Q3 requires multi-step reasoning about Voyager 1 — so we cover simple to complex queries.” “Then they vary three configuration knobs: 1️⃣ The synthesis method — whether the LLM reads chunks separately (map_rerank), all at once (stuff), or summarizes first (map_reduce). 2️⃣ The number of retrieved chunks — from 1 to 35. 3️⃣ The summary length in map_reduce — from 1 to 100 words.” “They measure F1 score for quality and response delay for speed, and plot the results in Figure 4.” “The goal is to observe how each knob affects performance — and they find that the optimal configuration shifts depending on query complexity. This motivates the need for per-query configuration adaptation, which is what METIS later does.

**Key Takeaways:**
*   Goal:  Test how changing each RAG configuration knob affects response time vs. quality.
*   Experiment Done: -  3 queries from MuSiQue (reasoning QA) dataset was selected
*   Q1: simple single-fact question - “In what county was William W. Blair born?”
*   Q2: moderate, cross-chunk comparison: - “Are Alison Skipper, Diane Gilliam Fisher, and Rachel McAdams from the same country?”
*   Q3: complex, multi-step reasoning- “When and why did Voyager 1, the spacecraft that detected storms on Neptune, leave our solar system?”
*   Experiments run:
*   Vary synthesis method → map_rerank, stuff, map_reduce
*   Vary number of retrieved chunks → from 1 to 35
*   Vary intermediate summary length (in map_reduce) → 1 to 100 words
*   Measurement:
*   Track F1 score (quality) and response delay (seconds) for each query.
*   Observe how optimal configurations shift by query complexity.

As we can see that for q1, a   Each knob affects both quality and delay.But they also interact with each other — changing one affects the others. Example: If you increase num_chunks, maybe you need to switch from stuff → map_reduce (because reading all chunks together becomes too long). If you shorten intermediate_length, maybe quality drops but delay improves. This is why optimizing RAG performance isn’t trivial — it’s a multi-dimensional trade-off. “This figure shows the core finding of the paper — how each RAG configuration knob affects the trade-off between F1-score (quality) and delay (speed).” “Each plot isolates one knob: (a) changes the synthesis method, (b) changes the number of retrieved chunks, and (c) changes the summary length.” “From the first plot — different synthesis methods work best for different query types: simple Q1 performs best with map_rerank, while more complex queries like Q2 and Q3 need stuff or map_reduce to combine multiple pieces of information.” “In the second plot — the optimal number of chunks depends on query complexity. Adding too many chunks increases delay and even hurts quality because of the lost-in-the-middle problem.” “In the third plot — summary length matters: short summaries are fine for easy questions, but longer summaries preserve reasoning context for complex ones.” “Overall takeaway: there’s no single best configuration — every query has its own sweet spot between accuracy and latency, which motivates the need for a dynamic system like METIS.”

**Key Takeaways:**
*   RESULTS: -
*   Different synthesis methods suit different query types — simple queries prefer map_rerank, while complex reasoning needs stuff or map_reduce.
*   Optimal number of chunks varies per query — adding too many causes “lost-in-the-middle,” hurting both quality and speed.
*   Summary length matters — short summaries work for easy questions; longer ones preserve reasoning for complex ones.

## Why per-query tuning matters ?

We need to tune per-query. But we can’t just brute force our way, because even just with some simple less combination in our knobs, the combination space will explode.  Hence, we need a system that: Quickly narrows down the huge configuration space Chooses good configurations cheaply and dynamically That’s exactly what METIS will do in the next section. “Now that we’ve seen how each knob affects quality and delay, the next question is — why not just tune the best configuration per query?” “This figure shows exactly that: when we adapt RAG settings per query, we achieve up to 3× lower delay for the same quality.” “Static configurations, shown by the Pareto boundary in blue, can’t keep up — to reach similar latency, they lose at least 10% in F1-score.” “So, per-query adaptation clearly helps — but brute-forcing all possible knob combinations is infeasible, since the configuration space grows exponentially.” “That’s where METIS comes in — it builds a system that can narrow down this huge space efficiently and pick the right configuration dynamically for each query.”

**Key Takeaways:**
*   Per-query tuning achieves up to 3× lower delay for the same quality.
*   Static configs can’t keep up — to match the same delay, they lose ≥ 10% in quality.

## Section 2 – METIS System Design

## What is METIS and what makes it different?

“Now that we’ve seen the motivation for per-query tuning, METIS is the system designed to make that practical.” “METIS doesn’t replace the LLM — it acts as a RAG controller, deciding how the LLM should process each query. Think of it as an autopilot that manages the RAG pipeline dynamically.” “It has three main components: 1️⃣ LLM Profiler — analyzes the incoming query and predicts its complexity, reasoning type, and information needs. 2️⃣ Configuration Space Pruner — uses that profile to narrow down from thousands of possible RAG settings to just a few promising ones. 3️⃣ Joint Scheduler — picks the best configuration that fits current GPU memory and load, so we balance delay and quality.” “The diagram shows this flow: the query first goes through the profiler → pruned configuration space → joint scheduler → and finally to retrieval and synthesis using the chosen settings.” “So what makes METIS different is this end-to-end adaptivity — it tunes configurations per query while being aware of system resources, something earlier RAG systems never did.”

**Key Takeaways:**
*   METIS is a RAG controller — it doesn’t replace the LLM; it controls how RAG runs. It has 3 major components: -
*   LLM Profiler
*   Configuration Space
*   Joint Scheduler

## How does METIS estimate what a query needs?

The first thing METIS does for each query is to create a “profile” of the question. Query Complexity — “Is this question hard or simple?”  Output: High / Low Low = simple “lookup” questions like “Who is NVIDIA’s CEO;  High = explanatory or multi-step questions like “Why did Voyager 1 leave the solar system?” Joint Reasoning Requirement — “Do I need to combine multiple facts?” Output: Yes / No No = one chunk contains the full answer.  e.g., “What is Tesla’s headquarters address?” Yes = you must connect data from several chunks.  e.g., “Compare Tesla’s and Ford’s 2023 profits.” Pieces of Information Required — “How many separate items must I look at?” Numeric estimate (1–10). e.g., For “Compare NVIDIA’s profits across 3 quarters,” → 3 pieces. Length of Summarization Needed — “How much should I condense each chunk?” 30 – 200 words typically. Complex, data-heavy queries need longer summaries; simple ones can get by with shorter ones. Modern LLMs are good at analyzing language structure and intent. They can tell the difference between a factual lookup (“Who”) and a reasoning question (“Why,” “Compare,” “Summarize”).

**Key Takeaways:**
*   METIS creates profile for every query. This profile is a short summary that tells the system:
*   Query Complexity: - How complex the question is ? Output: Low / High ?
*   Joint Reasoning Requirement: - Whether those pieces(chunks) must be reasoned about together ? Output: Yes / No
*   Pieces of Information Required: - How many pieces of information it might need ? Output: - Numeric estimate (1–10).
*   Length of Summarization Needed: - And how much summarization is necessary. Output : - 30 – 200 words typically

## How does METIS estimate what a query needs(contd.)?

Here shows the prompt that the authors use to get the answers to the 4 questions that help profile each query.  Also another example shows the metadata about the dataset that is passed while profiling that influences the LLM’s decision   They explicitly state: “We use a very simple prompt… we don’t perform any prompt tuning or optimizations.” Because their goal is not to craft the perfect prompt — it’s to show that even a straightforward natural-language instruction can yield strong profiling accuracy when paired with the METIS mapping and scheduling pipeline.  The authors also tested feeding the profiler more data (like which embedding model is used). It didn’t improve results much — because embedding models behave similarly. So they stick to simple, high-level metadata.

**Key Takeaways:**
*   f"""
*   For the given query = {get.query()}:
*   Analyse the language and internal structure of the query and provide the following information :
*   1. Does it need joint reasoning across multiple documents or not?
*   2. Provide a complexity profile for the query:
*   Complexity: High/Low
*   Joint Reasoning needed: Yes/No
*   3. Does this query need input chunks to be summarized, and if yes, provide a range in words for the summarized chunks.
*   4. How many pieces of information are needed to answer the query?
*   database_metadata = {get.metadata()}
*   chunk_size = {get.chunk_size()}
*   Estimate the query profile along with the database_metadata and chunk_size to provide the output.
*   """
*   Example — for KG RAG FinSec
*   def get_metadata():
*   metadata = "The dataset consists of multiple chunks of information from Fortune 500 companies on financial reports from every quarter of 2023. The chunk size is 1024 tokens."
*   return metadata

## How does METIS convert profiles into configurations?

Algorithm: -  If no joint reasoning → use map_rerank. Else if joint reasoning and low complexity → use stuff. Else (joint + high complexity) → consider stuff or map_reduce. Set num_chunks range to [n, 3n]: Lower bound n ensures you at least try to retrieve one chunk per needed piece of info. Upper bound 3n gives wiggle room because retrievers often need 2–3× redundancy to reliably grab the right info. Also leaves options for the scheduler to pick what fits GPU memory. Set intermediate_length to the profiler’s suggested range.  Why does this configuration output are ranges ?  Keeps quality high (we don’t prune away good options) Keeps search small (50–100× fewer configs) Leaves room for the scheduler to choose what fits current GPU memory

**Key Takeaways:**
*   METIS doesn’t ask that LLM to output exact knob values. Instead, it uses a cheap, rule-based mapper that converts those 4 signals into ranges for the three knobs:
*   synthesis_method ∈ {map_rerank, stuff, map_reduce}
*   num_chunks ∈ [n, 3n]
*   intermediate_length ∈ profiler’s suggested range (used only if map_reduce is an option)
*   Why not let the LLM output exact knob values?
*   Because that would require continuous re-training of the profiler to adapt to new pipelines and options (expensive, brittle).

## How does METIS choose configurations that fit available GPU memory?

Question: - Now we have a small set of good configs for this query. Which single config should we run right now? Answer: Pick the best one that fits GPU memory now (to avoid queuing), while staying inside the pruned, quality-safe set. Why? Because even a “computationally lighter” method can be slower overall if it doesn’t fit in memory and must wait. Why joint scheduling matters (vs. separating decisions) If you pick “theoretically fastest” config without checking memory (baseline), you can end up waiting, which makes it slower end-to-end. METIS avoids this by coupling config choice with live resource checks. Figure 8 intuition stuff is usually faster if it fits (one big prompt). But stuff is memory-hungry: long inputs (many chunks) can exceed available VRAM → the request waits in queue. map_reduce runs in smaller slices (mappers, then reducer). Even if it needs more total compute, its pieces fit into the current free memory and can start immediately. Result: lower wall-clock delay.

**Key Takeaways:**
*   The selection heuristic (“best-fit”)
*   For every candidate config in the pruned set, estimate memory need (mainly from num_chunks and, if relevant, intermediate_length).
*   Check current free GPU memory for the running batch.
*   Among configs that fit right now, pick the one with the highest memory footprint (i.e., the richest that still fits).
*   Within a quality-safe pruned set, “slightly more expensive” often correlates with slightly better quality (e.g., 6 chunks > 5 chunks), so take the best that fits now.
*   Example: pruned says num_chunks ∈ [5, 10], and both 5 and 6 fit — pick 6.
*   Run it immediately; don’t pick a config that would overflow memory (that would queue and inflate delay).
*   In short: Avoid queuing by picking the fattest config that fits now inside the pre-vetted (quality-safe) space.

## How does METIS handle edge cases and Relearning ?

Sometimes a question is too vague for any model (or human) to profile accurately. Example: “Compare current U.S. stock-market trends.” For reliability of profiler: - Every LLM internally outputs a log-probability (log-prob) for each generated token. That number measures how confident the model is about its own answer. So METIS uses those log-prob values as a proxy for confidence in the profile.They found empirically If confidence ≥ threshold (≈ 90 %), then trust the profile. If confidence < threshold, treat the profile as unreliable and ignore it. Use the “pruned configuration space” from the last 10 successful queries. For Relearning: -  For scheduling a config when no GPU space is left: - METIS falls back gracefully using the profile signals: If no joint reasoning → use map_rerank with as many chunks as fit. If joint reasoning → try stuff or map_reduce with fewer chunks that fit. It can also honor SLOs (e.g., strict latency budgets) by choosing the cheapest viable option. This “loose decoupling” keeps the profile-guided quality intent while still respecting system constraints.

**Key Takeaways:**
*   There are certain edge cases that METIS also handles: -
*   How METIS detects when the profiler is unreliable.
*   How METIS recovers or learns from those cases to improve next time.
*   What if no configuration fits in the GPU ?

So overall this is the flow of the architecture METIS. For every query a small 7B model first profiles the query, then “This diagram summarizes the end-to-end flow of how METIS works behind the scenes for every query.” “Step 1 – The process begins with a query and a small metadata summary of the dataset. A lightweight LLM Profiler (like GPT-4o or LLaMA-70B) analyzes the query — it estimates: • how complex it is, • whether joint reasoning is needed, • how many pieces of information are required, and • how much summarization might help.” “Step 2 – These four high-level outputs form the query profile. Then, using a rule-based mapping, METIS translates that profile into a reduced configuration space — it picks possible synthesis methods, a range for number of chunks, and a range for summary length.” “Step 3 – Finally, the Joint Scheduler looks at current GPU memory and chooses the single best configuration from that reduced space — the one that fits in memory while maintaining high quality.” “So, for each query, METIS goes from plain text → profile → narrowed configuration → best-fit execution, adapting in real time to both the query and the system resources.”

## Section 3 –  Implementation & Evaluation

## How is METIS implemented in practice?

For implementation details of METIS It is built on top of vLLM engine, and it's just addition of some lines of code, so very light and modular The Profiler LLM is a python Class compatible with openAi library, and huggingface API, any opensource LLM can be used for the profiler, the prompt needs to be passed properly and you will get the output The Retriever from vector DB is implemented using Cohere-embed-v3.0 on FAISS db. With chunks fetched from DB, they use langchain chaining to perform the synthesis that is stuff, map_reduce, map_rerank For GPU memory information they use pytorch and pynvml for estimation

## What datasets and metrics are used to evaluate METIS?

They build a classic RAG index: split docs into chunks (LangChain), embed with Cohere-embed-v3.0, store/search with FAISS IndexFlatL2, then send a Poisson stream of queries to simulate load.“To fairly test METIS, the authors use four diverse RAG datasets — each representing a different query style and reasoning need.” “They include: 🟢 SQuAD — simple single-hop QA, one paragraph answers. 🔵 Musique — multi-hop reasoning, combines facts from multiple sources. 🟣 KG RAG FinSec — financial document-level QA, needs multi-chunk retrieval. 🟠 QMSUM — summarization-based QA on meeting transcripts.” “For models, they use Mistral-7B-v3 and Llama-3.1-70B, both quantized for efficient inference. Hardware: dual-GPU NVIDIA A40 server with 384GB RAM.” “Metrics are split into two parts — • Quality: measured with F1-score, standard for QA tasks. • System performance: measured by delay (latency) and dollar cost, showing practical benefits beyond accuracy.” “Baselines include: • vLLM — fixed configuration (no adaptation). • Parrot* — better batching/scheduling but static configs. • AdaptiveRAG* — adapts based on query complexity but ignores resource cost.” “They simulate real-world load by chunking data with LangChain, embedding using Cohere-embed-v3.0, storing in FAISS, and sending a Poisson stream of queries — mimicking actual RAG traffic.”

**Key Takeaways:**
*   Models & hardware
*   Inference models: Mistral-7B-v3 (long context 32K) and Llama-3.1-70B (long context 128K), both AWQ-quantized.
*   Box: dual-GPU NVIDIA A40 server; 384 GB RAM; dual Xeon Gold 6130; 1 GPU serves Mistral-7B-v3; 2 GPUs serve Llama-70B.
*   Metrics
*   Quality: F1 on the generated answer (standard for RAG).
*   System: Delay (response latency) and Dollar cost (to compare against “just use a bigger model” strategies).
*   Datasets (give them different query styles)
*   SQuAD: single-hop reading comprehension.
*   MuSiQue: multi-hop reasoning QA.
*   KG RAG FinSec: financial doc-level QA (needs several chunks).
*   QMSUM: query-focused meeting summarization.
*   Baselines
*   vLLM (fixed configs): strong server with a static RAG setup.
*   Parrot*: advanced batching/scheduling but no per-query config adaptation.
*   AdaptiveRAG*: uses query complexity to pick RAG configs, ignores resource cost.

## How does METIS perform compared to existing systems?

“Now let’s look at how METIS performs against existing RAG serving systems like vLLM, Parrot*, and AdaptiveRAG*.” “Across all four datasets — KG RAG FinSec, Musique, SQuAD, and QMSUM — we see a consistent trend: ✅ METIS achieves 1.6× to 2.5× lower delay compared to the baselines, ✅ while maintaining or even improving quality by 12–18% in F1 score.” “For example, in KG RAG FinSec, METIS gives 16% higher F1 and 2.4× faster responses; in QMSUM, it’s 2.5× faster at the same quality.” “This happens because METIS adapts its configuration per query and jointly considers GPU memory — so it doesn’t waste time waiting for resources like fixed systems do.” “In short — METIS achieves faster answers without sacrificing accuracy — which is exactly the balance prior RAG systems struggled to achieve.”

**Key Takeaways:**
*   METIS Achieved: - Lower delay at same quality: 1.64–2.54× faster than AdaptiveRAG*; vs fixed configs (Parrot*/vLLM) of similar delay, METIS gets +12–18% F1.

## How does METIS perform compared to existing systems?

“This figure focuses on throughput — how many queries per second each system can handle at a fixed latency.” “Across all four datasets, METIS achieves 1.8× to 4.5× higher throughput than other baselines like Parrot* or vLLM.” “The reason is simple: METIS adapts configurations per query and in real time based on GPU memory availability. It doesn’t queue requests that won’t fit — it picks what fits now.” “In contrast, fixed systems waste compute cycles waiting for memory to free up or processing oversized configurations. METIS’s joint scheduling eliminates that waste.” “So at the same delay budget — say 1.8 seconds — METIS can handle several more queries simultaneously without losing response quality.”  Why: It adapts configs per query and picks what fits memory now, reducing queueing and wasted compute; fixed systems can’t exploit this.

**Key Takeaways:**
*   METIS Achieved: - Higher throughput: 1.8–4.5× more QPS at a fixed latency budget.

## Where do METIS’s gains come from?

“This slide breaks down why METIS performs better — where exactly the speedups and quality gains come from.” “Starting with Figure 12 — when they progressively add each METIS component: 1️⃣ Using just the LLM profiler and choosing the median config gives 1.4–1.68× delay reduction. 2️⃣ Adding batching (like Parrot’s system) gives a small boost — about 1.1–1.2× more. 3️⃣ Finally, combining that with resource-aware scheduling — picking the configuration that best fits current GPU memory — brings the total improvement to 1.45–1.75× faster execution.” “In Figure 13, they analyze cost efficiency — using bigger inference models like GPT-4o or Llama-70B doesn’t help. Those fixed systems cost 2.3–6.8× more and still get lower F1-scores compared to METIS.” “So METIS’s gains come not from using a larger model — but from smarter system design — profiling, batching, and GPU-aware scheduling together.

**Key Takeaways:**
*   (Figure-12) Performance with each component(ablation):
*   Use profiler + pick median inside the pruned ranges → 1.4–1.68× faster.
*   Add batching (Parrot-style)* → 1.1–1.2× extra.
*   Add resource-aware config selection (best-fit into current GPU) → another 1.45–1.75×.
*   Figure(13) Cost angle: Simply switching to a much larger inference model with fixed configs is 2.38–6.8× more expensive and still worse F1 than METIS. Even GPT-4o with fixed configs underperforms on F1 and costs 6.8× more in their comparisons.

## What is the Cost of Running METIS ?

“One of the best parts about METIS is that it’s efficient — even though it uses a larger LLM for profiling, the cost is minimal.” “Why? Because the profiler LLM only sees the query text and a one-line metadata summary — not the entire retrieved context.” → That means the profiler’s input is around 100× smaller than what the main LLM processes during answer generation. “It runs once per query, before retrieval — so its total runtime and compute footprint are very small compared to RAG inference.” “Even with a bigger model like GPT-4-level profilers, the cost of profiling is negligible compared to the gain in accuracy and delay reduction.” “In short — METIS achieves the goal of using an expensive model cheaply: it leverages LLM reasoning power only where it matters — for query understanding, not for generation.

**Key Takeaways:**
*   METIS uses a larger LLM for profiling than for generation (e.g., 7B parameter model), but that’s still cheap because the input to the profiler is tiny (just the short query + metadata).
*   Why it’s cheap:
*   Query length ≈ 100× shorter than the retrieved context the main LLM must read.
*   Profiler runs only once per query, not on the full document.
*   Even with a bigger model, total profiling cost ≪ RAG inference cost.
*   So METIS achieves its goal of using an expensive LLM in a cheap way — it only reads the short query, not the whole knowledge base.

## How sensitive is METIS to model or retriever changes?

“This slide tests how robust METIS is — what happens if we change the model or retriever setup?” “In Figure 14, they test profiler feedback: Occasionally, METIS seeds the profiler with the best output from a ‘golden’ configuration — the one that’s slow but very accurate. That feedback improves the profiler’s future predictions, leading to a 4–6% boost in F1-score over time. Importantly, METIS still enforces memory limits, so it doesn’t start choosing overly expensive configurations.” “In Figure 15, they test what happens when switching to a bigger inference model, like Llama-3.1-70B. METIS remains 2.1–2.4× faster than AdaptiveRAG*, even with the larger model. Fixed-config systems like Parrot* and vLLM fall behind by 7–10% in F1.” “So overall, METIS is quite robust — it keeps its speed and accuracy advantages even when the underlying model or retriever setup changes.”

**Key Takeaways:**
*   (Figure 14) Profiler feedback: Occasionally seeding the profiler with the best-answer output (from a “golden” expensive config) improves F1 by 4–6%. The scheduler still enforces memory constraints, so this doesn’t spiral into always-expensive choices.
*   (Figure 15) Bigger inference LLM (Llama-3.1-70B): METIS still 2.1–2.4× faster than AdaptiveRAG* at similar F1; Parrot*/vLLM fixed configs lag by 7–10% F1.

## Section 4 – Conclusion & Discussion

## What are the key takeaways from METIS? What are some positives of the paper ?

The core strength of METIS lies in unifying two worlds—system scheduling and model quality tuning. It’s compact, practical, and complementary to existing serving optimizations like chunked prefill or KV-cache reuse. The main takeaway: METIS turns RAG from a static pipeline into an intelligent controller that balances quality and latency dynamically—essentially an autopilot for retrieval-augmented generation.   “The core idea behind METIS is simple but powerful — it’s a RAG autopilot. It understands each query, picks the best configuration dynamically, and keeps improving over time.” “Its first strength is being the first system to jointly optimize both RAG quality and delay per query — no prior work has done that systematically.” “Second, METIS’s adaptive configuration + resource-aware scheduling gives it 1.6–2.5× faster responses and 12–18% better F1 scores than state-of-the-art baselines like Parrot* and AdaptiveRAG*.” “It’s lightweight — only about 2K lines of Python code, built on familiar tools like vLLM, FAISS, LangChain, and PyTorch — making it easy to adopt.” “It’s also modular and plug-and-play — can work with any retrieval or serving engine without needing retraining.” “Finally, it’s self-improving — using LLM confidence scores and periodic feedback to refine its profiling over time.” “In short — METIS unifies two worlds: system scheduling and model-level reasoning. It turns RAG from a static pipeline into an intelligent, adaptive controller that automatically balances accuracy and latency.”

**Key Takeaways:**
*   KEY STRENGTHS: -
*   First system to jointly optimize RAG quality and delay per query.
*   Adaptive configuration + resource-aware scheduling → 1.6 – 2.5× faster, +12 – 18 % F1.
*   Lightweight & modular — ~2 K LOC built on vLLM, FAISS, LangChain, PyTorch.
*   Plug-and-play with any retrieval or LLM serving engine.
*   Self-improving profiler — uses confidence & feedback to learn over time.
*   Big idea: METIS acts as a “RAG autopilot” that understands each query, adapts on the fly, and learns continuously.

## What are the current limitations and future directions?

METIS currently excels in classic retrieval→synthesis→answer setups, but future RAG systems are becoming more “agentic,” involving multiple reasoning hops. Extending METIS to coordinate configurations across such stages is an exciting open problem.  Another limitation is that METIS still treats its mapping rules heuristically — a learned or reinforcement-based approach could adapt better. Finally, KV-cache reuse and automatic metadata generation could further cut latency and make METIS plug-and-play for new domains.   “While METIS performs really well in today’s RAG setups, it still has some limitations that open exciting research directions.” “First, it’s designed for standard RAG pipelines — single retrieval and synthesis stages. Future RAG systems are moving toward multi-agent or chain-of-thought pipelines, where multiple reasoning steps or agents collaborate. METIS doesn’t yet handle that level of coordination.” “Second, there’s no KV-cache reuse — which means it doesn’t yet store or blend cached model states across queries. Efficient cache blending could drastically reduce latency for repeated or related queries.” “Third, METIS relies on heuristic mapping — a rule-based system to map LLM profile outputs to configuration knobs. A future learned or reinforcement-based mapper could make it even smarter and more adaptive.” “For future directions, the authors suggest three key paths: 1️⃣ Extend to multi-agent or multi-hop RAG, 2️⃣ Integrate KV-cache blending for reuse, 3️⃣ Auto-generate dataset metadata using LLM summarizers to make it plug-and-play.”

**Key Takeaways:**
*   Designed for standard RAG pipelines — not yet extended to multi-agent or chain-of-thought RAG.
*   No KV-cache reuse — storing blended caches across queries still an open challenge.
*   Heuristic mapping only — profiler + rule-mapping not fully learned or fine-tuned.
*   Future directions:
*   Extend to multi-stage / agentic RAG reasoning.
*   Integrate KV-cache blending for faster reuse.
*   Auto-generate dataset metadata using LLM summarizers.

## Thank you

Concerns: - Across all reviews, the common questions relate to three themes — rule interpretability, profiler cost, and system practicality. METIS addresses these by (1) using an LLM profiler that prunes 50–100× config space with negligible cost; (2) a joint scheduler that adapts to GPU state in real time; and (3) a fallback and feedback loop ensuring reliability. Our follow-up experiments add metadata/no-metadata ablations, profiler size sweeps, fairness and SLO tests, and a learned policy variant. Even under these stricter conditions, METIS continues to deliver 1.6–2.8× lower latency, 1.8–4.5× higher throughput, and ≤ 10 % overhead, proving its practical and general impact on RAG serving systems    Questions: -

**Key Takeaways:**
*   Q and A ?

