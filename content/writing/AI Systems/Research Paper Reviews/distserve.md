---
title: "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving"
date: "2024-03-24"
slug: "distserve"
layout: post
tags: ["AI Systems", "LLM Serving", "Disaggregation", "Systems", "Paper Review"]
---

> **The Core Thesis**: DistServe significantly improves Large Language Model (LLM) serving efficiency by disaggregating the compute-bound prefill phase from the memory-bound decode phase. Through intelligent resource allocation, a bandwidth-aware placement algorithm, and optimized parallelism, it eliminates phase interference, achieving up to 7.4× higher throughput and 12.6× tighter latency Service-Level Objectives (SLOs) compared to existing state-of-the-art systems.

## The Problem: Prefill-Decode Interference

Serving modern LLMs like GPT-4, LLaMA, or Gemini involves two distinct computational phases:
1.  **Prefill Phase**: Processes the entire input prompt simultaneously to generate the first token. This phase is compute-bound, highly parallelizable, and latency-sensitive (optimizing for Time to First Token, or TTFT).
2.  **Decode Phase**: Generates subsequent tokens one by one, autoregressively depending on previous outputs. This phase is memory-bound, sequential, and focuses on generation speed (optimizing for Time Per Output Token, or TPOT).

Existing serving systems, such as vLLM and DeepSpeed-MII, colocate both phases on the same GPU to maximize raw utilization. However, this creates severe inference interference. A compute-heavy prefill job can stall ongoing decoding for other requests, while memory-heavy decoding delays new prefills. This fundamental clash forces service providers to over-provision GPUs to meet strict latency requirements, resulting in poor cost-efficiency and reduced "goodput" (the rate of requests served that actually meet latency SLOs).

## The Solution: Architectural Disaggregation

DistServe proposes a paradigm shift: treating prefill and decoding as separate workloads running on distinct GPU groups. This disaggregation allows each phase to adopt tailored parallelism and batching strategies.

### Analyzing the Trade-offs

*   **Prefill Optimization**: For long prompts, a single request can saturate a GPU. DistServe keeps prefill batches small (grouped by total token length) to avoid unnecessary queuing delays. It uses intra-operator (tensor) parallelism at low request rates for tight latency, and inter-operator (pipeline) parallelism at high rates to reduce queuing.
*   **Decode Optimization**: Because decoding steps are lightweight, they handle larger batch sizes well. Disaggregation allows multiple prefill GPUs to feed Keys/Values (KV) caches into a single decode GPU without blocking new requests.

### Bandwidth-Aware Placement

Transferring the KV cache from the prefill GPU to the decode GPU introduces communication overhead (e.g., up to 1 GB per request for OPT-66B). DistServe uses specific placement algorithms based on the cluster's network topology:
*   **High-Bandwidth Clusters (InfiniBand)**: KV cache transmission time is negligible. DistServe enumerates possible combinations of inter- and intra-operator parallelism, simulates performance, and selects the configuration maximizing the request rate within latency goals.
*   **Low-Bandwidth Clusters (Ethernet/NVLink)**: To mitigate slow cross-node links, DistServe colocates prefill and decode segments corresponding to the same model layers on the same physical node. This localized placement ensures the KV cache is transferred swiftly via fast intra-node links (like NVLink).

### Runtime Orchestration

A central controller routes incoming requests to the prefill GPU with the shortest queue. Upon completing the prefill, the KV cache and initial token are routed to the least-loaded decode GPU. Decode GPUs pull KV caches when ready, preventing memory exhaustion.

## Results and Impact

DistServe was evaluated on a 32-GPU A100 cluster across various workloads (ShareGPT for chatbots, HumanEval for code, LongBench for summarization) using OPT models up to 175B parameters.

*   **Throughput Gains**: Achieved 2×–4.6× higher throughput than vLLM and 1.6×–7.4× higher than DeepSpeed-MII on chatbot workloads.
*   **Latency Improvements**: Delivered up to 12.6× tighter latency SLOs for long-input summarization tasks, ensuring smooth token streaming.
*   **Communication Overhead**: The bandwidth-aware placement kept KV cache transfer latency to under 0.1% of the total latency.

By targeting *goodput* rather than just raw throughput, DistServe proves that decoupling conflicting compute patterns is essential for scalable, SLO-compliant LLM inference.

## My Take on the Paper

DistServe presents a remarkably clean and elegant systems solution to a pervasive bottleneck in LLM deployment. 

**Strengths:**
*   **Pragmatic Design**: Disaggregation is a logical, highly effective concept that addresses the root cause of latency spikes rather than retrofitting workarounds.
*   **Hardware Awareness**: Integrating network topology limits directly into the placement algorithm bridges the gap between theoretical optimization and real-world data center constraints.
*   **Meaningful Metrics**: Focusing on SLO attainment (goodput) is much more relevant to production environments than raw tokens-per-second.

**Weaknesses:**
*   **Infrastructure Requirements**: The architecture inherently assumes multi-GPU clusters, limiting its applicability for single-node or consumer-grade deployments.
*   **System Complexity**: Disaggregation introduces significant complexity—managing KV cache transfers, dynamic GPU group coordination, and complex placement simulations.
*   **Evaluation Scope**: The evaluation relies exclusively on the older OPT model family. Validating against newer architectures (like LLaMA 3, Mistral, or MoE models) and heterogeneous cluster environments would strengthen the claims. Furthermore, fault tolerance during KV cache transfers is largely unaddressed.

## Open Questions

1.  **Dynamic Reallocation**: Could DistServe adapt its resource allocation or seamlessly switch GPU roles (from prefill to decode, and vice versa) online to handle highly variable, spikey traffic patterns common in production chatbot systems?
2.  **Multimodal Serving**: How would a disaggregated architecture adapt to multimodal models (e.g., vision-language), where the "prefill" stage involves diverse encoders with radically different compute, memory, and parallelization characteristics?
3.  **Heterogeneous Topology At Scale**: How resilient is the placement algorithm when deployed across multi-rack clusters where interconnect topologies become highly non-uniform (e.g., mixing PCIe-only nodes with NVLink islands)?