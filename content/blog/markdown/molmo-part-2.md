---
layout: post
title: "Seeing Is the Bottleneck: MOLMO's Image Preprocessing"
date: 2024-10-25
categories: [vlm, multimodal, research]
tags: [MOLMO, PixMo, VLM]
series: "MOLMO and the Anatomy of Modern Vision-Language Models"
part: 2
---

**Part 2 of a 4-part series on Vision-Language Model design.**  
[← Previous](/blog/molmo-part-1) | [Next →](/blog/molmo-part-3)

---

# Seeing Is the Bottleneck: MOLMO's Image Preprocessing

## Why MOLMO's Architecture Is Quietly Radical

At first glance, MOLMO's architecture looks almost conservative. There is no novel transformer variant, no exotic fusion module, and no end-to-end multimodal pretraining trick that fundamentally alters the standard VLM recipe. This is intentional.

MOLMO's architectural contribution is not about *inventing new components*, but about **treating known constraints as immovable facts** and designing around them. In particular, it takes seriously a constraint that many VLMs implicitly ignore:

> Vision Transformers are square, resolution-limited models operating on patchified images—while real-world visual reasoning is neither square nor low-resolution.

Most VLMs implicitly assume that resizing an image to a single fixed resolution is a harmless preprocessing step. MOLMO treats this assumption as false.

---

## The Core Mismatch: ViTs vs. Real Images

Vision Transformers like ViT-L/14 accept inputs of a fixed resolution (e.g., 336×336). This creates an unavoidable trade-off:

* Resize aggressively → preserve global layout, lose fine detail.
* Crop aggressively → preserve detail, lose context.

Most VLMs choose one side of this trade-off implicitly. MOLMO refuses to choose. Instead, it reframes the problem:

> What if the vision encoder sees *multiple coherent views* of the same image, each optimized for a different level of abstraction?

---

## Multi-Scale Tiling as a Representation Strategy

MOLMO's preprocessor produces **two complementary visual representations** from a single image:

1. **A low-resolution global view**
   * The entire image resized to 336×336.
   * Preserves scene-level context, object co-occurrence, and layout.

2. **Multiple high-resolution overlapping crops**
   * Each crop is 336×336.
   * Covers the image on a grid.
   * Overlaps adjacent crops to avoid boundary artifacts.

This is not a data augmentation trick. It is a **deliberate representational decomposition**. Figure 5 from the MOLMO paper illustrates this process:

![Converting an image into tokens. The image is turned into a single low-res and several overlapping high-res crops.](/images/molmo_tokens_sequence.png)

*Figure 5: Image-to-token conversion. The original image (top left) produces a low-resolution global view and multiple high-resolution crops (bottom left). Special tokens mark image start, end, and row boundaries.*

Each crop answers a different question:
* *What is happening overall?*
* *What fine details exist here?*
* *What text or small objects would be lost otherwise?*

---

## Why Overlap Matters (More Than It Seems)

The overlapping region between crops is not incidental. Without overlap, objects near crop boundaries get split, text gets truncated, and spatial continuity breaks. By overlapping crops, MOLMO ensures that **any visually meaningful region appears fully in at least one crop**. This guarantees that the vision encoder never sees "half an object" as its best view.

Figure 3 from the paper demonstrates the difference clearly:

![An image cropped without and with overlap. Overlapping crops ensure that central patches are encoded with neighboring context.](/images/molmo_overlap_comparison.png)

*Figure 3: Overlap vs. no-overlap cropping. Highlighted regions show areas used by the LLM. With overlap, the bike's brand name is always fully visible in at least one crop.*

From a reasoning perspective, this is crucial. The LLM can only reason over *complete visual evidence*, not fragmented patches.

---

## Padding Is Also a Modeling Choice

Real images rarely tile perfectly. MOLMO pads edge crops when needed, but does so explicitly:
* Each patch is tagged as real image, partial padding, or full padding.
* Padding-type embeddings tell the model what is *absence* versus *dark pixels*.

This avoids a subtle but common failure mode where models confuse black padding with visual content—particularly harmful in low-light or nighttime scenes.

---

## The Key Insight

The key insight here is not multi-scale cropping itself, but what it represents: **visual reasoning is scale-sensitive**. A single resolution cannot support both perception and interpretation. MOLMO treats scale as a **first-class axis of representation**, rather than something the model is expected to infer implicitly.

MOLMO reframes where architectural novelty should live: not necessarily in deeper encoders or larger language models, but in **how visual evidence is preserved, structured, and made accessible to reasoning mechanisms**.

This is why MOLMO is better understood as an architectural *correction* rather than an architectural *innovation*. It does not add complexity; it removes implicit assumptions that were never justified.

---

**Part 2 of a 4-part series on Vision-Language Model design.**  
[← Previous](/blog/molmo-part-1) | [Next →](/blog/molmo-part-3)
