Title: MLS  Paper Presentation
Date: 2025-12-12
Category: Research Paper Analysis
Slug: mls--paper-presentation
Summary: Detailed analysis and presentation notes for MLS  Paper Presentation.

<div class="download-box" style="margin-bottom: 2rem; padding: 1rem; background: var(--btn-bg); border-radius: 8px; display: inline-block;">
    <a href="{attach}../raw_content/CS8803-MLS- Paper Presentation.pptx" style="text-decoration: none; font-weight: bold;">
        📥 Download Original Slides (PPTX)
    </a>
</div>

## Blind Backdoors in Deep Learning Models

**Key Takeaways:**
*   Muskan Goyal
*   Nirjhar Deb
*   Vedaang Chopra

## Motivation & Problem

**Key Takeaways:**
*   Traditional backdoor vectors
*   Prior attacks rely on data poisoning, trojaning, or model replacement → all require access to training data or the trained model
*   Supply-chain exposure
*   ML codebases integrate from repos via CI → entry point for malicious commits
*   New blind attack
*   Tamper with loss-computation code, without seeing training data, model, or execution
*   Problem
*   Enables powerful backdoors while evading defenses designed for data/model attacks
*   Blind backdoors target loss computation in the CI pipeline.

## Threat Model

In this attack, the adversary doesn’t have access to the training data, the weights of the model, or even the training logs. Their only leverage is the ability to tamper with the loss-computation code. The paper assumes that the attacker does know the task the model is meant to solve and enough about the training code structure to know where to place the malicious snippet. That gives them just enough of a window to hide the backdoor.  Their goals can be targeted. The attacker can teach the model to perform an entirely different task whenever a trigger appears. The paper demos how a model that normally counts people can be made to identify specific individuals instead.   Finally, the triggers themselves can be very flexible. They might be imperceptible, like a single pixel in an image. They might be perceptible, like a physical object. Or, in the case of text, they can even be completely natural phrases which may occur in the data as is, which makes them even harder to detect.

**Key Takeaways:**
*   Attacker’s capability
*   Can tamper with loss computation code during training (supply-chain / code compromise). No direct access to training data, weights, or training logs.
*   Attacker’s knowledge about the model
*   Knows the problem the model solves and which parts of the training code to change but does not inspect dataset samples or final trained weights.
*   Attacker’s goal
*   Targetted : cause specific outputs/execute specific tasks
*   Can implement multiple targeted behaviors or task-level switches.
*   Perceptibility of trigger
*   Imperceptible (single pixel, tiny patch) or perceptible (physical object, phrase).
*   can use natural language as a trigger, so the model changes behavior when it sees certain words already in the data.

## When the Loss is the Trigger

**Key Takeaways:**
*   Wrap the loss
*   Injects a loss wrapper into training code. When clean loss (𝓁m < 𝑇) → route to backdoor objective
*   Synthetic triggers
*   Computes malicious loss 𝓁m* on attacker-fixed pseudo-inputs 𝜇,𝑣 through same model
*   Balanced objective
*   Uses MGDA to combine losses 𝓁blind = 𝛼0𝓁m + 𝛼1𝓁m* while keeping clean accuracy stable
*   Stealthy training
*   Backprop/optimizer stay unchanged → only the loss path is compromised, so attack remains blind to data and weights
*   Injected loss wrapper mixes clean and backdoor losses (MGDA) in training code.

## Main Contributions

**Key Takeaways:**
*   Limitation of prior work
*   Backdoors like BadNets (Gu et al., 2017) and model-replacement/trojaning assume access to training data or model weights → unrealistic in supply-chain settings
*   New contribution
*   Proposes the first blind, loss-computation backdoor attack, requiring only tampered training code
*   Stronger attacks
*   Demonstrates single-pixel, physical, semantic, and covert task-switching backdoors, even multiple triggers in one model, all while preserving main accuracy
*   Defense gap
*   Shows evasion of methods like Neural Cleanse (Wang et al., 2019), highlighting the need for new defenses (e.g., trusted computational graphs)

## Weaknesses - Methodology & Experiments

The core technique relies heavily on MGDA, which requires computing extra gradients every batch. That’s fine for smaller models, but it raises questions about whether it would scale to today’s massive training jobs, like large language models trained on hundreds of GPUs. The patched loss code runs the model on extra inputs and computes extra gradients for the backdoor loss. That’s an extra forward and backward pass per input which doubles the compute for those samples. In theory, this could be visible because training would slow down or GPUs would show higher utilization. But the paper doesn’t actually measure whether these overheads could be detected in practice. This also means that practically the attacker can only run it for some batches or samples which in turn raises questions about the effectiveness of the attack. Another limitation is the range of triggers they tested. They show the attack with a pixel, a patch, and one phrase, but they don’t test a wide variety of triggers or study how robust the attack would be under natural noise, transformations, or distribution shifts. Finally, while they do run ImageNet and RoBERTa experiments, the most interesting novelty — covert task-switching, like making the model do sums or multiplications — is only demonstrated on toy datasets like MultiMNIST. That leaves an open question of whether these complex backdoors would really scale up to large, real-world models. So, while the attack looks powerful in controlled settings, it’s not fully proven in terms of scalability or robustness.”

**Key Takeaways:**
*   Relies on MGDA balancing - costly at scale
*   Extra forward/backward work - detectable in practice since training systems can flag persistent increases or spikes in runtime, GPU utilization etc
*   Not tested with a wide variety of triggers - Paper only shows a few examples (single pixel, patch, a phrase). We don’t know how well the attack holds up across many possible triggers or under natural variation.
*   Limited evaluation on large models
*   Covert task-switching only shown on toy datasets - Novel backdoors like hidden extra tasks (sum/multiply) are proven on MultiMNIST, but not tested on large, real-world models

## Weaknesses - Broader Issues

First, the attack still depends on being able to access and modify the loss-computation code. That’s realistic in open supply chains, but in tightly controlled training setups, it might not be feasible. Alongside that, the attacker also needs to know enough about the training API to insert their code in exactly the right place, which narrows who can actually carry out this attack. Ethics is another area that feels under addressed. Semantic backdoors that flip sentiment based on a person’s name are powerful, but also raise serious risks of bias or targeted misuse. The paper doesn’t really dig into those implications. Finally, the presentation itself could be clearer. The math behind MGDA and Frank-Wolfe is written in a dense, technical style, with very little intuitive explanation. I know I struggled to get through those

**Key Takeaways:**
*   Requires access to loss code - realistic but not universal
*   Requires knowledge of training API - attacker must know where to insert code.
*   Ethical discussion is shallow
*   Dense explanations - MGDA + Frank-Wolfe math is hard to follow, little intuition

## Practical Examples

## Does this have a Git Repo ? 

**Key Takeaways:**
*   Yes -> https://github.com/ebagdasa/backdoors101

## Current and Future Practical Work

**Key Takeaways:**
*   Backdoors: - Pixel-pattern (incl. single-pixel) - traditional pixel modification attacks; Physical - attacks that are triggered by physical objects.; Semantic backdoors - attacks that don't modify the input (e.g. react on features already present in the scene).
*   TODO clean-label (good place to contribute).
*   Injection methods: -Data poisoning - adds backdoors into the dataset; Batch poisoning - injects backdoor samples directly into the batch during training; Loss poisoning - modifies the loss value during training (supports dynamic loss balancing, see Sec 3.4 )
*   TODO: model poisoning (good place to contribute!).
*   Datasets: -Image Classification - ImageNet, CIFAR-10, Pipa face identification, MultiMNIST, MNIST; Text - IMDB reviews datasets, Reddit (coming)
*   TODO: Face recognition, eg Celeba or VGG. We already have some code, but need expertise on producing good models (good place to contribute!).
*   Defenses:- Input perturbation - NeuralCleanse + added evasion; Model anomalies - SentiNet + added evasion; Spectral clustering / fine-pruning + added evasion.
*   TODO: Port Jupyter notebooks demonstrating defenses and evasions. Add new defenses and evasions (good place to contribute!).

## Q&A

