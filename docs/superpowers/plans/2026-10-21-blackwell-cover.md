# Blackwell cover worker implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement task by task.

**Goal:** Build and publish a separately tagged cover worker that generates audio on RTX PRO 6000 Blackwell MIG and existing Ampere/Ada GPUs.

**Architecture:** Extend the immutable tested cover image to preserve baked voice assets and Python 3.10/fairseq compatibility. Replace the inference stack with official CUDA 12.8 PyTorch wheels and a matching CUDA 12/cuDNN 9 ONNX Runtime. Adapt only checkpoint loading where newer PyTorch requires it.

**Tech stack:** Python 3.10, torch 2.7.1+cu128, torchaudio 2.7.1+cu128, torchvision 0.22.1+cu128, onnxruntime-gpu 1.22.0, numpy 1.26.4.

**Spec:** User request in conversation: a new cover container supporting PRO 6000, with current API and working older GPUs preserved.

## Constraints and review focus
- Never alter existing image tags; pin the legacy base digest.
- Preserve fixes for bounded MDX windows and original error propagation.
- Keep production Blackwell exclusion until actual hardware acceptance passes.
- Test trusted HuBERT loading against modern PyTorch; do not disable safe loading globally.
- Verify ONNX sessions really execute on CUDA, rather than silently falling back to CPU.
- Exercise built-in/custom voices, long audio and recovery after invalid input.
- Only task-owned build/test resources may change; monitor scarce builder disk space.

## Task 1: Modern runtime and checkpoint compatibility
- [x] Add Dockerfile.blackwell with pinned versions and immutable base.
- [x] Build candidate on test server, limiting temporary storage and resource use.
- [x] Reproduce checkpoint incompatibility under the new runtime with a focused test.
- [x] Implement the smallest scoped checkpoint change; run the complete worker test suite and real model CPU load.

## Task 2: Hardware acceptance and release
- [ ] Publish a uniquely tagged candidate, create isolated Blackwell endpoint (one worker).
- [ ] Full pipeline: built-in/custom voice, short/long audio, invalid input then fresh successful request. Decode and validate output duration, finite samples and non-silence.
- [ ] Repeat representative cases on an existing Ampere/Ada card.
- [ ] Review the branch, record image digest/hardware/results/rollback, push reviewed commits through Git.
- [ ] Stop temporary GPU resources and audio server; report remaining deployment step precisely.

## Decisions and evidence
- Root cause already reproduced in production: torch 2.0.1+cu118 fails at reflect padding/STFT on the Blackwell MIG worker; both reported tasks used the same worker.
- Use the existing clean worker worktree, preserving user edits in the original checkout.
- Runtime rebuild is isolated; publishing a new tag and Git pushes follow the user's existing authorization.

## Validation ledger (2026-10-21)
- Built runtime using official CUDA 12.8 wheels. torch compile flags include sm_120.
- RED: unmodified fairseq/HuBERT loading fails under torch 2.7.1 with weights-only rejection of fairseq.data.dictionary.Dictionary.
- GREEN: explicit baked checkpoint path/loading and restricted custom loading pass actual CPU HuBERT model construction.
- Review found removed np.int in shared f0 quantization and archive directory traversal bypassing checkpoint trust. Observed ten failing regressions under the new Linux runtime, then corrected both and included the fixed pipeline file in the Docker build.
- Complete corrected Linux worker suite: 32 passed; pip check: No broken requirements found.
- Real default startup on a CPU-only builder reaches the preflight and rejects absent CUDA as designed; Linux shell line endings pinned in .gitattributes.
- Independent review of both fixes: no remaining blocking issues.
- Image hardware acceptance remains pending; production image/exclusion unchanged.
