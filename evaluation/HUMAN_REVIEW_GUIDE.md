# Human Review Guide for LLM-as-a-Judge

This guide explains how to manually score the 50 generated replies in `human_judge_sample.csv`.
Your task is to act as the ground-truth human evaluator to see if the LLM judge aligns with human reasoning.

For each row, read the original customer message and the generated reply, then fill in the blank columns (`human_correctness`, `human_groundedness`, etc.) with a number from **1 to 5** based on the following rubric.

## 1. Correctness
Does the reply correctly address the customer's issue without hallucinating facts?
* 1: Completely incorrect or dangerous advice.
* 3: Partially correct but misses the core issue.
* 5: Perfectly addresses the issue with accurate information.

## 2. Groundedness
Is the reply based on historical support evidence, or did the model invent its own policy?
* 1: Completely invented policy, refund, or timeline.
* 3: Mostly generic but not explicitly hallucinated.
* 5: Strongly grounded in standard AppleSupport procedures (e.g., asking for DM, providing official links).

## 3. Helpfulness
Does this actually help the user?
* 1: Actively unhelpful or confusing.
* 3: Polite but generic (e.g., "We are looking into this").
* 5: Highly actionable steps or clear escalation instructions.

## 4. Relevance
Does the response directly connect to the specific issue raised?
* 1: Completely unrelated to the tweet.
* 3: Somewhat related but ignores specific context (e.g., mentions battery when they asked about a cracked screen).
* 5: Directly answers the exact question asked.

## 5. Tone
Is the tone empathetic, polite, and professional?
* 1: Rude, robotic, or dismissive.
* 3: Acceptable but dry.
* 5: Highly empathetic, polite, and on-brand for AppleSupport.

Once you have filled out the CSV, run `python evaluation/calculate_agreement.py` to compare your scores against the LLM's scores.
